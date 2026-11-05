"""Load grayscale heightmap images and calibrate them to real elevations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .bodies import Body


def load_gray(path: str | Path, alpha_threshold: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(gray, valid)`` for an image file.

    ``gray`` is a float array of brightness values in the image's native range
    (8-bit or 16-bit). Pixels whose alpha is below ``alpha_threshold`` (out of
    255) are marked invalid so transparent borders are not treated as terrain.
    """
    img = Image.open(path)
    valid = None

    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        gray = np.asarray(img, dtype=np.float64)
    else:
        if "A" in img.getbands() or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            valid = np.asarray(rgba)[..., 3] >= alpha_threshold
            img = rgba
        gray = np.asarray(img.convert("L"), dtype=np.float64)

    if valid is None:
        valid = np.ones(gray.shape, dtype=bool)
    return gray, valid


@dataclass
class Heightmap:
    """An elevation grid in an equirectangular (plate carree) projection.

    ``source`` is ``"image"`` for brightness calibrated to the body's real
    extremes, or ``"dem"`` when ``elevation`` comes straight from a measured
    elevation model (``gray`` then holds the raw DEM values).
    """

    body: Body
    gray: np.ndarray
    valid: np.ndarray
    elevation: np.ndarray
    lat_range: tuple[float, float] = (-90.0, 90.0)
    lon_range: tuple[float, float] = (-180.0, 180.0)
    gray_min: float = 0.0
    gray_max: float = 255.0
    source: str = "image"
    source_name: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    @property
    def lats(self) -> np.ndarray:
        """Latitude of each row's pixel centre (north at row 0)."""
        h = self.shape[0]
        lat_min, lat_max = self.lat_range
        return lat_max - (np.arange(h) + 0.5) * (lat_max - lat_min) / h

    @property
    def lons(self) -> np.ndarray:
        w = self.shape[1]
        lon_min, lon_max = self.lon_range
        return lon_min + (np.arange(w) + 0.5) * (lon_max - lon_min) / w

    @property
    def wraps_longitude(self) -> bool:
        return abs((self.lon_range[1] - self.lon_range[0]) - 360.0) < 1e-6

    @property
    def deg_per_px(self) -> tuple[float, float]:
        h, w = self.shape
        return (
            (self.lat_range[1] - self.lat_range[0]) / h,
            (self.lon_range[1] - self.lon_range[0]) / w,
        )

    @property
    def km_per_deg(self) -> float:
        return np.pi * self.body.mean_radius_km / 180.0

    @property
    def km_per_px(self) -> tuple[float, float]:
        """(north-south, east-west at the equator) pixel size in km."""
        dlat, dlon = self.deg_per_px
        return dlat * self.km_per_deg, dlon * self.km_per_deg

    def pixel_of(self, lat: float, lon: float) -> tuple[int, int]:
        h, w = self.shape
        lat_min, lat_max = self.lat_range
        lon_min, lon_max = self.lon_range
        if not lat_min <= lat <= lat_max:
            raise ValueError(f"latitude {lat} is outside the map ({lat_min} to {lat_max})")
        if self.wraps_longitude:
            lon = (lon - lon_min) % 360.0 + lon_min
        elif not lon_min <= lon <= lon_max:
            raise ValueError(f"longitude {lon} is outside the map ({lon_min} to {lon_max})")
        row = int(np.floor((lat_max - lat) / (lat_max - lat_min) * h))
        col = int(np.floor((lon - lon_min) / (lon_max - lon_min) * w))
        return int(np.clip(row, 0, h - 1)), int(np.clip(col, 0, w - 1))

    def pixels_of(self, lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised :meth:`pixel_of`; out-of-map coordinates are clipped to the edge."""
        h, w = self.shape
        lat_min, lat_max = self.lat_range
        lon_min, lon_max = self.lon_range
        lons = np.asarray(lons, dtype=np.float64)
        if self.wraps_longitude:
            lons = (lons - lon_min) % 360.0 + lon_min
        rows = np.floor((lat_max - np.asarray(lats)) / (lat_max - lat_min) * h).astype(int)
        cols = np.floor((lons - lon_min) / (lon_max - lon_min) * w).astype(int)
        return np.clip(rows, 0, h - 1), np.clip(cols, 0, w - 1)

    def latlon_of(self, row: int, col: int) -> tuple[float, float]:
        return float(self.lats[row]), float(self.lons[col])

    def elevation_at(self, lat: float, lon: float) -> float:
        return float(self.elevation[self.pixel_of(lat, lon)])

    def extreme_pixels(self) -> dict[str, tuple[int, int]]:
        elev = np.where(self.valid, self.elevation, np.nan)
        hi = np.unravel_index(np.nanargmax(elev), elev.shape)
        lo = np.unravel_index(np.nanargmin(elev), elev.shape)
        return {"highest": (int(hi[0]), int(hi[1])), "lowest": (int(lo[0]), int(lo[1]))}


def calibrate(
    gray: np.ndarray,
    valid: np.ndarray,
    body: Body,
    clip_percentiles: tuple[float, float] = (0.0, 100.0),
    gamma: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Linearly map brightness to elevation.

    The darkest valid pixel becomes ``body.min_elevation_m`` and the lightest
    becomes ``body.max_elevation_m``. ``clip_percentiles`` can be tightened
    (e.g. ``(0.01, 99.99)``) so a few noisy pixels don't define the extremes;
    ``gamma`` != 1 applies a power curve for maps that were not encoded linearly.
    """
    if not valid.any():
        raise ValueError("Heightmap has no valid (opaque) pixels")
    values = gray[valid]
    g_lo, g_hi = np.percentile(values, clip_percentiles)
    if g_hi <= g_lo:
        raise ValueError("Heightmap has no brightness variation; cannot calibrate")

    t = np.clip((gray - g_lo) / (g_hi - g_lo), 0.0, 1.0) ** gamma
    elevation = body.min_elevation_m + t * body.relief_m
    elevation = np.where(valid, elevation, np.nan)
    return elevation, float(g_lo), float(g_hi)


def read_sidecar(path: str | Path) -> dict:
    """Optional ``<image>.json`` next to a map, e.g. ``{"lat_range": [-77, 84]}``."""
    side = Path(path).with_suffix(".json")
    return json.loads(side.read_text()) if side.exists() else {}


def load_heightmap(
    path: str | Path,
    body: Body,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    clip_percentiles: tuple[float, float] = (0.0, 100.0),
    gamma: float = 1.0,
    alpha_threshold: int = 128,
) -> Heightmap:
    """Load an image heightmap; missing bounds come from a sidecar JSON or default to the whole globe."""
    meta = read_sidecar(path)
    lat_range = tuple(lat_range or meta.get("lat_range", (-90.0, 90.0)))
    lon_range = tuple(lon_range or meta.get("lon_range", (-180.0, 180.0)))
    gray, valid = load_gray(path, alpha_threshold=alpha_threshold)
    hm = heightmap_from_array(gray, body, valid, lat_range, lon_range, clip_percentiles, gamma)
    hm.source_name = f"image {Path(path).name}"
    return hm


def heightmap_from_array(
    gray: np.ndarray,
    body: Body,
    valid: np.ndarray | None = None,
    lat_range: tuple[float, float] = (-90.0, 90.0),
    lon_range: tuple[float, float] = (-180.0, 180.0),
    clip_percentiles: tuple[float, float] = (0.0, 100.0),
    gamma: float = 1.0,
) -> Heightmap:
    gray = np.asarray(gray, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(gray)
    elevation, g_lo, g_hi = calibrate(gray, valid, body, clip_percentiles, gamma)
    return Heightmap(
        body=body,
        gray=gray,
        valid=valid,
        elevation=elevation,
        lat_range=tuple(lat_range),
        lon_range=tuple(lon_range),
        gray_min=g_lo,
        gray_max=g_hi,
    )


def haversine_km(lat1, lon1, lat2, lon2, radius_km: float):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def heightmap_from_dem(
    elevation: np.ndarray,
    body: Body,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    source_name: str,
) -> Heightmap:
    elevation = np.asarray(elevation, dtype=np.float64)
    valid = np.isfinite(elevation)
    return Heightmap(
        body=body,
        gray=elevation,
        valid=valid,
        elevation=elevation,
        lat_range=tuple(lat_range),
        lon_range=tuple(lon_range),
        gray_min=float(np.nanmin(elevation)),
        gray_max=float(np.nanmax(elevation)),
        source="dem",
        source_name=source_name,
    )
