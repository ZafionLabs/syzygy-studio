"""A local patch of a planet: elevation for the shape, and a map image for the color."""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

from .heightmap import Heightmap

SPAN_KM = 720.0
SAMPLES = 96


def _window(radius_km: float, lat: float, span_km: float) -> tuple[float, float, float, float]:
    """Return half-spans in degrees plus the matching north and east distances in km."""
    km_per_deg = math.pi * radius_km / 180.0
    half_north = min(70.0, (span_km / 2.0) / km_per_deg)
    cos = max(0.25, math.cos(math.radians(lat)))
    half_east = min(70.0, (span_km / 2.0) / (km_per_deg * cos))
    north_km = half_north * 2.0 * km_per_deg
    east_km = half_east * 2.0 * km_per_deg * cos
    return half_north, half_east, north_km, east_km


def _bilinear(hm: Heightmap, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    height, width = hm.shape
    lat_min, lat_max = hm.lat_range
    lon_min, lon_max = hm.lon_range
    lons = np.asarray(lons, dtype=np.float64)
    if hm.wraps_longitude:
        lons = (lons - lon_min) % 360.0 + lon_min
    else:
        lons = np.clip(lons, lon_min, lon_max)
    row_f = np.clip((lat_max - np.asarray(lats, dtype=np.float64)) / (lat_max - lat_min) * height - 0.5, 0, height - 1.001)
    col_f = np.clip((lons - lon_min) / (lon_max - lon_min) * width - 0.5, 0, width - 1.001)
    row0 = np.floor(row_f).astype(int)
    col0 = np.floor(col_f).astype(int)
    row1 = np.clip(row0 + 1, 0, height - 1)
    col1 = np.clip(col0 + 1, 0, width - 1)
    row_t = row_f - row0
    col_t = col_f - col0
    values = hm.elevation
    v00 = values[row0, col0]
    v01 = values[row0, col1]
    v10 = values[row1, col0]
    v11 = values[row1, col1]
    mixed = v00 * (1 - row_t) * (1 - col_t) + v01 * (1 - row_t) * col_t + v10 * row_t * (1 - col_t) + v11 * row_t * col_t
    return np.where(np.isfinite(mixed), mixed, 0.0)


def ground_grid(hm: Heightmap, lat: float, lon: float, span_km: float = SPAN_KM, samples: int = SAMPLES) -> dict:
    """Elevation samples around one place, north at the first row and west at the first column."""
    lat = float(np.clip(lat, -90.0, 90.0))
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    span_km = float(np.clip(span_km, 40.0, 2000.0))
    samples = int(np.clip(samples, 8, 160))
    half_north, half_east, north_km, east_km = _window(hm.body.mean_radius_km, lat, span_km)
    lats = np.linspace(lat + half_north, lat - half_north, samples)
    lons = np.linspace(lon - half_east, lon + half_east, samples)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    elevation = np.rint(_bilinear(hm, lat_grid, lon_grid)).astype(int)
    return {
        "body": hm.body.name.lower(),
        "lat": lat,
        "lon": lon,
        "rows": samples,
        "cols": samples,
        "north_km": north_km,
        "east_km": east_km,
        "radius_km": hm.body.mean_radius_km,
        "elevation_m": elevation.ravel().tolist(),
    }


def _crop(image: Image.Image, lat: float, lon: float, half_north: float, half_east: float) -> Image.Image:
    width, height = image.size

    def row_of(value: float) -> float:
        return (90.0 - value) / 180.0 * height

    def col_of(value: float) -> float:
        return ((value + 180.0) % 360.0) / 360.0 * width

    top = row_of(lat + half_north)
    bottom = row_of(lat - half_north)
    left = col_of(lon - half_east)
    right = col_of(lon + half_east)
    if bottom <= top:
        bottom = min(height, top + 1)
    if right > left:
        cropped = image.crop((left, top, right, bottom))
    else:
        west = image.crop((left, top, width, bottom))
        east = image.crop((0, top, right, bottom))
        cropped = Image.new("RGB", (west.width + east.width, max(west.height, 1)))
        cropped.paste(west, (0, 0))
        cropped.paste(east, (west.width, 0))
    return cropped.resize((512, 512), Image.Resampling.BILINEAR)


def ground_png(raw: bytes, lat: float, lon: float, radius_km: float, span_km: float = SPAN_KM) -> bytes:
    """The same window as :func:`ground_grid`, taken from an equirectangular map."""
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    lat = float(np.clip(lat, -90.0, 90.0))
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    half_north, half_east, _, _ = _window(radius_km, lat, span_km)
    out = io.BytesIO()
    _crop(image, lat, lon, half_north, half_east).save(out, format="PNG")
    return out.getvalue()
