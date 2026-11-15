"""A local patch of a planet: elevation for the shape, and a map image for the color."""

from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .heightmap import Heightmap

SPAN_KM = 100.0
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
        "low_m": hm.body.min_elevation_m,
        "high_m": hm.body.max_elevation_m,
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


TILE = 256
_SATELLITE: dict[tuple[int, int, int], np.ndarray] = {}


def _zoom(half_east: float, pixels: int = 512) -> int:
    span = max(half_east * 2.0, 1e-4)
    level = math.log2(pixels * 360.0 / span / TILE)
    return int(np.clip(round(level), 2, 16))


def _mercator(lat: np.ndarray, lon: np.ndarray, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    scale = TILE * (1 << zoom)
    lat = np.clip(np.asarray(lat, dtype=np.float64), -85.05112878, 85.05112878)
    lon = np.asarray(lon, dtype=np.float64)
    x = (lon + 180.0) / 360.0 * scale
    lat_rad = np.deg2rad(lat)
    y = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * scale
    return x, y


def _google_tile(zoom: int, x: int, y: int) -> np.ndarray | None:
    key = (zoom, x, y)
    cached = _SATELLITE.get(key)
    if cached is not None:
        return cached
    url = f"https://mt{(x + y) % 4}.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"
    try:
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=8) as response:
            raw = response.read()
        tile = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB").resize((TILE, TILE)))
    except (OSError, ValueError):
        return None
    if len(_SATELLITE) > 256:
        _SATELLITE.clear()
    _SATELLITE[key] = tile
    return tile


def satellite_png(
    lat: float,
    lon: float,
    radius_km: float,
    span_km: float = SPAN_KM,
    fetch=None,
) -> bytes | None:
    """Google Maps satellite photo of the same window as :func:`ground_grid`."""
    lat = float(np.clip(lat, -90.0, 90.0))
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    half_north, half_east, _, _ = _window(radius_km, lat, span_km)
    zoom = _zoom(half_east)
    size = 512
    rows = np.linspace(lat + half_north, lat - half_north, size)
    cols = np.linspace(lon - half_east, lon + half_east, size)
    lon_grid, lat_grid = np.meshgrid(cols, rows)
    xs, ys = _mercator(lat_grid, lon_grid, zoom)
    x0 = int(np.floor(xs.min()))
    y0 = int(np.floor(ys.min()))
    x1 = int(np.ceil(xs.max()))
    y1 = int(np.ceil(ys.max()))
    tiles_x = 1 << zoom
    fetch = fetch or _google_tile
    wanted = []
    for ty in range(y0 // TILE, y1 // TILE + 1):
        if ty < 0 or ty >= tiles_x:
            continue
        for tx in range(x0 // TILE, x1 // TILE + 1):
            wanted.append((tx, ty))
    tiles: dict[tuple[int, int], np.ndarray] = {}
    if fetch is _google_tile:
        with ThreadPoolExecutor(max_workers=8) as pool:
            loaded = list(pool.map(lambda item: ((item[0], item[1]), fetch(zoom, item[0] % tiles_x, item[1])), wanted))
    else:
        loaded = [((tx, ty), fetch(zoom, tx % tiles_x, ty)) for tx, ty in wanted]
    for (tx, ty), tile in loaded:
        if tile is not None:
            tiles[(tx, ty)] = np.asarray(tile)
    if not tiles:
        return None
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    canvas = np.zeros((height, width, 3), np.uint8)
    for (tx, ty), tile in tiles.items():
        tile = np.asarray(Image.fromarray(tile).convert("RGB").resize((TILE, TILE)))
        left = tx * TILE - x0
        top = ty * TILE - y0
        src_x = max(0, -left)
        src_y = max(0, -top)
        dst_x = max(0, left)
        dst_y = max(0, top)
        copy_w = min(TILE - src_x, width - dst_x)
        copy_h = min(TILE - src_y, height - dst_y)
        if copy_w > 0 and copy_h > 0:
            canvas[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = tile[src_y:src_y + copy_h, src_x:src_x + copy_w]
    ix = np.clip(xs - x0, 0, width - 1.001)
    iy = np.clip(ys - y0, 0, height - 1.001)
    x_i = np.floor(ix).astype(int)
    y_i = np.floor(iy).astype(int)
    x_j = np.clip(x_i + 1, 0, width - 1)
    y_j = np.clip(y_i + 1, 0, height - 1)
    tx = (ix - x_i)[..., None]
    ty = (iy - y_i)[..., None]
    sampled = (
        canvas[y_i, x_i] * (1 - ty) * (1 - tx)
        + canvas[y_i, x_j] * (1 - ty) * tx
        + canvas[y_j, x_i] * ty * (1 - tx)
        + canvas[y_j, x_j] * ty * tx
    )
    out = io.BytesIO()
    Image.fromarray(sampled.astype(np.uint8), "RGB").save(out, format="PNG")
    return out.getvalue()


def ground_png(raw: bytes, lat: float, lon: float, radius_km: float, span_km: float = SPAN_KM) -> bytes:
    """The same window as :func:`ground_grid`, taken from an equirectangular map."""
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    lat = float(np.clip(lat, -90.0, 90.0))
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    half_north, half_east, _, _ = _window(radius_km, lat, span_km)
    out = io.BytesIO()
    _crop(image, lat, lon, half_north, half_east).save(out, format="PNG")
    return out.getvalue()
