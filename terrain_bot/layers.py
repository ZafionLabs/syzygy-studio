"""Extra science maps (temperature, ice, geology, weather, light, radiation)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .bodies import get_body

LAYER_DIR = Path(__file__).resolve().parents[1] / "data" / "layers"
_arrays: dict[tuple[str, str], np.ndarray] = {}


def catalog() -> dict:
    return json.loads((LAYER_DIR / "catalog.json").read_text())


def layer_path(body: str, layer: str) -> Path:
    from .updates import layer_file

    custom = layer_file(body, layer)
    path = custom or (LAYER_DIR / f"{body}_{layer}.png")
    if not path.is_file():
        raise ValueError(f"no {layer} map for {body}")
    return path


def layer_png(body: str, layer: str) -> bytes:
    return layer_path(body, layer).read_bytes()


def _rgb(raw: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def _layer_array(body: str, layer: str, overrides: dict[tuple[str, str], bytes] | None = None) -> np.ndarray | None:
    custom = (overrides or {}).get((body, layer))
    if custom is not None:
        return _rgb(custom)
    key = (body, layer)
    if key not in _arrays:
        try:
            _arrays[key] = _rgb(layer_png(body, layer))
        except ValueError:
            return None
    return _arrays[key]


def _pixel(image: np.ndarray, lat: float, lon: float) -> np.ndarray:
    height, width = image.shape[:2]
    lat = float(np.clip(lat, -90.0, 90.0))
    lon = (float(lon) + 180.0) % 360.0 - 180.0
    row = min(height - 1, int((90.0 - lat) / 180.0 * height))
    col = min(width - 1, int((lon + 180.0) / 360.0 * width))
    return image[row, col]


def _color_percent(left: np.ndarray, right: np.ndarray) -> float:
    distance = float(np.linalg.norm(left.astype(np.float64) - right.astype(np.float64)))
    return round(100.0 * float(np.clip(1.0 - distance / (255.0 * np.sqrt(3.0)), 0.0, 1.0)), 1)


def compare_maps(
    body_a: str,
    lat_a: float,
    lon_a: float,
    body_b: str,
    lat_b: float,
    lon_b: float,
    elevation_a: float,
    elevation_b: float,
    overrides: dict[tuple[str, str], bytes] | None = None,
) -> list[dict]:
    """How closely the two places agree on every map, from 0 to 100."""
    rows = []
    if np.isfinite(elevation_a) and np.isfinite(elevation_b):
        left = get_body(body_a)
        right = get_body(body_b)
        unit_a = float(np.clip((elevation_a - left.min_elevation_m) / left.relief_m, 0.0, 1.0))
        unit_b = float(np.clip((elevation_b - right.min_elevation_m) / right.relief_m, 0.0, 1.0))
        rows.append({"id": "elevation", "label": "Elevation", "percent": round(100.0 * (1.0 - abs(unit_a - unit_b)), 1)})
    else:
        rows.append({"id": "elevation", "label": "Elevation", "percent": None})
    info = catalog()
    for item in info["layers"]:
        layer = item["id"]
        if layer == "elevation":
            continue
        available = info["bodies"].get(body_a, {}).get(layer, {}).get("available") and info["bodies"].get(body_b, {}).get(layer, {}).get("available")
        if not available:
            rows.append({"id": layer, "label": item["label"], "percent": None})
            continue
        left = _layer_array(body_a, layer, overrides)
        right = _layer_array(body_b, layer, overrides)
        if left is None or right is None:
            rows.append({"id": layer, "label": item["label"], "percent": None})
            continue
        rows.append({
            "id": layer,
            "label": item["label"],
            "percent": _color_percent(_pixel(left, lat_a, lon_a), _pixel(right, lat_b, lon_b)),
        })
    return rows
