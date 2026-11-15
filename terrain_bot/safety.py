"""Landing-safety grids for Earth, the Moon, and Mars.

Each cell carries four scores from 0 to 1:

* hazard — slope, nearby relief, and, on Earth, open water
* sunlight — share of the year the ground is lit, from latitude and slope
* shadow — steep ground and polar night; on the Moon, ground that stays
  cold enough for ice in permanent shadow
* radiation — the radiation map already on that body (midnight heat on the
  Moon, fast neutrons on Mars, heat leaving the top of the atmosphere on Earth)

A higher score on hazard, shadow, or radiation is less safe. A higher sunlight
score is safer. The web map turns a weighted mix of the four into green for
safer ground and red for less safe ground.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from .heightmap import Heightmap
from .layers import LAYER_DIR

FACTORS = ("hazard", "sunlight", "shadow", "radiation")


@dataclass
class SafetyGrid:
    hazard: np.ndarray
    sunlight: np.ndarray
    shadow: np.ndarray
    radiation: np.ndarray
    slope_deg: np.ndarray
    roughness_m: np.ndarray
    water: np.ndarray


def _wrap_mean(values: np.ndarray, size: int) -> np.ndarray:
    """Box mean. Longitude wraps; the polar rows wrap too, which only affects those rows."""
    return uniform_filter(values, size=size, mode="wrap")


def _layer_index(body: str, layer: str, shape: tuple[int, int]) -> np.ndarray | None:
    path = LAYER_DIR / f"{body}_{layer}.png"
    if not path.is_file():
        return None
    image = Image.open(path).convert("RGB").resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    lo, hi = np.percentile(lum, (2, 98))
    if hi <= lo:
        return np.zeros(shape, dtype=np.float32)
    return np.clip((lum - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _slope_deg(hm: Heightmap) -> np.ndarray:
    elev = np.where(hm.valid, hm.elevation, 0.0)
    km_ns, km_ew = hm.km_per_px
    dy = max(km_ns, 1e-3) * 1000.0
    cos_lat = np.clip(np.cos(np.radians(hm.lats)), 0.15, 1.0)
    dx = np.maximum(km_ew * 1000.0 * cos_lat, 1.0)
    wrapped = np.concatenate([elev[:, -1:], elev, elev[:, :1]], axis=1)
    dz_dx = (wrapped[:, 2:] - wrapped[:, :-2]) / (2.0 * dx[:, None])
    dz_dy = np.gradient(elev, dy, axis=0)
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)


def build_safety(hm: Heightmap) -> SafetyGrid:
    """Scores for every pixel of a heightmap. Arrays are float32 in 0..1 except slope and roughness."""
    slope = _slope_deg(hm)
    km_ns, _km_ew = hm.km_per_px
    window = max(3, int(round(75.0 / max(km_ns, 1.0))) | 1)
    window = min(window, max(3, (min(hm.shape) // 5) | 1))
    elev = np.where(hm.valid, hm.elevation, 0.0).astype(np.float64)
    mean = _wrap_mean(elev, window)
    mean_sq = _wrap_mean(elev * elev, window)
    roughness = np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None)).astype(np.float32)

    slope_bad = np.clip(slope / 18.0, 0.0, 1.0)
    rough_bad = np.clip(roughness / 400.0, 0.0, 1.0)
    hazard = np.maximum(slope_bad, 0.85 * rough_bad)
    water = np.zeros(hm.shape, dtype=bool)
    if hm.body.name == "Earth":
        water = hm.valid & (hm.elevation < 0)
        hazard = np.where(water, np.maximum(hazard, 0.92), hazard)
    hazard = np.where(hm.valid, hazard, 1.0).astype(np.float32)

    daylight = np.clip(np.cos(np.radians(hm.lats)), 0.0, 1.0)[:, None]
    facing = np.clip(np.cos(np.radians(np.clip(slope, 0.0, 80.0))), 0.0, 1.0)
    sunlight = (daylight * (0.4 + 0.6 * facing)).astype(np.float32)
    sunlight = np.where(hm.valid, sunlight, 0.0)

    steep_shadow = np.clip(np.sin(np.radians(np.clip(slope, 0.0, 90.0))), 0.0, 1.0)
    polar = np.clip((np.abs(hm.lats)[:, None] - 55.0) / 35.0, 0.0, 1.0)
    shadow = np.clip(0.75 * steep_shadow + 0.55 * polar, 0.0, 1.0)
    body_key = hm.body.name.lower()
    if body_key == "moon":
        cold = _layer_index(body_key, "ice-water", hm.shape)
        if cold is not None:
            shadow = np.maximum(shadow, cold * 0.85)
    shadow = np.where(hm.valid, np.clip(shadow, 0.0, 1.0), 1.0).astype(np.float32)

    radiation = _layer_index(body_key, "radiation", hm.shape)
    if radiation is None:
        radiation = np.full(hm.shape, 0.5, dtype=np.float32)
    radiation = np.where(hm.valid, radiation, 1.0).astype(np.float32)

    return SafetyGrid(hazard, sunlight, shadow, radiation, slope, roughness, water)


def landing_safety(grid_or_values: SafetyGrid | dict, weights: dict[str, float], row: int | None = None, col: int | None = None) -> float | None:
    """0–100 safety for the given priority weights. ``None`` when every weight is zero."""
    weights = {name: max(0.0, float(weights.get(name, 0.0))) for name in FACTORS}
    total = sum(weights.values())
    if total <= 0:
        return None
    if isinstance(grid_or_values, SafetyGrid):
        if row is None or col is None:
            raise ValueError("row and col are required")
        values = {
            "hazard": float(grid_or_values.hazard[row, col]),
            "sunlight": float(grid_or_values.sunlight[row, col]),
            "shadow": float(grid_or_values.shadow[row, col]),
            "radiation": float(grid_or_values.radiation[row, col]),
        }
    else:
        values = {name: float(grid_or_values[name]) for name in FACTORS}
    bad = (
        weights["hazard"] * values["hazard"]
        + weights["sunlight"] * (1.0 - values["sunlight"])
        + weights["shadow"] * values["shadow"]
        + weights["radiation"] * values["radiation"]
    ) / total
    return float((1.0 - bad) * 100.0)


def radiation_line(body: str, radiation: float) -> str:
    pct = radiation * 100.0
    if body == "Moon":
        return f"Midnight heat radiation is {pct:.0f}% of the way from the coldest ground to the warmest rock."
    if body == "Mars":
        return f"Fast-neutron radiation from cosmic rays is {pct:.0f}% of the way from the lowest flux on Mars to the highest."
    return f"Heat leaving the top of the atmosphere is {pct:.0f}% of the way from the lowest to the highest."


def danger_lines(body: str, slope: float, roughness: float, sunlight: float, shadow: float, radiation: float, water: bool) -> list[str]:
    lines = []
    if water:
        lines.append("Open water covers this spot, so there is no dry ground to land on.")
    if slope >= 15:
        lines.append(f"The slope is {slope:.0f}°. That is steep enough to tip a lander.")
    elif slope >= 8:
        lines.append(f"The slope is {slope:.0f}°. The ground is tilted.")
    else:
        lines.append(f"The slope is {slope:.0f}°. The ground is nearly flat.")
    if roughness >= 250:
        lines.append(f"Nearby terrain rises and falls by about {roughness:.0f} m.")
    lines.append(f"Sunlight reaches this ground for about {sunlight * 100:.0f}% of the year.")
    if sunlight < 0.25:
        lines.append("Solar power is scarce here.")
    lines.append(f"Shadows cover about {shadow * 100:.0f}% of the nearby ground.")
    lines.append(radiation_line(body, radiation))
    return lines


def encode_png(grid: SafetyGrid) -> bytes:
    """RGBA image. Red, green, and blue are hazard, sunlight, and shadow. Alpha stores radiation from 1 to 255."""
    rgba = np.dstack([
        np.clip(np.round(grid.hazard * 255.0), 0, 255),
        np.clip(np.round(grid.sunlight * 255.0), 0, 255),
        np.clip(np.round(grid.shadow * 255.0), 0, 255),
        np.clip(np.round(1.0 + grid.radiation * 254.0), 1, 255),
    ]).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(out, format="PNG", optimize=True)
    return out.getvalue()


def describe_cell(hm: Heightmap, grid: SafetyGrid, lat: float, lon: float, weights: dict[str, float] | None = None) -> dict:
    row, col = hm.pixel_of(lat, lon)
    hazard = float(grid.hazard[row, col])
    sunlight = float(grid.sunlight[row, col])
    shadow = float(grid.shadow[row, col])
    radiation = float(grid.radiation[row, col])
    slope = float(grid.slope_deg[row, col])
    roughness = float(grid.roughness_m[row, col])
    water = bool(grid.water[row, col])
    weights = weights or {"hazard": 25, "sunlight": 25, "shadow": 25, "radiation": 25}
    return {
        "hazard": hazard,
        "sunlight": sunlight,
        "shadow": shadow,
        "radiation": radiation,
        "slope_deg": slope,
        "roughness_m": roughness,
        "water": water,
        "safety": landing_safety(
            {"hazard": hazard, "sunlight": sunlight, "shadow": shadow, "radiation": radiation},
            weights,
        ),
        "dangers": danger_lines(hm.body.name, slope, roughness, sunlight, shadow, radiation, water),
        "radiation_note": {
            "Moon": "Midnight heat the ground radiates. Rock that stays warm scores higher.",
            "Mars": "Fast neutrons made when cosmic rays hit the surface. A thinner sky lets more neutrons out.",
            "Earth": "Heat leaving the top of the atmosphere. Deserts and the tropics score higher.",
        }.get(hm.body.name, "Radiation measured for this body."),
    }
