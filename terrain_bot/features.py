"""Per-pixel terrain descriptors measured at fixed *physical* scales.

Earth and Moon maps have very different ground resolution (an equirectangular
pixel spans ~3.7x more kilometres on Earth), so neighbourhood statistics are
computed over windows defined in kilometres, with east-west windows widened by
1/cos(latitude) to follow the true ground distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .heightmap import Heightmap

DEFAULT_SCALES_KM = (75.0, 250.0)
MIN_COVERAGE = 0.5


@dataclass
class FeatureSet:
    names: list[str]
    values: np.ndarray  # (rows, cols, n_features)
    valid: np.ndarray  # (rows, cols)
    scales_km: tuple[float, ...]

    def at(self, row: int, col: int) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.values[row, col])}


def _cos_lat(hm: Heightmap) -> np.ndarray:
    return np.clip(np.cos(np.radians(hm.lats)), 0.02, 1.0)


def gaussian_smooth_km(hm: Heightmap, field: np.ndarray, weight: np.ndarray, sigma_km: float):
    """NaN-aware Gaussian smoothing with sigma given in ground kilometres.

    Returns ``(smoothed, coverage)`` where coverage is the fraction of the
    kernel that landed on valid pixels.
    """
    h, w = field.shape
    km_ns, km_ew = hm.km_per_px
    x_mode = "wrap" if hm.wraps_longitude else "nearest"

    num = np.where(weight > 0, field, 0.0) * weight
    den = weight.astype(np.float64)

    sy = sigma_km / km_ns
    if sy > 1e-3:
        num = gaussian_filter1d(num, sy, axis=0, mode="nearest")
        den = gaussian_filter1d(den, sy, axis=0, mode="nearest")

    cos_lat = _cos_lat(hm)
    out_num = np.empty_like(num)
    out_den = np.empty_like(den)
    for i in range(h):
        sx = min(sigma_km / (km_ew * cos_lat[i]), w / 2.0)
        if sx > 1e-3:
            out_num[i] = gaussian_filter1d(num[i], sx, mode=x_mode)
            out_den[i] = gaussian_filter1d(den[i], sx, mode=x_mode)
        else:
            out_num[i], out_den[i] = num[i], den[i]

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(out_den > 1e-9, out_num / out_den, np.nan)
    return smoothed, out_den


def slope_degrees(hm: Heightmap, elevation: np.ndarray) -> np.ndarray:
    km_ns, km_ew = hm.km_per_px
    dz_dy = np.gradient(elevation, axis=0) / (km_ns * 1000.0)
    if hm.wraps_longitude:
        dz_dx_px = (np.roll(elevation, -1, axis=1) - np.roll(elevation, 1, axis=1)) / 2.0
    else:
        dz_dx_px = np.gradient(elevation, axis=1)
    dz_dx = dz_dx_px / (km_ew * 1000.0 * _cos_lat(hm)[:, None])
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))


def compute_features(hm: Heightmap, scales_km=DEFAULT_SCALES_KM) -> FeatureSet:
    """Build the descriptor stack used for similarity matching.

    Features (all in real units):
      * ``elevation_m``           calibrated elevation of the pixel
      * ``mean_<s>km_m``          average elevation of the surrounding terrain
      * ``relief_<s>km_m``        local roughness (std-dev of elevation)
      * ``slope_<s0>km_deg``      regional slope at the finest scale
    """
    scales_km = tuple(float(s) for s in scales_km)
    weight = hm.valid.astype(np.float64)
    z = np.where(hm.valid, hm.elevation, 0.0)

    names = ["elevation_m"]
    layers = [hm.elevation]
    valid = hm.valid.copy()

    finest_mean = None
    for s in scales_km:
        mean, cov = gaussian_smooth_km(hm, z, weight, s)
        mean_sq, _ = gaussian_smooth_km(hm, z * z, weight, s)
        relief = np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None))
        valid &= cov >= MIN_COVERAGE
        names += [f"mean_{s:g}km_m", f"relief_{s:g}km_m"]
        layers += [mean, relief]
        if finest_mean is None:
            finest_mean = mean

    names.append(f"slope_{scales_km[0]:g}km_deg")
    layers.append(slope_degrees(hm, finest_mean))

    values = np.stack(layers, axis=-1)
    valid &= np.all(np.isfinite(values), axis=-1)
    return FeatureSet(names=names, values=values, valid=valid, scales_km=scales_km)
