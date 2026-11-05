"""Measured elevation models (real data) and validation of image heightmaps against them.

* Earth: NOAA ETOPO1 1 arc-minute global relief (land topography + ocean
  bathymetry, ice-sheet surface), sampled every 0.25 degrees via NOAA ERDDAP.
* Moon: NASA LRO LOLA gridded data record LDEM_4 (4 pixels/degree), heights
  relative to the 1737.4 km reference sphere.
* Mars: NASA MGS MOLA mission experiment gridded data record MEGDR (4
  pixels/degree), heights relative to the MOLA areoid.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .bodies import BODIES
from .heightmap import Heightmap, heightmap_from_dem

DEM_DIR = Path(__file__).resolve().parent.parent / "data" / "dem"


@dataclass(frozen=True)
class DemSource:
    body: str
    name: str
    filename: str
    url: str
    citation: str


DEM_SOURCES = {
    "earth": DemSource(
        body="earth",
        name="NOAA ETOPO1 (ice surface), sampled every 0.25 deg",
        filename="earth_etopo1_0p25deg.npz",
        url=(
            "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?"
            "altitude%5B(-90):15:(90)%5D%5B(-180):15:(180)%5D"
        ),
        citation=(
            "Amante, C. and B. W. Eakins, ETOPO1 1 Arc-Minute Global Relief Model, "
            "NOAA Technical Memorandum NESDIS NGDC-24, 2009"
        ),
    ),
    "moon": DemSource(
        body="moon",
        name="NASA LRO LOLA LDEM_4 (4 px/deg)",
        filename="moon_lola_ldem4.npz",
        url=(
            "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/"
            "data/lola_gdr/cylindrical/img/ldem_4.img"
        ),
        citation="Smith, D. E. et al., LRO LOLA Gridded Data Record LDEM_4, NASA PDS, 2017",
    ),
    "mars": DemSource(
        body="mars",
        name="NASA MGS MOLA MEGDR (4 px/deg)",
        filename="mars_mola_megdr4.npz",
        url=(
            "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/"
            "meg004/megt90n000cb.img"
        ),
        citation="Smith, D. E. et al., MGS MOLA Mission Experiment Gridded Data Record, NASA PDS, 2003",
    ),
}


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "terrain-bot"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def _earth_from_netcdf(blob: bytes) -> tuple[np.ndarray, tuple, tuple]:
    from scipy.io import netcdf_file

    nc = netcdf_file(io.BytesIO(blob), "r", mmap=False)
    lat = np.asarray(nc.variables["latitude"][:], dtype=np.float64)
    elev = np.asarray(nc.variables["altitude"][:], dtype=np.int16)
    nc.close()
    if lat[0] < lat[-1]:
        elev = elev[::-1]
    # Node-registered grid: drop the duplicated 180 deg column and express the
    # nodes as cell centres so the equirectangular helpers line up exactly.
    elev = elev[:, :-1]
    step = 0.25
    return elev, (-90.0 - step / 2, 90.0 + step / 2), (-180.0 - step / 2, 180.0 - step / 2)


def _moon_from_img(blob: bytes) -> tuple[np.ndarray, tuple, tuple]:
    dn = np.frombuffer(blob, dtype="<i2").reshape(720, 1440)
    metres = np.round(dn.astype(np.float64) * 0.5).astype(np.int16)
    # LDEM columns start at 0 deg E; roll so the grid runs from -180 to 180.
    return np.roll(metres, 720, axis=1), (-90.0, 90.0), (-180.0, 180.0)


def _mars_from_img(blob: bytes) -> tuple[np.ndarray, tuple, tuple]:
    metres = np.frombuffer(blob, dtype=">i2").reshape(720, 1440).astype(np.int16)
    # MEGDR columns start at 0 deg E; roll so the grid runs from -180 to 180.
    return np.roll(metres, 720, axis=1), (-90.0, 90.0), (-180.0, 180.0)


_PARSERS = {"earth": _earth_from_netcdf, "moon": _moon_from_img, "mars": _mars_from_img}


def download_dem(body: str, dest_dir: Path = DEM_DIR) -> Path:
    src = DEM_SOURCES[body]
    blob = _fetch(src.url)
    elev, lat_range, lon_range = _PARSERS[body](blob)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / src.filename
    np.savez_compressed(path, elevation=elev, lat_range=lat_range, lon_range=lon_range)
    return path


def dem_available(body: str, dem_dir: Path = DEM_DIR) -> bool:
    return (dem_dir / DEM_SOURCES[body].filename).exists()


def load_dem(body: str, dem_dir: Path = DEM_DIR) -> Heightmap:
    src = DEM_SOURCES[body]
    path = dem_dir / src.filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m terrain_bot download-dem`")
    with np.load(path) as f:
        elev = f["elevation"]
        lat_range = tuple(float(x) for x in f["lat_range"])
        lon_range = tuple(float(x) for x in f["lon_range"])
    return heightmap_from_dem(elev, BODIES[body], lat_range, lon_range, src.name)


def _isotonic(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: the closest non-decreasing sequence to ``y``."""
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi))
        wts.append(float(wi))
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wsum = wts[-2] + wts[-1]
            merged = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wsum
            size = sizes[-2] + sizes[-1]
            del vals[-1], wts[-1], sizes[-1]
            vals[-1], wts[-1], sizes[-1] = merged, wsum, size
    return np.repeat(vals, sizes)


def fit_image_to_dem(image_hm: Heightmap, dem_hm: Heightmap) -> Heightmap:
    """Re-calibrate an image with a brightness -> elevation curve learned from measured data.

    For every brightness level the median measured elevation of the pixels with
    that brightness is taken, then the curve is forced to rise with brightness.
    This undoes clipping and non-linear encodings, but cannot fix an image whose
    brightness isn't height (shaded relief).
    """
    rows, cols = np.nonzero(image_hm.valid)
    gray = np.round(image_hm.gray[rows, cols])
    real = dem_hm.elevation[dem_hm.pixels_of(image_hm.lats[rows], image_hm.lons[cols])]
    ok = np.isfinite(real)
    gray, real = gray[ok], real[ok]

    levels, inverse, counts = np.unique(gray, return_inverse=True, return_counts=True)
    medians = np.array([np.median(real[inverse == i]) for i in range(levels.size)])
    fitted = _isotonic(medians, counts)

    elevation = np.where(image_hm.valid, np.interp(image_hm.gray, levels, fitted), np.nan)
    return Heightmap(
        body=image_hm.body,
        gray=image_hm.gray,
        valid=image_hm.valid,
        elevation=elevation,
        lat_range=image_hm.lat_range,
        lon_range=image_hm.lon_range,
        gray_min=image_hm.gray_min,
        gray_max=image_hm.gray_max,
        source="image-fit",
        source_name=f"{image_hm.source_name} fitted to {dem_hm.source_name.split(' (')[0]}",
        lut=(levels, fitted),
    )


def compare_to_dem(image_hm: Heightmap, dem_hm: Heightmap) -> dict:
    """How well a calibrated image matches measured elevations at the same coordinates."""
    rows, cols = np.nonzero(image_hm.valid)
    lats, lons = image_hm.lats[rows], image_hm.lons[cols]
    real = dem_hm.elevation[dem_hm.pixels_of(lats, lons)]
    est = image_hm.elevation[rows, cols]
    ok = np.isfinite(real) & np.isfinite(est)
    real, est = real[ok], est[ok]
    w = np.cos(np.radians(lats[ok]))
    err = est - real

    out = {
        "body": image_hm.body.name,
        "image": image_hm.source_name,
        "reference": dem_hm.source_name,
        "pixels_compared": int(ok.sum()),
        "spearman": float(spearmanr(est, real)[0]),
        "rmse_m": float(np.sqrt(np.average(err**2, weights=w))),
        "median_abs_error_m": float(np.median(np.abs(err))),
        "bias_m": float(np.average(err, weights=w)),
    }
    if image_hm.body.name == "Earth":
        real_wet, est_wet = real < 0, est < 0
        out["underwater_fraction_real"] = float(np.average(real_wet, weights=w))
        out["underwater_fraction_image"] = float(np.average(est_wet, weights=w))
        out["land_water_agreement"] = float(np.average(real_wet == est_wet, weights=w))
    return out


def format_comparison(c: dict) -> str:
    lines = [
        f"{c['body']}: {c['image']} vs {c['reference']} ({c['pixels_compared']:,} pixels)",
        f"  rank correlation {c['spearman']:.3f}, RMSE {c['rmse_m']:,.0f} m, "
        f"median error {c['median_abs_error_m']:,.0f} m, bias {c['bias_m']:+,.0f} m",
    ]
    if "land_water_agreement" in c:
        lines.append(
            f"  underwater: {c['underwater_fraction_real']:.0%} in reality vs {c['underwater_fraction_image']:.0%} "
            f"in the image; land/water agrees on {c['land_water_agreement']:.0%} of the surface"
        )
    return "\n".join(lines)
