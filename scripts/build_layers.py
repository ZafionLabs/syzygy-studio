"""Build the extra Earth, Moon, and Mars map layers from public science products.

The products kept here are NASA GIBS (Earth) and the USGS Moon and Mars mosaics.
The other Moon and Mars maps are built by scripts/replace_layers.py.
Run from the repo root. Cached downloads live in /tmp/layers.
"""

from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "layers"
CACHE = Path("/tmp/layers")
UA = {"User-Agent": "terrain-bot/1.0 (educational)"}
W, H = 1440, 720


def fetch(url: str, name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    return dest


def gibs(layer: str, time: str, name: str, width: int = W, height: int = H) -> Image.Image:
    url = (
        "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
        f"?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS={layer}"
        f"&FORMAT=image/jpeg&CRS=EPSG:4326&BBOX=-90,-180,90,180"
        f"&WIDTH={width}&HEIGHT={height}&TIME={time}"
    )
    return Image.open(fetch(url, name)).convert("RGB")


def usgs(mapfile: str, layer: str, name: str) -> Image.Image:
    url = (
        "https://planetarymaps.usgs.gov/cgi-bin/mapserv"
        f"?map={mapfile}&service=WMS&request=GetMap&version=1.1.1"
        f"&layers={layer}&styles=&bbox=-180,-90,180,90&width={W}&height={H}"
        "&srs=EPSG:4326&format=image/jpeg&transparent=false"
    )
    return Image.open(fetch(url, name)).convert("RGB")


def save(image: Image.Image, name: str) -> None:
    image = image.resize((W, H), Image.Resampling.BILINEAR) if image.size != (W, H) else image
    image.save(OUT / name, format="PNG", optimize=True)


def colorize(values: np.ndarray, valid: np.ndarray, low: float, high: float) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    t = np.clip((values - low) / (high - low), 0, 1)
    rgba = colormaps["inferno"](np.nan_to_num(t, nan=0.0))
    rgba[..., 3] = np.where(valid, 1.0, 0.0)
    return Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")


def crustal_thickness() -> np.ndarray:
    blob = fetch("https://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz", "crust1.0.tar.gz").read_bytes()
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        text = archive.extractfile("crust1.bnds").read().decode()
    bounds = np.array([[float(part) for part in line.split()[:9]] for line in text.splitlines() if line.strip()])
    thickness = np.zeros(len(bounds))
    for index in (1, 2, 3, 4, 5, 6, 7):
        thickness += -(bounds[:, index + 1] - bounds[:, index])
    grid = thickness.reshape(180, 360)
    return np.array(Image.fromarray(grid.astype(np.float32), "F").resize((W, H), Image.Resampling.NEAREST))


def earth_ice(snow: Image.Image, sea_ice: Image.Image) -> Image.Image:
    dem = np.load(ROOT / "data" / "dem" / "earth_etopo1_0p25deg.npz")["elevation"]
    if dem.shape != (H, W):
        dem = np.array(Image.fromarray(dem.astype(np.float32), "F").resize((W, H), Image.Resampling.BILINEAR))
    water = dem < 0
    base = np.zeros((H, W, 3), np.uint8)
    base[water] = (14, 48, 92)
    base[~water] = (92, 74, 48)
    snow_a = np.asarray(snow.resize((W, H)))
    ice_a = np.asarray(sea_ice.resize((W, H)))
    snowy = snow_a.mean(axis=2) > 28
    icy = ice_a.mean(axis=2) > 28
    base[snowy] = snow_a[snowy]
    base[icy] = ice_a[icy]
    return Image.fromarray(base, "RGB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    save(gibs("MERRA2_2m_Air_Temperature_Monthly", "2024-07-01", "earth_temperature.jpg"), "earth_temperature.png")
    save(gibs("BlueMarble_NextGeneration", "2004-08-01", "earth_bluemarble.jpg"), "earth_light.png")
    save(gibs("IMERG_Precipitation_Rate", "2024-07-15", "earth_weather.jpg"), "earth_weather.png")
    save(gibs("CERES_EBAF_TOA_Longwave_Flux_All_Sky_Monthly", "2018-07-01", "earth_longwave.jpg"), "earth_radiation.png")
    snow = gibs("MODIS_Terra_NDSI_Snow_Cover", "2024-01-15", "earth_snow.jpg")
    sea_ice = gibs("AMSRU2_Sea_Ice_Concentration_25km", "2024-03-01", "earth_ice.jpg")
    save(earth_ice(snow, sea_ice), "earth_ice-water.png")
    save(usgs("/maps/earth/moon_simp_cyl.map", "LROC_WAC", "moon_light.jpg"), "moon_light.png")
    save(usgs("/maps/mars/mars_simp_cyl.map", "MDIM21_color", "mars_light.jpg"), "mars_light.png")
    print("Earth maps and the Moon and Mars mosaics are written.")
    print("The other Moon and Mars maps, and the source list, come from scripts/replace_layers.py.")


def _crust_pair() -> tuple[np.ndarray, np.ndarray]:
    values = crustal_thickness()
    return values, np.isfinite(values)



if __name__ == "__main__":
    main()
