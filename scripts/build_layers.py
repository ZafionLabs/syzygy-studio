"""Build the extra Earth, Moon, and Mars map layers from public science products.

The products are NASA GIBS (Earth), USGS Astrogeology WMS (Moon and Mars mosaics),
LRO Diviner browse images, MGS TES albedo and thermal inertia, and CRUST1.0.
Run from the repo root. Cached downloads live in /tmp/layers.
"""

from __future__ import annotations

import io
import json
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


def diviner_globe(name: str) -> Image.Image:
    """Crop the map frame (±60° latitude) out of a Diviner browse figure and pad the poles."""
    image = Image.open(CACHE / name).convert("RGB")
    frame = image.crop((174, 102, 3168, 1096))
    band = frame.resize((W, 480), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (W, H), (22, 24, 28))
    canvas.paste(band, (0, 120))
    return canvas


def tes_grids() -> tuple[np.ndarray, np.ndarray]:
    albedo = np.frombuffer(fetch(
        "https://pds-geosciences.wustl.edu/mgs/mgs-m-tes-special-v1/global_albedo_8ppd.img",
        "albedo.img",
    ).read_bytes(), dtype="<f4").reshape(1440, 2880)
    thermal = np.frombuffer(fetch(
        "https://pds-geosciences.wustl.edu/mgs/mgs-m-tes-special-v1/global_ti_60n_50s_8ppd.img",
        "ti.img",
    ).read_bytes(), dtype="<f4").reshape(882, 2880)
    albedo = albedo.reshape(360, 4, 720, 4).mean(axis=(1, 3))
    thermal_small = thermal.reshape(882, 720, 4).mean(axis=2)
    return albedo, thermal_small


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
    save(colorize(*_crust_pair(), 5, 70), "earth_geology.png")

    save(usgs("/maps/earth/moon_simp_cyl.map", "LROC_WAC", "moon_light.jpg"), "moon_light.png")
    for src, dest in (
        ("moon_temperature.jpg", "moon_temperature.png"),
        ("moon_geology.jpg", "moon_geology.png"),
    ):
        fetch(
            "https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/browse/gdr_l3/cylindrical/jpg/"
            + ("dgdr_st_avg_cyl_032_jpg.jpg" if "temperature" in src else "dgdr_ra_avg_cyl_032_jpg.jpg"),
            src,
        )
        save(diviner_globe(src), dest)

    save(usgs("/maps/mars/mars_simp_cyl.map", "MDIM21_color", "mars_light.jpg"), "mars_light.png")
    themis = np.array(usgs("/maps/mars/mars_simp_cyl.map", "THEMIS_night", "mars_temperature.jpg"))
    themis[themis.min(axis=2) > 245] = (22, 24, 28)
    save(Image.fromarray(themis, "RGB"), "mars_temperature.png")

    albedo, thermal = tes_grids()
    raw = np.frombuffer((CACHE / "albedo.img").read_bytes(), dtype="<f4").reshape(1440, 2880)
    albedo_full = raw.reshape(H, 1440 // H, W, 2880 // W).max(axis=(1, 3))
    lat = 90.0 - (np.arange(H) + 0.5) * (180.0 / H)
    frost = (albedo_full >= 0.28) & (np.abs(lat)[:, None] >= 65.0)
    ice = np.full((H, W, 3), (92, 52, 42), np.uint8)
    ice[frost] = (245, 240, 220)
    save(Image.fromarray(ice, "RGB"), "mars_ice-water.png")
    thermal_full = np.full((H, W), np.nan)
    # 882 lines at 8 px/deg span 60.125 N to 50.125 S. The globe is 4 px/deg.
    top = int(round((90.0 - 60.125) * (H / 180.0)))
    band = int(round((60.125 - (-50.125)) * (H / 180.0)))
    thermal_full[top:top + band] = _fit_rows(thermal, band)
    valid = np.isfinite(thermal_full) & (thermal_full > 0)
    save(colorize(np.nan_to_num(thermal_full), valid, 50, 400).convert("RGB"), "mars_geology.png")

    (OUT / "catalog.json").write_text(json.dumps(CATALOG, indent=2) + "\n")
    print("wrote", len(list(OUT.glob('*.png'))), "maps")


def _crust_pair() -> tuple[np.ndarray, np.ndarray]:
    values = crustal_thickness()
    return values, np.isfinite(values)


def _fit_rows(values: np.ndarray, rows: int) -> np.ndarray:
    image = Image.fromarray(values.astype(np.float32), "F").resize((W, rows), Image.Resampling.BILINEAR)
    return np.array(image)


CATALOG = {
    "layers": [
        {"id": "elevation", "label": "Elevation"},
        {"id": "temperature", "label": "Temperature"},
        {"id": "ice-water", "label": "Ice and water"},
        {"id": "geology", "label": "Geology"},
        {"id": "weather", "label": "Weather"},
        {"id": "light", "label": "Light"},
        {"id": "radiation", "label": "Radiation"},
    ],
    "bodies": {
        "earth": {
            "temperature": {
                "available": True,
                "title": "Air temperature, 2 m",
                "source": "NASA GIBS / MERRA-2 monthly, July 2024",
                "low": "colder",
                "high": "warmer",
                "note": "July 2024 monthly air temperature about 2 m above the surface. Yellow and red are warmer; blue and purple are colder.",
            },
            "ice-water": {
                "available": True,
                "title": "Ocean, sea ice, and snow",
                "source": "ETOPO1 plus AMSR2 sea ice (March 2024) and MODIS snow (15 Jan 2024)",
                "low": "open water",
                "high": "ice or snow",
                "note": "Blue is ocean and brown is land, from ETOPO1. Sea ice is the March 2024 AMSR2 concentration. Snow is the 15 January 2024 MODIS snow cover, including the Antarctic ice sheet.",
            },
            "geology": {
                "available": True,
                "title": "Crustal thickness",
                "source": "CRUST1.0, Laske, Masters, Ma, and Pasyanos 2013",
                "low": "thinner crust",
                "high": "thicker crust",
                "note": "Thickness of ice, sediment, and crystalline crust, with the ocean water left out. Ocean crust is thin; continents and mountains are thick.",
            },
            "weather": {
                "available": True,
                "title": "Rain and snow rate",
                "source": "NASA GIBS / IMERG, 15 July 2024",
                "low": "little precipitation",
                "high": "heavy precipitation",
                "note": "One day of NASA IMERG precipitation. Green is lighter rain and red is heavier. Dark areas had little or no precipitation that day.",
            },
            "light": {
                "available": True,
                "title": "Blue Marble",
                "source": "NASA Blue Marble Next Generation, August 2004",
                "low": "darker surface",
                "high": "brighter surface",
                "note": "Cloud-free view of reflected sunlight. Ice and desert are bright; forest and ocean are darker.",
            },
            "radiation": {
                "available": True,
                "title": "Outgoing longwave radiation",
                "source": "NASA CERES EBAF, July 2018",
                "low": "less heat leaving",
                "high": "more heat leaving",
                "note": "Heat radiation leaving the top of the atmosphere in July 2018. Deserts and the tropics radiate more; Antarctica radiates less.",
            },
        },
        "moon": {
            "temperature": {
                "available": True,
                "title": "Regolith temperature variation",
                "source": "LRO Diviner average, PDS browse dgdr_st_avg",
                "low": "smaller variation",
                "high": "larger variation",
                "note": "Diviner average regolith temperature variation from 60°S to 60°N. The poles are outside this browse image, so they are left dark.",
            },
            "ice-water": {
                "available": False,
                "note": "There is no separate global ice map in this set. Water ice can persist in the coldest polar shadows, and those latitudes are outside the Diviner browse image.",
            },
            "geology": {
                "available": True,
                "title": "Rock abundance",
                "source": "LRO Diviner average rock abundance, PDS browse dgdr_ra_avg",
                "low": "fewer rocks",
                "high": "more rocks",
                "note": "Fraction of the surface covered by exposed rock, from 60°S to 60°N. Fresh craters show up as small bright spots. The poles are outside this browse image.",
            },
            "weather": {
                "available": False,
                "note": "The Moon has no atmosphere, so it has no weather.",
            },
            "light": {
                "available": True,
                "title": "Reflected sunlight",
                "source": "USGS / LRO WAC global mosaic",
                "low": "darker maria",
                "high": "brighter highlands",
                "note": "Lunar Reconnaissance Orbiter Wide Angle Camera mosaic. The dark maria are basaltic lava; the highlands are brighter.",
            },
            "radiation": {
                "available": False,
                "note": "No global surface radiation-dose map is included. The Moon has no global magnetic field, so cosmic rays reach the surface almost everywhere.",
            },
        },
        "mars": {
            "temperature": {
                "available": True,
                "title": "Nighttime infrared",
                "source": "USGS / Mars Odyssey THEMIS IR night mosaic",
                "low": "cooler at night",
                "high": "warmer at night",
                "note": "THEMIS nighttime infrared. Rock stays warmer after sunset than dust does. The polar regions are not in this mosaic, so they are left dark.",
            },
            "ice-water": {
                "available": True,
                "title": "Frost and ice",
                "source": "MGS TES bolometric albedo, Christensen 2001",
                "low": "bare ground",
                "high": "frost or ice",
                "note": "TES bolometric albedo of 0.28 or higher, kept only poleward of 65°. That leaves the polar caps and leaves out the bright tropical dust. Bare ground is dark.",
            },
            "geology": {
                "available": True,
                "title": "Thermal inertia",
                "source": "MGS TES thermal inertia, Mellon et al., 60°N to 50°S",
                "low": "dust",
                "high": "rock or crust",
                "note": "How quickly the ground takes up and releases heat, in J per square metre per kelvin per square-root second. Dust is low; rock and duricrust are high. The far north and south are outside this map.",
            },
            "weather": {
                "available": False,
                "note": "No global daily weather image is included. Dust storms are the main weather on Mars.",
            },
            "light": {
                "available": True,
                "title": "Viking color mosaic",
                "source": "USGS / Viking MDIM 2.1 color mosaic",
                "low": "darker surface",
                "high": "brighter surface",
                "note": "Viking orbiter color mosaic. The bright spots at the poles are ice. Hellas is the bright basin in the south.",
            },
            "radiation": {
                "available": False,
                "note": "No global surface radiation-dose map is included. Mars Odyssey and Curiosity measured radiation along their paths, not as a full-planet map.",
            },
        },
    },
}


if __name__ == "__main__":
    main()
