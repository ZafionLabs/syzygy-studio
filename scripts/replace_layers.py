"""Replace partial Moon and Mars maps with full-globe measurements.

Earth's temperature, ice, geology, weather, light, and radiation maps are left
as they are. This rewrites the Moon and Mars maps that were cropped figures or
stand-ins, fills the empty panels from the same public archives, and writes
data/layers/catalog.json with a citation for every map.

Cached downloads live in /tmp/layers. Run from the repo root after the Earth
maps already exist:

    python3 scripts/replace_layers.py
"""

from __future__ import annotations

import json
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "layers"
CACHE = Path("/tmp/layers")
UA = {"User-Agent": "terrain-bot/1.0 (educational)"}
W, H = 1440, 720
SIGMA = 5.670374419e-8


def fetch(url: str, name: str, weak_tls: bool = False) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    context = None
    if weak_tls:
        context = ssl.create_default_context()
        context.set_ciphers("DEFAULT:@SECLEVEL=1")
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300, context=context) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    return dest


def colorize(values: np.ndarray, valid: np.ndarray, low: float, high: float) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    span = high - low if high > low else 1.0
    t = np.clip((values - low) / span, 0, 1)
    rgb = (colormaps["inferno"](np.nan_to_num(t, nan=0.0))[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = (22, 24, 28)
    return Image.fromarray(rgb, "RGB")


def limits(values: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    sample = values[valid]
    return float(np.percentile(sample, 2)), float(np.percentile(sample, 98))


def save(image: Image.Image, name: str) -> None:
    if image.size != (W, H):
        image = image.resize((W, H), Image.Resampling.BILINEAR)
    image.save(OUT / name, format="PNG", optimize=True)


def upscale_nearest(values: np.ndarray) -> np.ndarray:
    image = Image.fromarray(values.astype(np.float32), "F")
    return np.array(image.resize((W, H), Image.Resampling.NEAREST))


def diviner_grid(name: str) -> np.ndarray:
    """0.5° Diviner raster. File order is not assumed; longitude and latitude are."""
    lon, lat, val = np.loadtxt(CACHE / name, unpack=True)
    grid = np.full((360, 720), np.nan)
    iy = np.clip(np.floor((90.0 - lat) / 0.5).astype(int), 0, 359)
    ix = np.clip(np.floor((lon + 180.0) / 0.5).astype(int), 0, 719)
    grid[iy, ix] = val
    return upscale_nearest(grid)


def neutron_grid(name: str) -> np.ndarray:
    """Odyssey NS table: longitude 0–360° E, latitude, flux, at 0.5°."""
    lon, lat, val = np.loadtxt(CACHE / name, delimiter=",", unpack=True)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    grid = np.full((360, 720), np.nan)
    # Exact zero is the unmeasured polar rows. Fast-neutron tables use -100 for the same gap.
    good = np.isfinite(val) & (val > 0)
    iy = np.clip(np.floor((90.0 - lat) / 0.5).astype(int), 0, 359)
    ix = np.clip(np.floor((lon + 180.0) / 0.5).astype(int), 0, 719)
    grid[iy[good], ix[good]] = val[good]
    return upscale_nearest(grid)


def tes_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Afternoon surface temperature and dust opacity for every Mars Year 25 season."""
    lon_l, lat_l, dust_l, temp_l = [], [], [], []
    for path in sorted(CACHE.glob("tes_my25_*.dat")):
        with path.open() as handle:
            next(handle)
            for line in handle:
                parts = line.split()
                if len(parts) < 11:
                    continue
                lon = float(parts[3])
                if lon > 180.0:
                    lon -= 360.0
                lon_l.append(lon)
                lat_l.append(float(parts[4]))
                dust_l.append(float(parts[7]))
                temp_l.append(float(parts[10]))
    return (
        np.array(lon_l),
        np.array(lat_l),
        np.array(temp_l),
        np.array(dust_l),
    )


def bin_mean(lon: np.ndarray, lat: np.ndarray, val: np.ndarray, valid: np.ndarray, step: float = 5.0) -> np.ndarray:
    nlat, nlon = int(180 / step), int(360 / step)
    acc = np.zeros((nlat, nlon))
    count = np.zeros((nlat, nlon))
    iy = np.clip(np.floor((90.0 - lat[valid]) / step).astype(int), 0, nlat - 1)
    ix = np.clip(np.floor((lon[valid] + 180.0) / step).astype(int), 0, nlon - 1)
    np.add.at(acc, (iy, ix), val[valid])
    np.add.at(count, (iy, ix), 1)
    out = np.full((nlat, nlon), np.nan)
    mask = count > 0
    out[mask] = acc[mask] / count[mask]
    return upscale_nearest(out)


def thermal_inertia() -> np.ndarray:
    """Putzig 2007 nightside map, 20 pixels/degree, simple cylindrical, north at the top."""
    raw = np.fromfile(CACHE / "global_ti_night_2007.img", dtype=">i2").reshape(3600, 7200).astype(np.float32)
    blocks = raw.reshape(H, 5, W, 5)
    valid = blocks > 0
    count = valid.sum(axis=(1, 3))
    total = np.where(valid, blocks, 0).sum(axis=(1, 3))
    out = np.full((H, W), np.nan)
    good = count > 0
    out[good] = total[good] / count[good]
    return out


def moon_geology() -> Image.Image:
    archive = CACHE / "moon_geology.zip"
    member = (
        "Unified_Geologic_Map_of_the_Moon_RASTER/graphical_versions/"
        "Unified_Geologic_Map_of_the_Moon_Geology_32ppd.tif"
    )
    dest = CACHE / "moon_geology.tif"
    if not dest.exists():
        with zipfile.ZipFile(archive) as bundle:
            dest.write_bytes(bundle.read(member))
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(dest).convert("RGB")
    # Nearest keeps each rock unit's color. Averaging would blend the legend together.
    return image.resize((W, H), Image.Resampling.NEAREST)


def download() -> None:
    base = "http://luna1.diviner.ucla.edu/~jpierre/diviner/level4_raster_data/"
    for name in ("diviner_tbol_max.xyz", "diviner_tbol_min.xyz", "diviner_tbol_hour00.xyz"):
        fetch(base + name, name, weak_tls=True)
    fetch(
        "https://pds-geosciences.wustl.edu/ody/urn-nasa-pds-odyssey_grs_special/data/ns_epithermal_020917.tab",
        "ns_epithermal.tab",
    )
    fetch(
        "https://pds-geosciences.wustl.edu/ody/urn-nasa-pds-odyssey_grs_special/data/ns_fast_020917.tab",
        "ns_fast.tab",
    )
    fetch(
        "https://repository.hou.usra.edu/bitstreams/9f33fbfa-f748-479c-8d9b-0e3fd8f36ace/download",
        "moon_geology.zip",
    )
    root = "https://atmos.nmsu.edu/PDS/data/PDS4/mgs_tes_atmos_dust-ice/data_IR_Aerosol_Optical_Depth/MY25/"
    for start in range(0, 360, 30):
        end = start + 30
        fetch(
            f"{root}TES_COD_IR_MY25_Ls{start:03d}_Ls{end:03d}.dat",
            f"tes_my25_{start:03d}.dat",
        )
    fetch(
        "https://pds-geosciences.wustl.edu/mgs/mgs-m-tes-5-timap-v1/mgst_9001/data/global_ti_night_2007.img",
        "global_ti_night_2007.img",
    )


def report(name: str, values: np.ndarray, valid: np.ndarray) -> None:
    sample = values[valid]
    print(
        f"{name}: valid {valid.mean():.1%}  "
        f"min {sample.min():.2f}  p2 {np.percentile(sample, 2):.2f}  "
        f"median {np.median(sample):.2f}  p98 {np.percentile(sample, 98):.2f}  max {sample.max():.2f}"
    )


def spot(values: np.ndarray, lat: float, lon: float) -> float:
    y = int(np.clip((90.0 - lat) / 180.0 * values.shape[0], 0, values.shape[0] - 1))
    x = int(np.clip((lon + 180.0) / 360.0 * values.shape[1], 0, values.shape[1] - 1))
    return float(values[y, x])


def entry(title, source, low, high, note, citation, ramp) -> dict:
    return {
        "available": True,
        "title": title,
        "source": source,
        "low": low,
        "high": high,
        "note": note,
        "citation": citation,
        "ramp": ramp,
    }


def catalog() -> dict:
    return {
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
                "temperature": entry(
                    "Air temperature, 2 m",
                    "NASA GIBS / MERRA-2 monthly, July 2024",
                    "colder",
                    "warmer",
                    "July 2024 monthly air temperature about 2 m above the surface. Yellow and red are warmer; blue and purple are colder.",
                    "Gelaro, R., and others (2017), The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2), Journal of Climate. Map: NASA GIBS layer MERRA2_2m_Air_Temperature_Monthly, July 2024.",
                    "heat",
                ),
                "ice-water": entry(
                    "Ocean, sea ice, and snow",
                    "ETOPO1 plus AMSR2 sea ice (March 2024) and MODIS snow (15 Jan 2024)",
                    "open water",
                    "ice or snow",
                    "Blue is ocean and brown is land, from ETOPO1. Sea ice is the March 2024 AMSR2 concentration. Snow is the 15 January 2024 MODIS snow cover, including the Antarctic ice sheet.",
                    "Amante, C., and B. W. Eakins (2009), ETOPO1 1 arc-minute global relief, NOAA. Sea ice: JAXA AMSR2 via NASA GIBS AMSRU2_Sea_Ice_Concentration_25km, March 2024. Snow: NASA MODIS Terra NDSI snow cover, 15 January 2024.",
                    "photo",
                ),
                "geology": entry(
                    "Crustal thickness",
                    "CRUST1.0, Laske, Masters, Ma, and Pasyanos 2013",
                    "thinner crust",
                    "thicker crust",
                    "Thickness of ice, sediment, and crystalline crust, with the ocean water left out. Ocean crust is thin; continents and mountains are thick.",
                    "Laske, G., G. Masters, Z. Ma, and M. Pasyanos (2013), Update on CRUST1.0, a 1-degree global model of Earth's crust. Geophysical Research Abstracts 15, EGU2013-2658.",
                    "heat",
                ),
                "weather": entry(
                    "Rain and snow rate",
                    "NASA GIBS / IMERG, 15 July 2024",
                    "little precipitation",
                    "heavy precipitation",
                    "One day of NASA IMERG precipitation. Green is lighter rain and red is heavier. Dark areas had little or no precipitation that day.",
                    "Huffman, G. J., and others, NASA IMERG. Map: NASA GIBS layer IMERG_Precipitation_Rate for 15 July 2024.",
                    "rain",
                ),
                "light": entry(
                    "Blue Marble",
                    "NASA Blue Marble Next Generation, August 2004",
                    "darker surface",
                    "brighter surface",
                    "Cloud-free view of reflected sunlight. Ice and desert are bright; forest and ocean are darker.",
                    "Stöckli, R., E. Vermote, N. Saleous, R. Simmon, and D. Herring, The Blue Marble Next Generation, NASA. Map: NASA GIBS BlueMarble_NextGeneration, August 2004.",
                    "photo",
                ),
                "radiation": entry(
                    "Outgoing longwave radiation",
                    "NASA CERES EBAF, July 2018",
                    "less heat leaving",
                    "more heat leaving",
                    "Heat radiation leaving the top of the atmosphere in July 2018. Deserts and the tropics radiate more; Antarctica radiates less.",
                    "Loeb, N. G., and others, CERES Energy Balanced and Filled top-of-atmosphere fluxes. Map: NASA GIBS CERES_EBAF_TOA_Longwave_Flux_All_Sky_Monthly, July 2018.",
                    "heat",
                ),
            },
            "moon": {
                "temperature": entry(
                    "Hottest surface temperature",
                    "LRO Diviner bolometric maximum, 2009–2015",
                    "colder",
                    "hotter",
                    "The hottest temperature Diviner recorded in each 0.5° cell, from July 2009 through April 2015. Dark maria get hotter than the bright highlands. The poles are included.",
                    "Williams, J.-P., D. A. Paige, B. T. Greenhagen, and E. Sefton-Nash (2017), The global surface temperatures of the Moon as measured by the Diviner Lunar Radiometer Experiment, Icarus 283, 300–325.",
                    "heat",
                ),
                "ice-water": entry(
                    "Coldest surface temperature",
                    "LRO Diviner bolometric minimum, 2009–2015",
                    "warmer",
                    "colder, ice can persist",
                    "The coldest temperature in each cell. Water ice is stable where the surface stays below about 110 K, which on this map is the bright polar ground. Most of the Moon is too warm.",
                    "Williams, J.-P., and others (2017), Icarus 283, 300–325, minimum bolometric temperature. Ice stability near 110 K: Paige, D. A., and others (2010), Science 330, 479–482.",
                    "heat",
                ),
                "geology": entry(
                    "Geologic units",
                    "USGS unified geologic map, Fortezzo, Spudis, and Harrel 2020",
                    "unit color",
                    "unit color",
                    "Each color is a rock unit on the 1:5,000,000 unified map, drawn in the same simple-cylindrical frame as the other Moon maps. Maria, highlands, and crater ejecta are separate colors.",
                    "Fortezzo, C. M., P. D. Spudis, and S. L. Harrel (2020), Release of the digital unified global geologic map of the Moon at 1:5,000,000 scale. USGS raster, simple cylindrical, public domain.",
                    "photo",
                ),
                "weather": entry(
                    "Day–night temperature swing",
                    "LRO Diviner maximum minus minimum, 2009–2015",
                    "smaller change",
                    "larger change",
                    "The Moon has no atmosphere, so it has no wind, clouds, or storms. This is the change from the coldest to the hottest temperature of the ground, which is how the surface environment changes through a lunar day.",
                    "Williams, J.-P., and others (2017), Icarus 283, 300–325. The map is the Diviner bolometric maximum minus the minimum on the same 0.5° grid.",
                    "heat",
                ),
                "light": entry(
                    "Reflected sunlight",
                    "USGS / LRO WAC global mosaic",
                    "darker maria",
                    "brighter highlands",
                    "Lunar Reconnaissance Orbiter Wide Angle Camera mosaic. The dark maria are basaltic lava; the highlands are brighter.",
                    "USGS Astrogeology WMS, Lunar Reconnaissance Orbiter Wide Angle Camera global mosaic, simple cylindrical.",
                    "photo",
                ),
                "radiation": entry(
                    "Heat leaving at midnight",
                    "LRO Diviner local midnight temperature, 2009–2015",
                    "less heat leaving",
                    "more heat leaving",
                    "Infrared power the surface radiates at local midnight, from the measured midnight temperature. Rock stays warmer, and radiates more, than dust. This is heat radiation, not a cosmic-ray dose.",
                    "Williams, J.-P., and others (2017), Icarus 283, 300–325, bolometric temperature at local hour 0. Power is σT⁴ with σ = 5.67×10⁻⁸ W/m²/K⁴.",
                    "heat",
                ),
            },
            "mars": {
                "temperature": entry(
                    "Afternoon surface temperature",
                    "MGS TES afternoon temperature, Mars Year 25",
                    "colder",
                    "warmer",
                    "Surface temperature retrieved from TES infrared spectra near 14:00 local time, averaged in 5° cells across Mars Year 25. Empty cells, mostly near the winter pole of a given season, are dark.",
                    "Smith, M. D. (2004), Interannual variability in TES atmospheric observations of Mars, Icarus 167, 148–165. Values from the NASA PDS TES column-opacity bundle (Montabone, Smith, Wolff, and Cantor) for all of Mars Year 25.",
                    "heat",
                ),
                "ice-water": entry(
                    "Hydrogen and water ice",
                    "Mars Odyssey neutron spectrometer, epithermal neutrons",
                    "less hydrogen",
                    "more hydrogen",
                    "Fewer epithermal neutrons means more hydrogen in the top metre of soil. In these first 25.7 days of mapping the southern high latitudes are the strong hydrogen signal. The unmeasured polar rows are dark.",
                    "Feldman, W. C., and others (2002), Global distribution of neutrons from Mars: results from Mars Odyssey, Science 297, 75–78. NASA PDS table ns_epithermal_020917.",
                    "heat",
                ),
                "geology": entry(
                    "Nighttime thermal inertia",
                    "MGS TES nightside thermal inertia, Putzig and Mellon 2007",
                    "dust",
                    "rock or duricrust",
                    "How fast the ground takes up and gives off heat, over the whole planet. Dust is low; rock and duricrust are high. A colored geologic-unit plate in this projection is not the map used here, because the published unit sheet is not simple cylindrical and would put clicks in the wrong place.",
                    "Putzig, N. E., and M. T. Mellon (2007), Apparent thermal inertia and the surface heterogeneity of Mars, Icarus 191, 68–94. NASA PDS map GLOBAL_TI_NIGHT_2007, simple cylindrical, 20 pixels per degree.",
                    "heat",
                ),
                "weather": entry(
                    "Dust in the atmosphere",
                    "MGS TES column dust optical depth, Mars Year 25",
                    "clearer air",
                    "more dust",
                    "Dust opacity at 9.3 μm, averaged in 5° cells across Mars Year 25. Southern summer raises the dust. Empty cells are dark.",
                    "Montabone, L., and others (2015), Eight-year climatology of dust optical depth on Mars, Icarus 251, 65–95. Retrievals: NASA PDS TES infrared column dust optical depth for all of Mars Year 25.",
                    "heat",
                ),
                "light": entry(
                    "Viking color mosaic",
                    "USGS / Viking MDIM 2.1 color mosaic",
                    "darker surface",
                    "brighter surface",
                    "Viking orbiter color mosaic. The bright spots at the poles are ice. Hellas is the bright basin in the south.",
                    "USGS Astrogeology WMS, Viking Mars Digital Image Model 2.1 color mosaic, simple cylindrical.",
                    "photo",
                ),
                "radiation": entry(
                    "Fast neutrons from cosmic rays",
                    "Mars Odyssey neutron spectrometer, fast neutrons",
                    "fewer fast neutrons",
                    "more fast neutrons",
                    "Fast neutrons made when cosmic rays hit the surface. High ground such as Olympus Mons has a thinner atmosphere overhead, so more fast neutrons escape. This is a measured neutron flux, not a dose in milligray.",
                    "Feldman, W. C., and others (2002), Science 297, 75–78. NASA PDS table ns_fast_020917, the first 25.7 days of mapping.",
                    "heat",
                ),
            },
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    download()
    tmax = diviner_grid("diviner_tbol_max.xyz")
    tmin = diviner_grid("diviner_tbol_min.xyz")
    tnight = diviner_grid("diviner_tbol_hour00.xyz")
    swing = tmax - tmin
    emitted = SIGMA * np.clip(tnight, 0, None) ** 4
    for name, grid in (("moon Tmax", tmax), ("moon Tmin", tmin), ("moon swing", swing), ("moon midnight W/m2", emitted)):
        report(name, grid, np.isfinite(grid))
    print(
        "moon maria Tmax",
        spot(tmax, 8, 24),
        "highlands Tmax",
        spot(tmax, -20, 160),
        "south pole Tmin",
        spot(tmin, -89, 0),
    )

    epi = neutron_grid("ns_epithermal.tab")
    fast = neutron_grid("ns_fast.tab")
    hydrogen = -epi
    report("mars epithermal", epi, np.isfinite(epi))
    report("mars fast", fast, np.isfinite(fast))
    print(
        "olympus fast",
        spot(fast, 18.6, 226.2 - 360),
        "hellas fast",
        spot(fast, -42.4, 70.5),
        "north epi",
        spot(epi, 80, 0),
        "equator epi",
        spot(epi, 0, 0),
    )

    lon, lat, temp, dust = tes_fields()
    temp_ok = (temp > 130) & (temp < 350)
    dust_ok = dust >= 0
    temp_map = bin_mean(lon, lat, temp, temp_ok)
    dust_map = bin_mean(lon, lat, dust, dust_ok)
    report("mars Tsurf", temp_map, np.isfinite(temp_map))
    report("mars dust", dust_map, np.isfinite(dust_map))
    print("olympus T", spot(temp_map, 18.6, 226.2 - 360), "hellas T", spot(temp_map, -42.4, 70.5))

    inertia = thermal_inertia()
    report("mars TI", inertia, np.isfinite(inertia))

    save(colorize(tmax, np.isfinite(tmax), *limits(tmax, np.isfinite(tmax))), "moon_temperature.png")
    # Bright end is colder, so ice-stable ground stands out at the poles.
    cold = -tmin
    save(colorize(cold, np.isfinite(cold), *limits(cold, np.isfinite(cold))), "moon_ice-water.png")
    save(moon_geology(), "moon_geology.png")
    save(colorize(swing, np.isfinite(swing), *limits(swing, np.isfinite(swing))), "moon_weather.png")
    save(colorize(emitted, np.isfinite(emitted), *limits(emitted, np.isfinite(emitted))), "moon_radiation.png")

    save(colorize(temp_map, np.isfinite(temp_map), *limits(temp_map, np.isfinite(temp_map))), "mars_temperature.png")
    save(colorize(hydrogen, np.isfinite(hydrogen), *limits(hydrogen, np.isfinite(hydrogen))), "mars_ice-water.png")
    # A few cells are thousands of SI units and would otherwise flatten the dust-to-rock range.
    save(colorize(inertia, np.isfinite(inertia), 40, 400), "mars_geology.png")
    save(colorize(dust_map, np.isfinite(dust_map), *limits(dust_map, np.isfinite(dust_map))), "mars_weather.png")
    save(colorize(fast, np.isfinite(fast), *limits(fast, np.isfinite(fast))), "mars_radiation.png")

    (OUT / "catalog.json").write_text(json.dumps(catalog(), indent=2) + "\n")
    print("wrote replacement maps")


if __name__ == "__main__":
    main()
