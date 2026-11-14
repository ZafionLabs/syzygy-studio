"""Build the place-name list for Earth, the seafloor, the Moon, and Mars.

Names come from three published gazetteers and are not invented:

- Earth cities, countries, and land features: Natural Earth
- Undersea features: GEBCO Gazetteer of Undersea Feature Names (IHO-IOC SCUFN)
- Moon and Mars: USGS Gazetteer of Planetary Nomenclature (IAU-approved names)

The result is frontend/public/names.json. Each record is
[name, latitude, longitude, minimum zoom, kind].
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from urllib.request import urlopen

import shapefile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "public" / "names.json"
CACHE = Path("/tmp/names")

URLS = {
    "moon.zip": "https://asc-planetarynames-data.s3.us-west-2.amazonaws.com/MOON_nomenclature_center_pts.zip",
    "mars.zip": "https://asc-planetarynames-data.s3.us-west-2.amazonaws.com/MARS_nomenclature_center_pts.zip",
    "cities.zip": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places_simple.zip",
    "countries.zip": "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
    "phys.zip": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_geography_regions_points.zip",
    "elev.zip": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_geography_regions_elevation_points.zip",
}
GEBCO = "https://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/Undersea_Features/FeatureServer"


def fetch(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists() or path.stat().st_size < 1000:
        path.write_bytes(urlopen(URLS[name], timeout=120).read())
    return path


def reader(name: str) -> shapefile.Reader:
    archive = zipfile.ZipFile(fetch(name))
    stem = next(item[:-4] for item in archive.namelist() if item.endswith(".shp"))
    folder = CACHE / Path(stem).name
    folder.mkdir(exist_ok=True)
    for item in archive.namelist():
        if item.startswith(stem):
            (folder / Path(item).name).write_bytes(archive.read(item))
    return shapefile.Reader(str(folder / Path(stem).name))


def wrap(lon: float) -> float:
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def planet_zoom(diameter: float) -> float:
    if diameter >= 400:
        return 1
    if diameter >= 150:
        return 1.4
    if diameter >= 60:
        return 2.2
    if diameter >= 25:
        return 3.5
    if diameter >= 10:
        return 6
    return 10


def planet(name: str) -> list[list]:
    rows = []
    for record in reader(name).iterRecords():
        label = (record["clean_name"] or record["name"] or "").strip()
        if not label:
            continue
        kind = str(record["type"] or "feature").split(",")[0].strip().lower()
        rows.append([
            label,
            round(float(record["center_lat"]), 3),
            round(wrap(float(record["center_lon"])), 3),
            planet_zoom(float(record["diameter"] or 0)),
            kind or "feature",
        ])
    return rows


def point_rows(name: str, kind: str, lat_key: str, lon_key: str, label_key: str, zoom_key: str | None, default_zoom: float) -> list[list]:
    rows = []
    for record in reader(name).iterRecords():
        data = record.as_dict()
        label = str(data.get(label_key) or "").strip()
        lat = data.get(lat_key)
        lon = data.get(lon_key)
        if not label or lat in (None, "") or lon in (None, ""):
            continue
        zoom = default_zoom
        if zoom_key and data.get(zoom_key) not in (None, ""):
            zoom = float(data[zoom_key])
        rows.append([label, round(float(lat), 3), round(wrap(float(lon)), 3), round(zoom, 1), kind])
    return rows


def countries() -> list[list]:
    rows = []
    for record in reader("countries.zip").iterRecords():
        data = record.as_dict()
        label = str(data.get("NAME") or "").strip()
        lat = data.get("LABEL_Y")
        lon = data.get("LABEL_X")
        if not label or lat in (None, "") or lon in (None, ""):
            continue
        rows.append([label, round(float(lat), 3), round(wrap(float(lon)), 3), 1, "country"])
    return rows


def ocean_zoom(kind: str) -> float:
    text = kind.lower()
    if any(word in text for word in ("trench", "trough")):
        return 1
    if any(word in text for word in ("ridge", "rise", "basin", "plateau", "plain")):
        return 1.6
    if any(word in text for word in ("seamount", "guyot", "bank", "reef", "canyon", "fracture", "escarpment")):
        return 2.8
    return 4


def centroid(geometry: dict) -> tuple[float, float] | None:
    points = geometry.get("points")
    if not points:
        paths = geometry.get("paths") or geometry.get("rings") or []
        points = [point for path in paths for point in path]
    if not points:
        return None
    lon = sum(point[0] for point in points) / len(points)
    lat = sum(point[1] for point in points) / len(points)
    if not math.isfinite(lat) or not math.isfinite(lon):
        return None
    return lat, lon


def gebco() -> list[list]:
    rows = []
    for layer in (0, 1, 2):
        offset = 0
        while True:
            url = (
                f"{GEBCO}/{layer}/query?where=1%3D1&outFields=NAME,TYPE"
                f"&returnGeometry=true&resultOffset={offset}&resultRecordCount=2000&f=json"
            )
            payload = json.load(urlopen(url, timeout=120))
            features = payload.get("features") or []
            for feature in features:
                attrs = feature.get("attributes") or {}
                label = str(attrs.get("NAME") or "").strip()
                center = centroid(feature.get("geometry") or {})
                if not label or center is None:
                    continue
                kind = str(attrs.get("TYPE") or "undersea feature").strip()
                rows.append([label, round(center[0], 3), round(wrap(center[1]), 3), ocean_zoom(kind), kind.lower()])
            offset += len(features)
            if not features or not payload.get("exceededTransferLimit"):
                break
    return rows


def dedupe(rows: list[list]) -> list[list]:
    best: dict[tuple, list] = {}
    for row in rows:
        key = (row[0].casefold(), round(row[1], 1), round(row[2], 1))
        current = best.get(key)
        if current is None or row[3] < current[3]:
            best[key] = row
    return list(best.values())


def main() -> None:
    earth = dedupe(
        point_rows("cities.zip", "city", "latitude", "longitude", "name", "min_zoom", 5)
        + countries()
        + point_rows("phys.zip", "land", "lat_y", "long_x", "name", "min_zoom", 4)
        + point_rows("elev.zip", "land", "lat_y", "long_x", "name", "min_zoom", 3)
    )
    ocean = dedupe(gebco())
    moon = dedupe(planet("moon.zip"))
    mars = dedupe(planet("mars.zip"))
    payload = {
        "citation": {
            "earth": "Natural Earth populated places, countries, and geographic points. Public domain.",
            "ocean": "GEBCO Gazetteer of Undersea Feature Names, IHO-IOC Sub-Committee on Undersea Feature Names, hosted by NOAA NCEI.",
            "moon": "Gazetteer of Planetary Nomenclature, USGS Astrogeology. Names approved by the International Astronomical Union.",
            "mars": "Gazetteer of Planetary Nomenclature, USGS Astrogeology. Names approved by the International Astronomical Union.",
        },
        "earth": earth,
        "ocean": ocean,
        "moon": moon,
        "mars": mars,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    counts = {key: len(payload[key]) for key in ("earth", "ocean", "moon", "mars")}
    print(counts, "bytes", OUT.stat().st_size)
    needed = {
        "earth": ["Tokyo", "Cairo", "Everest"],
        "ocean": ["Mariana"],
        "moon": ["Tycho", "Mare Imbrium"],
        "mars": ["Olympus Mons", "Hellas"],
    }
    for body, labels in needed.items():
        names = {row[0] for row in payload[body]}
        for label in labels:
            hits = [item for item in names if label.lower() in item.lower()]
            print(body, label, hits[:4])
            if not hits:
                raise SystemExit(f"missing {label} on {body}")


if __name__ == "__main__":
    main()
