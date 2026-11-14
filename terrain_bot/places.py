"""Name the place at a coordinate and fetch pictures of it.

Names come from official gazetteers stored in ``data/gazetteer`` plus live
look-ups for Earth's land (Wikipedia, OpenStreetMap):

* Moon and Mars: IAU Gazetteer of Planetary Nomenclature (USGS).
* Earth sea floor: IHO-IOC GEBCO Gazetteer of Undersea Feature Names.
* Earth land: Wikipedia articles near the point, with the OpenStreetMap
  (Nominatim) region as context.

Pictures are a photo from the matching Wikipedia/Wikidata entry when one exists,
plus an orbital view centred on the point: NASA Blue Marble with bathymetry
(Earth), LRO WAC mosaic (Moon) and Viking MDIM 2.1 colour mosaic (Mars).
"""

from __future__ import annotations

import gzip
import io
import json
import math
import re
import struct
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

from .bodies import BODIES

GAZ_DIR = Path(__file__).resolve().parent.parent / "data" / "gazetteer"
USER_AGENT = "terrain-bot/0.1 (planetary terrain similarity research tool)"
TIMEOUT_S = 8

IAU_URL = "https://asc-planetarynames-data.s3.us-west-2.amazonaws.com/{body}_nomenclature_center_pts.zip"
GEBCO_URL = "https://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/Undersea_Features/FeatureServer"
MARINE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
              "ne_10m_geography_marine_polys.geojson")
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
NOMINATIM = "https://nominatim.openstreetmap.org/reverse"

VIEW_SOURCES = {
    "earth": (
        "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi",
        "BlueMarble_ShadedRelief_Bathymetry",
        "NASA Blue Marble with GEBCO bathymetry (GIBS)",
    ),
    "moon": (
        "https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/earth/moon_simp_cyl.map",
        "LROC_WAC",
        "NASA LRO Wide Angle Camera mosaic (USGS)",
    ),
    "mars": (
        "https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/mars/mars_simp_cyl.map",
        "MDIM21_color",
        "Viking MDIM 2.1 colour mosaic (USGS)",
    ),
}

# Wikipedia geosearch "type" values, most interesting first.
WIKI_TYPE_RANK = {"landmark": 0, "mountain": 0, "waterbody": 1, "glacier": 1, "isle": 1, "river": 2,
                  "adm1st": 3, "adm2nd": 4, "city": 5, "adm3rd": 6}
# Articles about events that happened at a place rather than the place itself.
EVENT_TITLE = re.compile(r"^\d{3,4}\b|expedition|disaster|avalanche|accident|crash|earthquake|battle|incident", re.I)
LINEAR_TYPES = ("Rima", "Vallis", "Fossa", "Chasma", "Catena", "Dorsum", "Rupes", "Labyrinthus", "Sulcus", "Linea")


# --------------------------------------------------------------------------- downloads


def _get(url: str, params: dict | None = None, timeout: float = TIMEOUT_S) -> bytes:
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _get_json(url: str, params: dict | None = None, timeout: float = TIMEOUT_S) -> dict:
    return json.loads(_get(url, params, timeout))


def _read_dbf(blob: bytes) -> list[dict]:
    n, header_len, rec_len = struct.unpack("<IHH", blob[4:12])
    fields = []
    for i in range(32, header_len - 1, 32):
        fields.append((blob[i : i + 11].split(b"\0")[0].decode(), blob[i + 16]))
    rows = []
    for k in range(n):
        rec = blob[header_len + k * rec_len : header_len + (k + 1) * rec_len]
        if rec[:1] == b"*":
            continue
        pos, row = 1, {}
        for name, length in fields:
            row[name] = rec[pos : pos + length].decode("utf-8", "replace").strip()
            pos += length
        rows.append(row)
    return rows


def download_iau(body: str) -> Path:
    blob = _get(IAU_URL.format(body=body.upper()), timeout=120)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        dbf = next(n for n in z.namelist() if n.endswith(".dbf"))
        rows = _read_dbf(z.read(dbf))
    features = []
    for r in rows:
        if "Dropped" in r["approval"] or not r["center_lat"]:
            continue
        lon = float(r["center_lon"])
        features.append(
            {
                "name": r["name"] or r["clean_name"],
                "type": r["type"].split(",")[0],
                "lat": round(float(r["center_lat"]), 4),
                "lon": round(lon - 360 if lon > 180 else lon, 4),
                "diameter_km": round(float(r["diameter"] or 0), 2),
                "id": r["link"].rstrip("/").rsplit("/", 1)[-1],
                "origin": r["origin"],
            }
        )
    return _save(f"{body}_iau.json.gz", {"source": "IAU Gazetteer of Planetary Nomenclature (USGS)", "features": features})


def _densify(coords: list, step_deg: float = 0.1) -> list:
    out = []
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step_deg))
        out.extend([[x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n] for t in range(n)])
    out.append(list(coords[-1]))
    return out


def download_gebco() -> Path:
    features = []
    for layer, kind in ((0, "point"), (1, "line"), (2, "polygon")):
        offset = 0
        while True:
            page = _get_json(
                f"{GEBCO_URL}/{layer}/query",
                {"where": "1=1", "outFields": "NAME,TYPE,FEATURE_ID", "returnGeometry": "true",
                 "geometryPrecision": 3, "outSR": 4326, "resultOffset": offset, "resultRecordCount": 1000, "f": "json"},
                timeout=120,
            )
            for f in page["features"]:
                a, g = f["attributes"], f.get("geometry") or {}
                if kind == "point":
                    parts = [[[g["x"], g["y"]]]] if "x" in g else []
                else:
                    parts = g.get("paths") or g.get("rings") or []
                if not parts:
                    continue
                ftype = "" if (a["TYPE"] or "").upper() in ("", "UNKNOWN") else a["TYPE"]
                features.append({"name": a["NAME"], "type": ftype, "id": a["FEATURE_ID"], "kind": kind,
                                 "parts": [[[round(x, 3), round(y, 3)] for x, y in p] for p in parts]})
            offset += len(page["features"])
            if not page.get("exceededTransferLimit") or not page["features"]:
                break
    return _save("earth_gebco.json.gz", {"source": "IHO-IOC GEBCO Gazetteer of Undersea Feature Names", "features": features})


def _save(name: str, payload: dict) -> Path:
    GAZ_DIR.mkdir(parents=True, exist_ok=True)
    path = GAZ_DIR / name
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    return path


def download_marine() -> Path:
    data = json.loads(_get(MARINE_URL, timeout=120))
    features = []
    for f in data["features"]:
        props, geom = f["properties"], f["geometry"]
        if props["featurecla"] in ("reef", "river") or not geom:
            continue
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings = [[[round(x, 2), round(y, 2)] for x, y in ring] for poly in polys for ring in poly]
        name = props.get("name_en") or props.get("name")
        if not name:
            continue
        features.append({"name": name.title() if name.isupper() else name, "kind": props["featurecla"], "rings": rings})
    return _save("earth_marine.json.gz", {"source": "Natural Earth marine areas", "features": features})


def download_gazetteers() -> list[Path]:
    return [download_iau("moon"), download_iau("mars"), download_gebco(), download_marine()]


def _load(name: str) -> dict | None:
    path = GAZ_DIR / name
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- geometry helpers


def _unit(lat, lon) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def _arc_km(chord, radius_km: float):
    return 2 * radius_km * np.arcsin(np.clip(np.asarray(chord) / 2, 0, 1))


def _distance_text(km: float) -> str:
    return f"{km:,.0f} km" if km >= 10 else f"{km:.1f} km"


# --------------------------------------------------------------------------- Moon / Mars


class PlanetaryGazetteer:
    """IAU named features with centre and diameter; picks the feature the point is in (or nearest to)."""

    def __init__(self, body: str):
        data = _load(f"{body}_iau.json.gz")
        if data is None:
            raise FileNotFoundError(f"Run `python -m terrain_bot download-gazetteer` to fetch {body} names")
        self.body = BODIES[body]
        self.source = data["source"]
        self.features = data["features"]
        lat = np.array([f["lat"] for f in self.features])
        lon = np.array([f["lon"] for f in self.features])
        self.radius_km = np.array([f["diameter_km"] for f in self.features]) / 2
        self.minor = np.array([f["type"] in ("Satellite Feature", "Landing site name") for f in self.features])
        self.tree = cKDTree(_unit(lat, lon))

    def lookup(self, lat: float, lon: float, pixel_km: float = 0.0) -> dict:
        R = self.body.mean_radius_km
        reach = min(2.0, (self.radius_km.max() + 300) / R)
        idx = np.array(self.tree.query_ball_point(_unit(lat, lon), reach), dtype=int)
        if idx.size == 0:
            return {"name": None}
        d = _arc_km(np.linalg.norm(self.tree.data[idx] - _unit(lat, lon), axis=1), R)
        r = self.radius_km[idx]
        inside = d <= np.maximum(r, 1.0)
        # Features smaller than about a map pixel can't be told apart at this resolution.
        big_enough = 2 * r >= max(pixel_km, 1.0)

        def pick(mask):
            cand = np.flatnonzero(mask & ~self.minor[idx]) if (mask & ~self.minor[idx]).any() else np.flatnonzero(mask)
            return cand

        cand = pick(inside & big_enough)
        if cand.size:
            best = cand[np.argmin(r[cand])]
            relation = "inside"
        else:
            cand = pick(np.ones_like(inside))
            edge = np.maximum(d - r, 0)
            best = cand[np.argmin(edge[cand])]
            relation = "near"
        f = self.features[idx[best]]
        linear = np.array([self.features[k]["type"].startswith(LINEAR_TYPES) for k in idx])
        # A linear feature's "diameter" is its length, so it only contains other linear features (chasma in valles).
        allowed = ~linear | linear[best]
        larger = np.flatnonzero(inside & ~self.minor[idx] & allowed & (r > r[best]))
        containers = [self.features[idx[j]] for j in larger[np.argsort(r[larger])][:3]]
        dist_centre = float(d[best])
        dist_edge = max(dist_centre - float(r[best]), 0.0)
        return {
            "name": _planetary_display_name(f),
            "feature_name": f["name"],
            "type": f["type"],
            "diameter_km": f["diameter_km"],
            "relation": relation,
            "distance_km": round(dist_centre if relation == "inside" else dist_edge, 1),
            "description": _planetary_description(f, relation, dist_centre, dist_edge),
            "context": "inside " + " / ".join(_planetary_display_name(c) for c in containers) if containers else None,
            "origin": f["origin"] or None,
            "gazetteer_id": f["id"],
            "gazetteer_url": f"https://planetarynames.wr.usgs.gov/Feature/{f['id']}",
            "source": self.source,
        }


def _planetary_display_name(f: dict) -> str:
    name, ftype = f["name"], f["type"]
    if ftype.lower().startswith("crater") and "crater" not in name.lower():
        return f"{name} crater"
    return name


def _planetary_description(f: dict, relation: str, d_centre: float, d_edge: float) -> str:
    span = "long" if f["type"].startswith(LINEAR_TYPES) else "across"
    size = f"{f['diameter_km']:,.0f} km {span}" if f["diameter_km"] >= 1 else f"{f['diameter_km'] * 1000:,.0f} m {span}"
    if relation == "inside":
        where = "at its centre" if d_centre < max(2.0, f["diameter_km"] * 0.1) else f"{_distance_text(d_centre)} from its centre"
    else:
        where = f"{_distance_text(d_edge)} outside its rim"
    return f"{f['type']}, {size}; this point is {where}"


# --------------------------------------------------------------------------- Earth sea floor


class UnderseaGazetteer:
    """GEBCO undersea features: nearest named point/line, or the smallest polygon containing the point."""

    def __init__(self):
        data = _load("earth_gebco.json.gz")
        if data is None:
            raise FileNotFoundError("Run `python -m terrain_bot download-gazetteer` to fetch undersea names")
        self.source = data["source"]
        self.features = data["features"]
        verts, owner, self.polygons = [], [], []
        for i, f in enumerate(self.features):
            if f["kind"] != "polygon":
                for part in f["parts"]:
                    pts = _densify(part) if len(part) > 1 else part
                    verts.extend(pts)
                    owner.extend([i] * len(pts))
            else:
                rings = [np.array(p) for p in f["parts"]]
                area = sum(abs(_ring_area(r)) for r in rings)
                self.polygons.append((i, [MplPath(r) for r in rings], area))
        verts = np.array(verts, dtype=float)
        self.owner = np.array(owner)
        self.tree = cKDTree(_unit(verts[:, 1], verts[:, 0]))

        marine = _load("earth_marine.json.gz") or {"features": []}
        self.seas = []
        for f in marine["features"]:
            rings = [np.array(r) for r in f["rings"] if len(r) >= 3]
            area = sum(abs(_ring_area(r)) for r in rings)
            self.seas.append((f["name"], [MplPath(r) for r in rings], area))

    def sea(self, lat: float, lon: float) -> str | None:
        """Smallest named sea/gulf/ocean containing the point (Natural Earth)."""
        hits = [(area, name) for name, paths, area in self.seas if sum(p.contains_point((lon, lat)) for p in paths) % 2]
        return min(hits)[1] if hits else None

    def lookup(self, lat: float, lon: float, near_km: float = 30.0) -> dict:
        R = BODIES["earth"].mean_radius_km
        containing = [
            (area, i) for i, paths, area in self.polygons if sum(p.contains_point((lon, lat)) for p in paths) % 2 == 1
        ]
        container = self.features[min(containing)[1]] if containing else None
        sea = self.sea(lat, lon)

        chord, j = self.tree.query(_unit(lat, lon))
        near = (self.features[self.owner[j]], float(_arc_km(chord, R)))

        context = [f"inside {_undersea_name(container)}"] if container else []
        if sea:
            context.append(f"in the {sea}" if not sea.lower().startswith(("the ", "lake ")) else f"in {sea}")
        if near and near[1] <= near_km:
            f, km = near
            relation = "on" if km < 5 else "near"
            desc = f"{f['type'] or 'Undersea feature'}; this point is {'on it' if relation == 'on' else _distance_text(km) + ' away'}"
        elif container:
            f, km, relation = container, 0.0, "inside"
            desc = f"{f['type'] or 'Undersea feature'}; this point is inside its outline"
            context = context[1:]
        else:
            nearest = f"nearest named feature: {_undersea_name(near[0])}, {_distance_text(near[1])} away" if near else None
            return {
                "name": f"{sea} sea floor" if sea else "Open sea floor",
                "relation": "inside",
                "description": "No named undersea feature within 30 km",
                "context": nearest,
                "source": self.source + (" + Natural Earth" if sea else ""),
            }
        return {
            "name": _undersea_name(f),
            "type": f["type"],
            "relation": relation,
            "distance_km": round(km, 1),
            "description": desc,
            "context": ", ".join(context) or None,
            "source": self.source,
        }


def _ring_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _undersea_name(f: dict) -> str:
    name, ftype = f["name"], f["type"]
    return name if not ftype or ftype.lower() in name.lower() else f"{name} {ftype}"


# --------------------------------------------------------------------------- live web look-ups


_nominatim_lock = threading.Lock()
_nominatim_last = [0.0]


@lru_cache(maxsize=4096)
def _region(lat: float, lon: float) -> str | None:
    """OpenStreetMap region/country name (Nominatim allows one request per second)."""
    with _nominatim_lock:
        wait = 1.05 - (time.time() - _nominatim_last[0])
        if wait > 0:
            time.sleep(wait)
        _nominatim_last[0] = time.time()
        try:
            j = _get_json(NOMINATIM, {"lat": lat, "lon": lon, "format": "json", "zoom": 8, "accept-language": "en"})
        except Exception:
            return None
    if "error" in j:
        return None
    addr = j.get("address", {})
    parts = [addr.get(k) for k in ("county", "state_district", "state", "region", "country")]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return ", ".join(out[-3:]) or j.get("display_name")


@lru_cache(maxsize=4096)
def _wiki_nearby(lat: float, lon: float) -> dict | None:
    try:
        j = _get_json(WIKI_API, {"action": "query", "list": "geosearch", "gscoord": f"{lat}|{lon}", "gsradius": 10000,
                                  "gslimit": 30, "gsprop": "type", "format": "json"})
    except Exception:
        return None
    hits = j.get("query", {}).get("geosearch", [])
    if not hits:
        return None
    best = min(
        hits,
        key=lambda h: (
            WIKI_TYPE_RANK.get(h.get("type"), 7),
            bool(EVENT_TITLE.search(h["title"])),
            round(h["dist"], -2),
            len(h["title"]),
        ),
    )
    return {"title": best["title"], "distance_km": best["dist"] / 1000, "type": best.get("type")}


@lru_cache(maxsize=4096)
def wiki_page(title: str) -> dict | None:
    """Summary, thumbnail and URL of an English Wikipedia article (follows redirects)."""
    try:
        j = _get_json(WIKI_API, {"action": "query", "titles": title, "redirects": 1, "prop": "pageimages|extracts|info",
                                  "pithumbsize": 480, "exintro": 1, "explaintext": 1, "exsentences": 2,
                                  "inprop": "url", "format": "json"})
    except Exception:
        return None
    pages = j.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)
    if not page or "missing" in page:
        return None
    return {
        "title": page["title"],
        "url": page.get("fullurl"),
        "summary": page.get("extract") or None,
        "photo": (page.get("thumbnail") or {}).get("source"),
    }


@lru_cache(maxsize=4096)
def _wikidata_for_gazetteer(gazetteer_id: str) -> dict | None:
    """Wikipedia title and Commons image of an IAU feature, via Wikidata property P2824."""
    q = (
        "SELECT ?article ?image WHERE { ?item wdt:P2824 \"%s\" . OPTIONAL { ?item wdt:P18 ?image } "
        "OPTIONAL { ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> } } LIMIT 1"
        % gazetteer_id.replace('"', "")
    )
    try:
        j = _get_json(WIKIDATA_SPARQL, {"query": q, "format": "json"})
    except Exception:
        return None
    rows = j.get("results", {}).get("bindings", [])
    if not rows:
        return None
    row = rows[0]
    article = row.get("article", {}).get("value")
    image = row.get("image", {}).get("value")
    return {
        "title": urllib.parse.unquote(article.rsplit("/", 1)[-1]).replace("_", " ") if article else None,
        "photo": f"{image.replace('http://', 'https://')}?width=480" if image else None,
    }


# --------------------------------------------------------------------------- public API


@dataclass
class PlaceFinder:
    online: bool = True

    def __post_init__(self):
        self._gaz: dict[str, object] = {}
        self._lock = threading.Lock()

    def _gazetteer(self, key: str):
        with self._lock:
            if key not in self._gaz:
                try:
                    self._gaz[key] = UnderseaGazetteer() if key == "earth" else PlanetaryGazetteer(key)
                except FileNotFoundError:
                    self._gaz[key] = None
            return self._gaz[key]

    def describe(self, body: str, lat: float, lon: float, elevation_m: float | None = None, pixel_km: float = 0.0) -> dict:
        body = body.lower()
        lat, lon = round(float(lat), 3), round(float(lon), 3)
        place: dict = {"body": BODIES[body].name, "lat": lat, "lon": lon, "name": None}
        if body == "earth":
            place.update(self._describe_earth(lat, lon, elevation_m))
        else:
            gaz = self._gazetteer(body)
            if gaz is not None:
                place.update(gaz.lookup(lat, lon, pixel_km))
            if self.online and place.get("gazetteer_id"):
                wd = _wikidata_for_gazetteer(place["gazetteer_id"])
                if wd:
                    page = wiki_page(wd["title"]) if wd.get("title") else None
                    place["wiki"] = page
                    place["photo"] = (page or {}).get("photo") or wd.get("photo")
        if not place.get("name"):
            place["name"] = f"Unnamed area on {BODIES[body].name}"
        view = VIEW_SOURCES[body]
        place["view_credit"] = view[2]
        place["view_url"] = f"/api/place-image?body={body}&lat={lat}&lon={lon}"
        return place

    def _describe_earth(self, lat: float, lon: float, elevation_m: float | None) -> dict:
        underwater = elevation_m is not None and elevation_m < 0
        out: dict = {}
        if underwater:
            gaz = self._gazetteer("earth")
            if gaz is not None:
                out.update(gaz.lookup(lat, lon))
            if self.online and out.get("name"):
                page = wiki_page(out["name"])
                if page and _same_topic(page["title"], out["name"]):
                    out["wiki"] = page
                    out["photo"] = page.get("photo")
            return out
        if not self.online:
            return out
        near = _wiki_nearby(lat, lon)
        region = _region(lat, lon)
        if near:
            page = wiki_page(near["title"]) or {}
            out.update(
                {
                    "name": near["title"],
                    "type": near.get("type"),
                    "relation": "near",
                    "distance_km": round(near["distance_km"], 1),
                    "description": f"Nearest notable place on Wikipedia, {_distance_text(near['distance_km'])} away",
                    "wiki": page or None,
                    "photo": page.get("photo"),
                    "source": "Wikipedia",
                }
            )
            out["context"] = region
        elif region:
            out.update({"name": region.split(", ")[0], "context": ", ".join(region.split(", ")[1:]) or None,
                        "description": "No Wikipedia article within 10 km; region from OpenStreetMap",
                        "source": "OpenStreetMap Nominatim"})
        return out


def _same_topic(title: str, name: str) -> bool:
    return title.lower().split(" (")[0] == name.lower()


@lru_cache(maxsize=512)
def orbital_view(body: str, lat: float, lon: float, half_km: float = 150.0, size: int = 300) -> bytes:
    """JPEG of real imagery centred on the point, with a crosshair."""
    from PIL import Image, ImageDraw

    url, layer, _ = VIEW_SOURCES[body]
    R = BODIES[body].mean_radius_km
    km_per_deg = math.pi * R / 180
    h = half_km / km_per_deg
    w = min(180.0, h / max(math.cos(math.radians(lat)), 0.05))
    lat0, lat1 = lat - h, lat + h
    if lat0 < -90:
        lat0, lat1 = -90, -90 + 2 * h
    if lat1 > 90:
        lat0, lat1 = 90 - 2 * h, 90
    lon0, lon1 = lon - w, lon + w
    if lon0 < -180:
        lon0, lon1 = -180, -180 + 2 * w
    if lon1 > 180:
        lon0, lon1 = 180 - 2 * w, 180
    blob = _get(
        url,
        {"SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap", "LAYERS": layer, "STYLES": "", "SRS": "EPSG:4326",
         "BBOX": f"{lon0:.5f},{lat0:.5f},{lon1:.5f},{lat1:.5f}", "WIDTH": size, "HEIGHT": size, "FORMAT": "image/jpeg"},
        timeout=20,
    )
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    x = (lon - lon0) / (lon1 - lon0) * img.width
    y = (lat1 - lat) / (lat1 - lat0) * img.height
    d = ImageDraw.Draw(img)
    for color, wdt in (("black", 4), ("#ff4040", 2)):
        d.line([(x - 14, y), (x - 5, y)], fill=color, width=wdt)
        d.line([(x + 5, y), (x + 14, y)], fill=color, width=wdt)
        d.line([(x, y - 14), (x, y - 5)], fill=color, width=wdt)
        d.line([(x, y + 5), (x, y + 14)], fill=color, width=wdt)
    bar_km = 50 if half_km <= 200 else 200
    bar_px = bar_km / (2 * half_km) * img.width
    d.rectangle([8, img.height - 16, 8 + bar_px, img.height - 12], fill="white", outline="black")
    d.text((12 + bar_px, img.height - 20), f"{bar_km} km", fill="white", stroke_width=2, stroke_fill="black")
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()
