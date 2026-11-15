"""Small browser UI for the terrain bot (standard library HTTP server)."""

from __future__ import annotations

import io
import json
import math
import re
import secrets
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .dem import compare_to_dem, load_dem
from .updates import active_updates, prepare_map
from .features import compute_features
from .explain import describe_surface, explain_query, explain_top, feature_label, feature_unit
from .heightmap import Heightmap, heightmap_from_dem
from .ground import SPAN_KM, ground_grid, ground_png, satellite_png
from .layers import catalog, compare_maps, layer_png
from .safety import FACTORS, build_safety, describe_cell, encode_png
from .matcher import MODES, SURFACES, Match, MatchConfig, SitePoint, TerrainMatcher
from .places import PlaceFinder, orbital_view
from .report import calibration_check

STATIC = Path(__file__).resolve().parent / "static"
CMAPS = {"earth": "gist_earth", "moon": "cividis", "mars": "copper"}
SOURCE_LABELS = {
    "dem": "Real measured data (NOAA ETOPO1, NASA LOLA, NASA MOLA)",
    "images": "Your images, darkest = lowest point, lightest = highest point",
    "images-fit": "Your images, brightness curve fitted to real data",
}
SOURCE_ORDER = ("dem", "images", "images-fit")
SESSION_COOKIE = "terrain_session"
MAX_VISITS = 24
MAX_VISIT_BYTES = 200_000_000


class _Visit:
    def __init__(self) -> None:
        self.elevation: dict[str, Heightmap] = {}
        self.layers: dict[tuple[str, str], bytes] = {}
        self.pngs: dict[tuple[str, str], bytes] = {}
        self.safety: dict[str, tuple[object, bytes]] = {}
        self.matchers: dict[tuple[str, str], TerrainMatcher] = {}
        self.top: dict[tuple, dict] = {}
        self.notes: list[dict] = []
        self.stored = 0


def _clean(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    return obj


def _site(p: SitePoint) -> dict:
    return {
        "body": p.body,
        "lat": p.lat,
        "lon": p.lon,
        "gray": p.gray,
        "features": p.features,
        "surface": describe_surface(p.body, p.elevation_m),
        "underwater": p.body == "Earth" and p.elevation_m < 0,
    }


def _match(m: Match) -> dict:
    return {
        "rank": m.rank,
        "similarity": m.similarity,
        "distance": m.distance,
        "elevation_difference_m": m.elevation_difference_m,
        "a": _site(m.a),
        "b": _site(m.b),
        "breakdown": [
            {**row, "label": feature_label(row["feature"]), "unit": feature_unit(row["feature"])} for row in m.breakdown
        ],
    }


def render_map_png(hm: Heightmap, cmap_name: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    t = (hm.elevation - hm.body.min_elevation_m) / hm.body.relief_m
    rgba = colormaps[cmap_name](np.clip(np.nan_to_num(t, nan=0.0), 0, 1))
    rgba[..., 3] = np.where(hm.valid, 1.0, 0.0)
    buf = io.BytesIO()
    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(buf, format="PNG")
    return buf.getvalue()


class TerrainApp:
    """Serves one or more data sources; matchers and global results are built lazily and cached."""

    def __init__(self, datasets: dict[str, dict[str, Heightmap]], config: MatchConfig, default_source: str, online: bool = True):
        self.datasets = {k: datasets[k] for k in sorted(datasets, key=lambda k: SOURCE_ORDER.index(k) if k in SOURCE_ORDER else 99)}
        self._features: dict[tuple[str, str], object] = {}
        self.places = PlaceFinder(online=online)
        self.base_config = config
        self.default_source = default_source
        self._matchers: dict[tuple[str, str], TerrainMatcher] = {}
        self._top: dict[tuple, dict] = {}
        self._lock = threading.RLock()
        self._safety: dict[str, tuple[object, bytes]] = {}
        self._sessions: OrderedDict[str, _Visit] = OrderedDict()
        self._pngs = {
            (src, body): render_map_png(hm, CMAPS.get(body, "viridis"))
            for src, maps in self.datasets.items()
            for body, hm in maps.items()
        }
        self._info = self._build_info()
        bodies = list(self.datasets[default_source])
        self.top(8, config.mode, default_source, "any", bodies[0], bodies[1])

    def _source(self, source: str | None) -> str:
        source = source or self.default_source
        if source not in self.datasets:
            raise ValueError(f"source must be one of {sorted(self.datasets)}")
        return source

    def _visit(self, session_id: str) -> _Visit | None:
        if not session_id:
            return None
        with self._lock:
            visit = self._sessions.get(session_id)
            if visit is not None:
                self._sessions.move_to_end(session_id)
            return visit

    def _body_of(self, body: str):
        for maps in self.datasets.values():
            if body in maps:
                return maps[body].body
        raise ValueError(f"unknown body {body!r}")

    def _trim_visits(self, incoming: int, keep: str) -> None:
        while self._sessions and (
            len(self._sessions) > MAX_VISITS
            or sum(item.stored for item in self._sessions.values()) + incoming > MAX_VISIT_BYTES
        ):
            oldest = next(iter(self._sessions))
            if oldest == keep and len(self._sessions) == 1:
                break
            if oldest == keep:
                self._sessions.move_to_end(oldest)
                continue
            self._sessions.popitem(last=False)

    def remember_map(self, session_id: str, prepared: dict) -> dict:
        """Keep a validated map for this browser visit only. Other visitors keep the bundled map."""
        body = prepared["body"]
        kind = prepared["kind"]
        payload: bytes = prepared["payload"]
        with self._lock:
            self._trim_visits(len(payload), session_id)
            visit = self._sessions.setdefault(session_id, _Visit())
            self._sessions.move_to_end(session_id)
            if kind == "elevation":
                with np.load(io.BytesIO(payload)) as packed:
                    elev = packed["elevation"]
                    lat_range = tuple(float(x) for x in packed["lat_range"])
                    lon_range = tuple(float(x) for x in packed["lon_range"])
                hm = heightmap_from_dem(elev, self._body_of(body), lat_range, lon_range, f"This visit ({prepared['file']})")
                visit.elevation[body] = hm
                for src, maps in self.datasets.items():
                    if body in maps:
                        visit.pngs[(src, body)] = render_map_png(hm, CMAPS.get(body, "viridis"))
                visit.matchers.clear()
                visit.top.clear()
            else:
                visit.layers[(body, kind)] = payload
            visit.safety.pop(body, None)
            visit.notes = [note for note in visit.notes if not (note["body"] == body and note["kind"] == kind)]
            visit.notes.append({"body": body, "kind": kind, "file": prepared["file"]})
            visit.stored += len(payload)
            notes = list(visit.notes)
        return {"body": body, "kind": kind, "file": prepared["file"], "bytes": len(payload), "scope": "visit", "updates": notes}

    def visit_updates(self, session_id: str) -> list[dict]:
        visit = self._visit(session_id)
        return list(visit.notes) if visit else []

    def matcher(self, mode: str, source: str, session_id: str = "") -> TerrainMatcher:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        visit = self._visit(session_id)
        if visit and visit.elevation:
            key = (source, mode)
            with self._lock:
                cached = visit.matchers.get(key)
                if cached is None:
                    cfg = MatchConfig(**{**self.base_config.__dict__, "mode": mode})
                    maps = dict(self.datasets[source])
                    maps.update({body: hm for body, hm in visit.elevation.items() if body in maps})
                    fs = {}
                    for body, hm in maps.items():
                        if body in visit.elevation:
                            fs[body] = compute_features(hm, cfg.scales_km)
                        else:
                            if (source, body) not in self._features:
                                self._features[(source, body)] = compute_features(hm, cfg.scales_km)
                            fs[body] = self._features[(source, body)]
                    cached = TerrainMatcher(list(maps.values()), cfg, feature_sets=fs)
                    visit.matchers[key] = cached
                return cached
        key = (source, mode)
        with self._lock:
            if key not in self._matchers:
                cfg = MatchConfig(**{**self.base_config.__dict__, "mode": mode})
                maps = self.datasets[source]
                for body, hm in maps.items():
                    if (source, body) not in self._features:
                        self._features[(source, body)] = compute_features(hm, cfg.scales_km)
                fs = {body: self._features[(source, body)] for body in maps}
                self._matchers[key] = TerrainMatcher(list(maps.values()), cfg, feature_sets=fs)
            return self._matchers[key]

    def _build_info(self) -> dict:
        out = {
            "modes": list(MODES),
            "surfaces": list(SURFACES),
            "default_mode": self.base_config.mode,
            "default_source": self.default_source,
            "sources": {},
        }
        for src, maps in self.datasets.items():
            bodies = {}
            for key, hm in maps.items():
                b = hm.body
                bodies[key] = {
                    "name": b.name,
                    "source_name": hm.source_name,
                    "lat_range": hm.lat_range,
                    "lon_range": hm.lon_range,
                    "width": hm.shape[1],
                    "height": hm.shape[0],
                    "pixel_km": hm.km_per_px[0],
                    "color_min_m": b.min_elevation_m,
                    "color_max_m": b.max_elevation_m,
                    "landmarks": [
                        {"kind": kind, "name": lm.name, "lat": lm.lat, "lon": lm.lon, "elevation_m": lm.elevation_m}
                        for kind, lm in (("highest", b.highest), ("lowest", b.lowest))
                    ],
                    "calibration": calibration_check(hm),
                }
            out["sources"][src] = {"label": SOURCE_LABELS.get(src, src), "bodies": bodies}
        if "dem" in self.datasets:
            out["validation"] = [
                {**compare_to_dem(self.datasets[src][b], self.datasets["dem"][b]), "source": src}
                for b in self.datasets["dem"]
                for src in ("images", "images-fit")
                if b in self.datasets.get(src, {})
            ]
        out["updates"] = active_updates()
        return out

    def info(self) -> dict:
        return self._info

    def map_png(self, source: str, body: str, session_id: str = "") -> bytes:
        visit = self._visit(session_id)
        key = (source, body)
        if visit and key in visit.pngs:
            return visit.pngs[key]
        if key not in self._pngs:
            raise ValueError(f"unknown map {source}/{body}")
        return self._pngs[key]

    def layer_bytes(self, body: str, layer: str, session_id: str = "") -> bytes:
        visit = self._visit(session_id)
        if visit and (body, layer) in visit.layers:
            return visit.layers[(body, layer)]
        return layer_png(body, layer)

    def _map_hm(self, body: str, source: str | None, session_id: str) -> Heightmap | None:
        body = body.lower()
        visit = self._visit(session_id)
        if visit and body in visit.elevation:
            return visit.elevation[body]
        return self.datasets[self._source(source)].get(body)

    def place(self, body: str, lat: float, lon: float, source: str | None = None, session_id: str = "") -> dict:
        """Name, description and pictures of the place at a coordinate."""
        body = body.lower()
        visit = self._visit(session_id)
        hm = visit.elevation.get(body) if visit and body in visit.elevation else None
        hm = hm or self.datasets.get("dem", {}).get(body) or self.datasets[self._source(source)].get(body)
        if hm is None:
            raise ValueError(f"unknown body {body!r}")
        r, c = hm.pixel_of(lat, lon)
        elev = float(hm.elevation[r, c])
        return self.places.describe(body, lat, lon, elev if np.isfinite(elev) else None, pixel_km=hm.km_per_px[0])

    def place_image(self, body: str, lat: float, lon: float, half_km: float) -> bytes:
        if body not in ("earth", "moon", "mars"):
            raise ValueError(f"unknown body {body!r}")
        return orbital_view(body, round(lat, 3), round(lon, 3), float(np.clip(half_km, 20, 1500)))

    def _safety_hm(self, body: str, session_id: str = "") -> Heightmap:
        body = body.lower()
        visit = self._visit(session_id)
        if visit and body in visit.elevation:
            return visit.elevation[body]
        if body in self.datasets.get("dem", {}):
            return self.datasets["dem"][body]
        for maps in self.datasets.values():
            if body in maps:
                return maps[body]
        raise ValueError(f"unknown body {body!r}")

    def safety_grid(self, body: str, session_id: str = ""):
        body = body.lower()
        visit = self._visit(session_id)
        personal = bool(visit and (body in visit.elevation or any(key[0] == body for key in visit.layers)))
        with self._lock:
            store = visit.safety if personal and visit is not None else self._safety
            cached = store.get(body)
            if cached is None:
                hm = self._safety_hm(body, session_id)
                layers = {kind: data for (owner, kind), data in visit.layers.items() if owner == body} if personal and visit is not None else None
                grid = build_safety(hm, layers)
                cached = (grid, encode_png(grid))
                store[body] = cached
            return cached

    def safety_png(self, body: str, session_id: str = "") -> bytes:
        return self.safety_grid(body, session_id)[1]

    def safety_at(self, body: str, lat: float, lon: float, weights: dict[str, float] | None = None, session_id: str = "") -> dict:
        hm = self._safety_hm(body, session_id)
        grid, _png = self.safety_grid(body, session_id)
        cell = describe_cell(hm, grid, lat, lon, weights)
        cell["body"] = hm.body.name
        cell["lat"] = lat
        cell["lon"] = lon
        return cell

    def height(self, body: str, lat: float, lon: float, source: str | None = None, session_id: str = "") -> dict:
        hm = self._map_hm(body, source, session_id)
        if hm is None:
            raise ValueError(f"unknown body {body!r}")
        r, c = hm.pixel_of(lat, lon)
        elev = float(hm.elevation[r, c])
        payload = {
            "body": hm.body.name,
            "lat": lat,
            "lon": lon,
            "elevation_m": elev,
            "gray": float(hm.gray[r, c]),
            "surface": describe_surface(hm.body.name, elev) if math.isfinite(elev) else "no data",
        }
        try:
            weights = {name: float(25) for name in FACTORS}
            payload["safety"] = self.safety_at(body, lat, lon, weights, session_id)
        except ValueError:
            pass
        return payload

    def ground(self, body: str, lat: float, lon: float, km: float = SPAN_KM, session_id: str = "", samples: int | None = None) -> dict:
        hm = self._map_hm(body, "dem", session_id)
        if hm is None:
            raise ValueError(f"unknown body {body!r}")
        if samples:
            return ground_grid(hm, lat, lon, km, samples)
        return ground_grid(hm, lat, lon, km)

    def ground_image(self, body: str, lat: float, lon: float, layer: str, km: float = SPAN_KM, session_id: str = "", pixels: int = 512) -> bytes:
        hm = self._map_hm(body, "dem", session_id)
        if hm is None:
            raise ValueError(f"unknown body {body!r}")
        if body.lower() == "earth" and layer == "elevation":
            photo = satellite_png(lat, lon, hm.body.mean_radius_km, km, pixels=pixels)
            if photo:
                return photo
        raw = self.map_png("dem", body, session_id) if layer == "elevation" else self.layer_bytes(body, layer, session_id)
        return ground_png(raw, lat, lon, hm.body.mean_radius_km, km)

    def query(self, body: str, lat: float, lon: float, k: int, mode: str, source: str | None, surface: str,
              target: str | None = None, session_id: str = "") -> dict:
        m = self.matcher(mode, self._source(source), session_id)
        target = target or m.default_target(body)
        origin, results = m.query(body, lat, lon, k, surface=surface, target=target)
        return self._with_maps({
            "origin": _site(origin),
            "matches": [_match(x) for x in results],
            "steps": explain_query(m, body, origin, results, surface, target),
        }, session_id)

    def apply_update(self, body: str, kind: str) -> dict:
        """Drop cached maps for one body and load the replacement that was just saved."""
        body = body.lower()
        with self._lock:
            replaced = [src for src, maps in self.datasets.items() if body in maps]
            if not replaced:
                raise ValueError(f"unknown body {body!r}")
            if kind == "elevation":
                hm = load_dem(body)
                for src in replaced:
                    self.datasets[src][body] = hm
                    self._pngs[(src, body)] = render_map_png(hm, CMAPS.get(body, "viridis"))
                self._features = {key: value for key, value in self._features.items() if key[1] != body}
                self._matchers.clear()
                self._top.clear()
            else:
                from .layers import layer_path

                layer_path(body, kind)
            self._safety.pop(body, None)
            self._info = self._build_info()
        return {"replaced": True, "body": body, "kind": kind, "sources": replaced, "updates": active_updates()}

    def top(self, n: int, mode: str, source: str | None, surface: str, a: str, b: str, session_id: str = "") -> dict:
        source = self._source(source)
        visit = self._visit(session_id)
        key = (source, mode, surface, n, a, b)
        store = visit.top if visit and visit.elevation else self._top
        if key not in store:
            m = self.matcher(mode, source, session_id if visit and visit.elevation else "")
            matches = m.top_matches(n, surface=surface, a=a, b=b)
            store[key] = {"matches": [_match(x) for x in matches], "steps": explain_top(m, matches, surface)}
        return self._with_maps(store[key], session_id)

    def _with_maps(self, payload: dict, session_id: str) -> dict:
        visit = self._visit(session_id)
        overrides = dict(visit.layers) if visit else {}
        matches = []
        for row in payload["matches"]:
            copied = dict(row)
            left, right = copied["a"], copied["b"]
            copied["maps"] = compare_maps(
                left["body"].lower(), left["lat"], left["lon"],
                right["body"].lower(), right["lat"], right["lon"],
                left["features"]["elevation_m"], right["features"]["elevation_m"],
                overrides,
            )
            matches.append(copied)
        return {**payload, "matches": matches}


def make_handler(app: TerrainApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _session(self) -> str:
            match = re.search(rf"(?:^|;\s*){SESSION_COOKIE}=([A-Za-z0-9_-]{{16,80}})", self.headers.get("Cookie") or "")
            return match.group(1) if match else ""

        def _send(self, status: int, body: bytes, ctype: str, cache: bool = False, session_cookie: str = ""):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
            if session_cookie:
                self.send_header("Set-Cookie", f"{SESSION_COOKIE}={session_cookie}; Path=/; HttpOnly; SameSite=Lax")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload, session_cookie: str = ""):
            self._send(status, json.dumps(_clean(payload)).encode(), "application/json", session_cookie=session_cookie)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            path = url.path
            mode = q.get("mode", app.base_config.mode)
            source = q.get("source")
            surface = q.get("surface", "any")
            try:
                if path in ("/", "/index.html"):
                    self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/favicon.ico":
                    self._send(204, b"", "image/x-icon", cache=True)
                elif path == "/api/layers":
                    self._json(200, catalog())
                elif path == "/api/maps":
                    self._json(200, {"updates": app.visit_updates(self._session()), "scope": "visit"})
                elif path.startswith("/map/layer/") and path.endswith(".png"):
                    body, layer = path[len("/map/layer/"):-4].split("/")
                    session = self._session()
                    personal = bool(app._visit(session) and (body, layer) in app._visit(session).layers) if session else False
                    self._send(200, app.layer_bytes(body, layer, session), "image/png", cache=not personal)
                elif path.startswith("/map/") and path.endswith(".png"):
                    parts = path[len("/map/"):-4].split("/")
                    src, body = (parts if len(parts) == 2 else (app.default_source, parts[0]))
                    session = self._session()
                    visit = app._visit(session)
                    personal = bool(visit and (src, body) in visit.pngs)
                    self._send(200, app.map_png(src, body, session), "image/png", cache=not personal)
                elif path == "/api/info":
                    self._json(200, app.info())
                elif path == "/api/height":
                    self._json(200, app.height(q["body"], float(q["lat"]), float(q["lon"]), source, self._session()))
                elif path == "/api/safety":
                    weights = {name: float(q[name]) for name in FACTORS if name in q}
                    self._json(200, app.safety_at(q["body"], float(q["lat"]), float(q["lon"]), weights or None, self._session()))
                elif path.startswith("/api/safety/") and path.endswith(".png"):
                    self._send(200, app.safety_png(path[len("/api/safety/"):-4], self._session()), "image/png")
                elif path == "/api/place":
                    self._json(200, app.place(q["body"], float(q["lat"]), float(q["lon"]), source, self._session()))
                elif path == "/api/ground":
                    samples = int(q["samples"]) if q.get("samples") else None
                    self._json(200, app.ground(q["body"], float(q["lat"]), float(q["lon"]), float(q.get("km", SPAN_KM)), self._session(), samples))
                elif path == "/api/ground.png":
                    image = app.ground_image(q["body"], float(q["lat"]), float(q["lon"]), q.get("layer", "elevation"), float(q.get("km", SPAN_KM)), self._session(), int(q.get("px", 512)))
                    self._send(200, image, "image/png")
                elif path == "/api/place-image":
                    img = app.place_image(q["body"], float(q["lat"]), float(q["lon"]), float(q.get("km", 150)))
                    self._send(200, img, "image/jpeg", cache=True)
                elif path == "/api/query":
                    k = min(int(q.get("k", 5)), 25)
                    self._json(200, app.query(q["body"], float(q["lat"]), float(q["lon"]), k, mode, source, surface,
                                              q.get("target"), self._session()))
                elif path == "/api/top":
                    self._json(200, app.top(min(int(q.get("n", 10)), 50), mode, source, surface,
                                            q.get("a", "earth"), q.get("b", "moon"), self._session()))
                else:
                    self._json(404, {"error": "not found"})
            except (ValueError, KeyError) as exc:
                self._json(400, {"error": str(exc) if not isinstance(exc, KeyError) else f"missing {exc}"})

        def do_POST(self):
            url = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            if url.path != "/api/maps":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > 80_000_000:
                self._json(400, {"error": "send the map file in the request body, up to 80 MB"})
                return
            raw = self.rfile.read(length)
            try:
                prepared = prepare_map(q["body"], q["kind"], raw, q.get("name", ""))
                session = self._session() or secrets.token_urlsafe(24)
                minted = self._session() == ""
                self._json(200, app.remember_map(session, prepared), session_cookie=session if minted else "")
            except (ValueError, KeyError) as exc:
                self._json(400, {"error": str(exc) if not isinstance(exc, KeyError) else f"missing {exc}"})

    return Handler


def serve(app: TerrainApp, host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"Terrain bot UI running at http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
