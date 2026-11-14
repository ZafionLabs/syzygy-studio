"""Small browser UI for the terrain bot (standard library HTTP server)."""

from __future__ import annotations

import io
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .dem import compare_to_dem
from .features import compute_features
from .explain import describe_surface, explain_query, explain_top, feature_label, feature_unit
from .heightmap import Heightmap
from .layers import catalog, layer_png
from .matcher import MODES, SURFACES, Match, MatchConfig, SitePoint, TerrainMatcher
from .report import calibration_check

STATIC = Path(__file__).resolve().parent / "static"
CMAPS = {"earth": "gist_earth", "moon": "cividis", "mars": "copper"}
SOURCE_LABELS = {
    "dem": "Real measured data (NOAA ETOPO1, NASA LOLA, NASA MOLA)",
    "images": "Your images, darkest = lowest point, lightest = highest point",
    "images-fit": "Your images, brightness curve fitted to real data",
}
SOURCE_ORDER = ("dem", "images", "images-fit")


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

    def __init__(self, datasets: dict[str, dict[str, Heightmap]], config: MatchConfig, default_source: str):
        self.datasets = {k: datasets[k] for k in sorted(datasets, key=lambda k: SOURCE_ORDER.index(k) if k in SOURCE_ORDER else 99)}
        self._features: dict[tuple[str, str], object] = {}
        self.base_config = config
        self.default_source = default_source
        self._matchers: dict[tuple[str, str], TerrainMatcher] = {}
        self._top: dict[tuple, dict] = {}
        self._lock = threading.Lock()
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

    def matcher(self, mode: str, source: str) -> TerrainMatcher:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
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
        return out

    def info(self) -> dict:
        return self._info

    def map_png(self, source: str, body: str) -> bytes:
        key = (source, body)
        if key not in self._pngs:
            raise ValueError(f"unknown map {source}/{body}")
        return self._pngs[key]

    def height(self, body: str, lat: float, lon: float, source: str | None = None) -> dict:
        hm = self.datasets[self._source(source)].get(body)
        if hm is None:
            raise ValueError(f"unknown body {body!r}")
        r, c = hm.pixel_of(lat, lon)
        elev = float(hm.elevation[r, c])
        return {
            "body": hm.body.name,
            "lat": lat,
            "lon": lon,
            "elevation_m": elev,
            "gray": float(hm.gray[r, c]),
            "surface": describe_surface(hm.body.name, elev) if math.isfinite(elev) else "no data",
        }

    def query(self, body: str, lat: float, lon: float, k: int, mode: str, source: str | None, surface: str,
              target: str | None = None) -> dict:
        m = self.matcher(mode, self._source(source))
        target = target or m.default_target(body)
        origin, results = m.query(body, lat, lon, k, surface=surface, target=target)
        return {
            "origin": _site(origin),
            "matches": [_match(x) for x in results],
            "steps": explain_query(m, body, origin, results, surface, target),
        }

    def top(self, n: int, mode: str, source: str | None, surface: str, a: str, b: str) -> dict:
        source = self._source(source)
        key = (source, mode, surface, n, a, b)
        if key not in self._top:
            m = self.matcher(mode, source)
            matches = m.top_matches(n, surface=surface, a=a, b=b)
            self._top[key] = {"matches": [_match(x) for x in matches], "steps": explain_top(m, matches, surface)}
        return self._top[key]


def make_handler(app: TerrainApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, status: int, body: bytes, ctype: str, cache: bool = False):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload):
            self._send(status, json.dumps(_clean(payload)).encode(), "application/json")

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
                elif path.startswith("/map/layer/") and path.endswith(".png"):
                    body, layer = path[len("/map/layer/"):-4].split("/")
                    self._send(200, layer_png(body, layer), "image/png")
                elif path.startswith("/map/") and path.endswith(".png"):
                    parts = path[len("/map/"):-4].split("/")
                    src, body = (parts if len(parts) == 2 else (app.default_source, parts[0]))
                    self._send(200, app.map_png(src, body), "image/png", cache=True)
                elif path == "/api/info":
                    self._json(200, app.info())
                elif path == "/api/height":
                    self._json(200, app.height(q["body"], float(q["lat"]), float(q["lon"]), source))
                elif path == "/api/query":
                    k = min(int(q.get("k", 5)), 25)
                    self._json(200, app.query(q["body"], float(q["lat"]), float(q["lon"]), k, mode, source, surface,
                                              q.get("target")))
                elif path == "/api/top":
                    self._json(200, app.top(min(int(q.get("n", 10)), 50), mode, source, surface,
                                            q.get("a", "earth"), q.get("b", "moon")))
                else:
                    self._json(404, {"error": "not found"})
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
