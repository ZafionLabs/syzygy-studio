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

from .heightmap import Heightmap
from .matcher import MODES, Match, MatchConfig, SitePoint, TerrainMatcher
from .report import calibration_check

STATIC = Path(__file__).resolve().parent / "static"
CMAPS = {"earth": "gist_earth", "moon": "cividis"}


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
    return {"body": p.body, "lat": p.lat, "lon": p.lon, "gray": p.gray, "features": p.features}


def _match(m: Match) -> dict:
    return {
        "rank": m.rank,
        "similarity": m.similarity,
        "elevation_difference_m": m.elevation_difference_m,
        "a": _site(m.a),
        "b": _site(m.b),
    }


def render_map_png(hm: Heightmap, cmap_name: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    t = (hm.elevation - hm.body.min_elevation_m) / hm.body.relief_m
    rgba = colormaps[cmap_name](np.nan_to_num(t, nan=0.0))
    rgba[..., 3] = np.where(hm.valid, 1.0, 0.0)
    buf = io.BytesIO()
    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(buf, format="PNG")
    return buf.getvalue()


class TerrainApp:
    """Holds one matcher per mode, built lazily from the same calibrated maps."""

    def __init__(self, earth: Heightmap, moon: Heightmap, config: MatchConfig):
        self.maps = {"earth": earth, "moon": moon}
        self.base_config = config
        self._matchers: dict[str, TerrainMatcher] = {}
        self._lock = threading.Lock()
        self._pngs = {k: render_map_png(hm, CMAPS[k]) for k, hm in self.maps.items()}
        self.matcher(config.mode)

    def matcher(self, mode: str) -> TerrainMatcher:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self._lock:
            if mode not in self._matchers:
                cfg = MatchConfig(**{**self.base_config.__dict__, "mode": mode})
                self._matchers[mode] = TerrainMatcher(self.maps["earth"], self.maps["moon"], cfg)
            return self._matchers[mode]

    def info(self) -> dict:
        out = {"modes": list(MODES), "default_mode": self.base_config.mode, "bodies": {}}
        for key, hm in self.maps.items():
            b = hm.body
            out["bodies"][key] = {
                "name": b.name,
                "lat_range": hm.lat_range,
                "lon_range": hm.lon_range,
                "width": hm.shape[1],
                "height": hm.shape[0],
                "min_elevation_m": b.min_elevation_m,
                "max_elevation_m": b.max_elevation_m,
                "landmarks": [
                    {"kind": kind, "name": lm.name, "lat": lm.lat, "lon": lm.lon, "elevation_m": lm.elevation_m}
                    for kind, lm in (("highest", b.highest), ("lowest", b.lowest))
                ],
                "calibration": calibration_check(hm),
            }
        return out

    def map_png(self, body: str) -> bytes:
        if body not in self._pngs:
            raise ValueError(f"unknown body {body!r}")
        return self._pngs[body]

    def height(self, body: str, lat: float, lon: float) -> dict:
        hm = self.maps.get(body)
        if hm is None:
            raise ValueError(f"unknown body {body!r}")
        r, c = hm.pixel_of(lat, lon)
        return {"body": hm.body.name, "lat": lat, "lon": lon, "elevation_m": float(hm.elevation[r, c]),
                "gray": float(hm.gray[r, c])}

    def query(self, body: str, lat: float, lon: float, k: int, mode: str) -> dict:
        origin, results = self.matcher(mode).query(body, lat, lon, k)
        return {"origin": _site(origin), "matches": [_match(m) for m in results]}

    def top(self, n: int, mode: str) -> dict:
        return {"matches": [_match(m) for m in self.matcher(mode).top_matches(n)]}


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
            try:
                if path in ("/", "/index.html"):
                    self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/favicon.ico":
                    self._send(204, b"", "image/x-icon", cache=True)
                elif path.startswith("/map/") and path.endswith(".png"):
                    self._send(200, app.map_png(path[len("/map/"):-4]), "image/png", cache=True)
                elif path == "/api/info":
                    self._json(200, app.info())
                elif path == "/api/height":
                    self._json(200, app.height(q["body"], float(q["lat"]), float(q["lon"])))
                elif path == "/api/query":
                    self._json(200, app.query(q["body"], float(q["lat"]), float(q["lon"]),
                                              min(int(q.get("k", 5)), 25), q.get("mode", app.base_config.mode)))
                elif path == "/api/top":
                    self._json(200, app.top(min(int(q.get("n", 10)), 50), q.get("mode", app.base_config.mode)))
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
