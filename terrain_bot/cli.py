"""Command-line interface and interactive chat bot."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import dem
from .bodies import EARTH, MOON
from .explain import describe_surface, explain_query, explain_top, format_steps
from .features import DEFAULT_SCALES_KM
from .heightmap import load_heightmap
from .matcher import MODES, SURFACES, MatchConfig, TerrainMatcher
from .report import calibration_check, format_calibration, format_matches, write_report

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCES = ("dem", "images")


def _default_source() -> str:
    return "dem" if dem.dem_available("earth") and dem.dem_available("moon") else "images"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", choices=SOURCES, default=_default_source(),
                   help="'dem' = measured elevations (NOAA ETOPO1 + NASA LOLA); 'images' = calibrated grayscale images")
    p.add_argument("--earth", default=str(DATA_DIR / "earth_heightmap.png"), help="Earth grayscale heightmap image")
    p.add_argument("--moon", default=str(DATA_DIR / "moon_heightmap.png"), help="Moon grayscale heightmap image")
    for body in ("earth", "moon"):
        p.add_argument(f"--{body}-lat-range", nargs=2, type=float, default=None, metavar=("SOUTH", "NORTH"),
                       help="image latitude bounds (default: sidecar .json next to the image, else -90 90)")
        p.add_argument(f"--{body}-lon-range", nargs=2, type=float, default=None, metavar=("WEST", "EAST"))
    p.add_argument("--clip-percentiles", nargs=2, type=float, default=(0.0, 100.0), metavar=("LOW", "HIGH"),
                   help="brightness percentiles mapped to the lowest/highest real elevation (default: true min/max)")
    p.add_argument("--gamma", type=float, default=1.0, help="power curve applied to brightness before scaling")
    p.add_argument("--mode", choices=MODES, default="terrain",
                   help="'terrain' compares elevation plus local relief/slope; 'elevation' compares height only")
    p.add_argument("--surface", choices=SURFACES, default="any",
                   help="restrict Earth points to land (>= sea level) or underwater (< sea level)")
    p.add_argument("--scales-km", nargs="+", type=float, default=list(DEFAULT_SCALES_KM),
                   help="neighbourhood sizes (ground km) used for terrain descriptors")
    p.add_argument("--min-separation-deg", type=float, default=5.0,
                   help="minimum arc distance between reported matches on each body")


def _load_images(args) -> tuple:
    clip = tuple(args.clip_percentiles)
    earth = load_heightmap(args.earth, EARTH, args.earth_lat_range, args.earth_lon_range, clip, args.gamma)
    moon = load_heightmap(args.moon, MOON, args.moon_lat_range, args.moon_lon_range, clip, args.gamma)
    return earth, moon


def _load(args) -> tuple:
    if args.source == "dem":
        return dem.load_dem("earth"), dem.load_dem("moon")
    return _load_images(args)


def _config(args) -> MatchConfig:
    return MatchConfig(mode=args.mode, scales_km=tuple(args.scales_km), min_separation_deg=args.min_separation_deg)


def _matcher(args) -> TerrainMatcher:
    earth, moon = _load(args)
    return TerrainMatcher(earth, moon, _config(args))


def cmd_calibrate(args) -> int:
    for hm in _load(args):
        print(format_calibration(calibration_check(hm)))
        print()
    return 0


def cmd_validate(args) -> int:
    images = _load_images(args)
    for body, hm in zip(("earth", "moon"), images):
        print(dem.format_comparison(dem.compare_to_dem(hm, dem.load_dem(body))))
    return 0


def cmd_download_dem(args) -> int:
    for body in ("earth", "moon"):
        print(f"Downloading {dem.DEM_SOURCES[body].name} ...")
        print("  saved", dem.download_dem(body))
    return 0


def cmd_analyze(args) -> int:
    matcher = _matcher(args)
    matches = matcher.top_matches(args.top, surface=args.surface)
    paths = write_report(matcher, matches, Path(args.out), plots=not args.no_plots)
    print(paths["summary"].read_text())
    if not args.no_steps:
        print(format_steps(explain_top(matcher, matches, args.surface)))
    print("\nWrote:", ", ".join(str(p) for p in paths.values()))
    return 0


def cmd_query(args) -> int:
    matcher = _matcher(args)
    origin, results = matcher.query(args.body, args.lat, args.lon, args.top, surface=args.surface)
    print(_describe_origin(origin))
    print(format_matches(results))
    if not args.no_steps:
        print()
        print(format_steps(explain_query(matcher, args.body, origin, results, args.surface)))
    return 0


def _describe_origin(origin) -> str:
    f = origin.features
    extra = ", ".join(f"{k}={v:,.1f}" for k, v in f.items() if k != "elevation_m")
    return f"{origin.body} at {origin.lat:.2f}, {origin.lon:.2f}: {describe_surface(origin.body, origin.elevation_m)}; {extra}"


LANDMARKS = {
    "everest": ("earth", EARTH.highest),
    "mariana": ("earth", EARTH.lowest),
    "challenger": ("earth", EARTH.lowest),
    "selenean": ("moon", MOON.highest),
    "antoniadi": ("moon", MOON.lowest),
    "trieste": ("moon", MOON.lowest),
}

CHAT_HELP = """\
Ask me things like:
  earth 27.99 86.93        -> most similar places on the Moon to that Earth point
  moon -70.4 -172.5        -> most similar places on Earth to that Moon point
  moon 0 0 underwater      -> only Earth sea-floor matches ('land' for dry land only)
  everest | mariana | selenean | antoniadi   (landmark shortcuts)
  ... steps                -> add 'steps' to any query to see how it was worked out
  height earth 36.1 -112.1 -> elevation at a point
  top 10                   -> globally most similar Earth/Moon pairs
  calibration              -> data sources and calibration
  help | quit"""


def handle_chat(matcher: TerrainMatcher, line: str, k: int = 5) -> str:
    text = line.strip().lower()
    if not text:
        return ""
    if text in ("help", "?"):
        return CHAT_HELP
    if text.startswith("calib"):
        return "\n".join(format_calibration(calibration_check(hm)) for hm in (matcher.a.hm, matcher.b.hm))

    surface = "underwater" if re.search(r"under ?water|ocean|sea ?floor", text) else "land" if re.search(r"\bland\b", text) else "any"
    steps = bool(re.search(r"\b(steps?|how|explain)\b", text))
    numbers = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]

    if text.startswith("top"):
        n = int(numbers[0]) if numbers else 10
        matches = matcher.top_matches(n, surface=surface)
        out = format_matches(matches)
        return out + ("\n\n" + format_steps(explain_top(matcher, matches, surface)) if steps else "")

    def answer(body, lat, lon, title=""):
        origin, results = matcher.query(body, lat, lon, k, surface=surface)
        out = (title + "\n" if title else "") + f"{_describe_origin(origin)}\n{format_matches(results)}"
        if steps:
            out += "\n\n" + format_steps(explain_query(matcher, body, origin, results, surface))
        return out

    for key, (body, lm) in LANDMARKS.items():
        if key in text:
            return answer(body, lm.lat, lm.lon, f"{lm.name} (real: {lm.elevation_m:,.0f} m)")

    body = "moon" if "moon" in text or "lunar" in text else "earth" if "earth" in text else None
    if body is None or len(numbers) < 2:
        return "I need a body and a coordinate, e.g. 'earth 27.99 86.93'. Type 'help' for more."
    lat, lon = numbers[0], numbers[1]
    try:
        if text.startswith(("height", "elev")):
            hm = matcher.a.hm if body == "earth" else matcher.b.hm
            return f"{hm.body.name} at {lat:.2f}, {lon:.2f}: {describe_surface(hm.body.name, hm.elevation_at(lat, lon))}"
        return answer(body, lat, lon)
    except ValueError as exc:
        return str(exc)


def cmd_chat(args) -> int:
    print("Loading elevation data and building terrain index...")
    matcher = _matcher(args)
    print("Terrain bot ready.\n" + CHAT_HELP)
    while True:
        try:
            line = input("\nterrain-bot> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip().lower() in ("quit", "exit", "q"):
            return 0
        try:
            print(handle_chat(matcher, line, args.top))
        except ValueError as exc:
            print(exc)


def cmd_serve(args) -> int:
    from .web import TerrainApp, serve

    print("Loading elevation data and building terrain index...")
    datasets = {"images": _load_images(args)}
    if dem.dem_available("earth") and dem.dem_available("moon"):
        datasets["dem"] = (dem.load_dem("earth"), dem.load_dem("moon"))
    default = args.source if args.source in datasets else "images"
    serve(TerrainApp(datasets, _config(args), default_source=default), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terrain-bot",
        description="Find look-alike Earth/Moon terrain using real elevations (measured DEMs or calibrated images).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="find the globally most similar Earth/Moon locations")
    _add_common(p)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out", default="results")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-steps", action="store_true", help="don't print the step-by-step explanation")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("query", help="find places on the other body most similar to one coordinate")
    _add_common(p)
    p.add_argument("--body", choices=("earth", "moon"), required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--no-steps", action="store_true", help="don't print the step-by-step explanation")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("calibrate", help="show the data source / calibration and landmark checks")
    _add_common(p)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("validate", help="measure how accurate the calibrated images are against real DEMs")
    _add_common(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("download-dem", help="(re)download the real elevation models into data/dem")
    p.set_defaults(func=cmd_download_dem)

    p = sub.add_parser("chat", help="interactive terrain bot")
    _add_common(p)
    p.add_argument("--top", type=int, default=5)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("serve", help="run the browser UI")
    _add_common(p)
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other machines to connect")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
