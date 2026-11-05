"""Command-line interface and interactive chat bot."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .bodies import EARTH, MOON
from .features import DEFAULT_SCALES_KM
from .heightmap import load_heightmap
from .matcher import MODES, MatchConfig, TerrainMatcher
from .report import calibration_check, format_calibration, format_matches, write_report

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--earth", default=str(DATA_DIR / "earth_heightmap.png"), help="Earth grayscale heightmap image")
    p.add_argument("--moon", default=str(DATA_DIR / "moon_heightmap.png"), help="Moon grayscale heightmap image")
    for body in ("earth", "moon"):
        p.add_argument(f"--{body}-lat-range", nargs=2, type=float, default=(-90.0, 90.0), metavar=("SOUTH", "NORTH"))
        p.add_argument(f"--{body}-lon-range", nargs=2, type=float, default=(-180.0, 180.0), metavar=("WEST", "EAST"))
    p.add_argument("--clip-percentiles", nargs=2, type=float, default=(0.0, 100.0), metavar=("LOW", "HIGH"),
                   help="brightness percentiles mapped to the lowest/highest real elevation (default: true min/max)")
    p.add_argument("--gamma", type=float, default=1.0, help="power curve applied to brightness before scaling")
    p.add_argument("--mode", choices=MODES, default="terrain",
                   help="'terrain' compares elevation plus local relief/slope; 'elevation' compares height only")
    p.add_argument("--scales-km", nargs="+", type=float, default=list(DEFAULT_SCALES_KM),
                   help="neighbourhood sizes (ground km) used for terrain descriptors")
    p.add_argument("--min-separation-deg", type=float, default=5.0,
                   help="minimum arc distance between reported matches on each body")


def _load(args) -> tuple:
    clip = tuple(args.clip_percentiles)
    earth = load_heightmap(args.earth, EARTH, tuple(args.earth_lat_range), tuple(args.earth_lon_range), clip, args.gamma)
    moon = load_heightmap(args.moon, MOON, tuple(args.moon_lat_range), tuple(args.moon_lon_range), clip, args.gamma)
    return earth, moon


def _matcher(args) -> TerrainMatcher:
    earth, moon = _load(args)
    cfg = MatchConfig(mode=args.mode, scales_km=tuple(args.scales_km), min_separation_deg=args.min_separation_deg)
    return TerrainMatcher(earth, moon, cfg)


def cmd_calibrate(args) -> int:
    for hm in _load(args):
        print(format_calibration(calibration_check(hm)))
        print()
    return 0


def cmd_analyze(args) -> int:
    matcher = _matcher(args)
    matches = matcher.top_matches(args.top)
    paths = write_report(matcher, matches, Path(args.out), plots=not args.no_plots)
    print(paths["summary"].read_text())
    print("Wrote:", ", ".join(str(p) for p in paths.values()))
    return 0


def cmd_query(args) -> int:
    matcher = _matcher(args)
    origin, results = matcher.query(args.body, args.lat, args.lon, args.top)
    print(_describe_origin(origin))
    print(format_matches(results))
    return 0


def _describe_origin(origin) -> str:
    f = origin.features
    extra = ", ".join(f"{k}={v:,.1f}" for k, v in f.items() if k != "elevation_m")
    return (
        f"{origin.body} at {origin.lat:.2f}, {origin.lon:.2f}: elevation {origin.elevation_m:,.0f} m "
        f"(gray {origin.gray:.0f}); {extra}"
    )


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
  everest | mariana | selenean | antoniadi   (landmark shortcuts)
  height earth 36.1 -112.1 -> calibrated elevation at a point
  top 10                   -> globally most similar Earth/Moon pairs
  calibration              -> how gray levels map to real elevations
  help | quit"""


def handle_chat(matcher: TerrainMatcher, line: str, k: int = 5) -> str:
    text = line.strip().lower()
    if not text:
        return ""
    if text in ("help", "?"):
        return CHAT_HELP
    if text.startswith("calib"):
        return "\n".join(format_calibration(calibration_check(hm)) for hm in (matcher.a.hm, matcher.b.hm))

    numbers = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    if text.startswith("top"):
        n = int(numbers[0]) if numbers else 10
        return format_matches(matcher.top_matches(n))

    for key, (body, lm) in LANDMARKS.items():
        if key in text:
            origin, results = matcher.query(body, lm.lat, lm.lon, k)
            return f"{lm.name} (real: {lm.elevation_m:,.0f} m)\n{_describe_origin(origin)}\n{format_matches(results)}"

    body = "moon" if "moon" in text or "lunar" in text else "earth" if "earth" in text else None
    if body is None or len(numbers) < 2:
        return "I need a body and a coordinate, e.g. 'earth 27.99 86.93'. Type 'help' for more."
    lat, lon = numbers[0], numbers[1]
    try:
        if text.startswith(("height", "elev")):
            hm = matcher.a.hm if body == "earth" else matcher.b.hm
            return f"{hm.body.name} at {lat:.2f}, {lon:.2f}: {hm.elevation_at(lat, lon):,.0f} m"
        origin, results = matcher.query(body, lat, lon, k)
    except ValueError as exc:
        return str(exc)
    return f"{_describe_origin(origin)}\n{format_matches(results)}"


def cmd_chat(args) -> int:
    print("Loading heightmaps and building terrain index...")
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

    print("Loading heightmaps and building terrain index...")
    earth, moon = _load(args)
    cfg = MatchConfig(mode=args.mode, scales_km=tuple(args.scales_km), min_separation_deg=args.min_separation_deg)
    serve(TerrainApp(earth, moon, cfg), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terrain-bot",
        description="Calibrate Earth/Moon grayscale heightmaps to real elevations and find look-alike terrain.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="find the globally most similar Earth/Moon locations")
    _add_common(p)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out", default="results")
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("query", help="find places on the other body most similar to one coordinate")
    _add_common(p)
    p.add_argument("--body", choices=("earth", "moon"), required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--top", type=int, default=5)
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("calibrate", help="show the gray-to-elevation calibration and landmark checks")
    _add_common(p)
    p.set_defaults(func=cmd_calibrate)

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
