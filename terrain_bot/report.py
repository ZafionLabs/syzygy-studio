"""Text, CSV/JSON and figure outputs for terrain matches."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .heightmap import Heightmap, haversine_km
from .matcher import Match, TerrainMatcher


def calibration_check(hm: Heightmap) -> dict:
    """Compare the map's brightest/darkest pixels with the real landmark positions."""
    body = hm.body
    ext = hm.extreme_pixels()
    out = {
        "body": body.name,
        "datum": body.datum,
        "gray_range": [hm.gray_min, hm.gray_max],
        "elevation_range_m": [body.min_elevation_m, body.max_elevation_m],
        "metres_per_gray_level": body.relief_m / (hm.gray_max - hm.gray_min),
        "pixel_size_km": list(hm.km_per_px),
    }
    for key, landmark in (("highest", body.highest), ("lowest", body.lowest)):
        r, c = ext[key]
        lat, lon = hm.latlon_of(r, c)
        try:
            at_real = hm.elevation_at(landmark.lat, landmark.lon)
        except ValueError:
            at_real = float("nan")
        out[key] = {
            "landmark": landmark.name,
            "real_elevation_m": landmark.elevation_m,
            "real_lat": landmark.lat,
            "real_lon": landmark.lon,
            "source": landmark.source,
            "map_pixel_lat": lat,
            "map_pixel_lon": lon,
            "map_pixel_offset_km": float(haversine_km(lat, lon, landmark.lat, landmark.lon, body.mean_radius_km)),
            "map_elevation_at_real_location_m": at_real,
        }
    return out


def format_calibration(check: dict) -> str:
    lines = [
        f"{check['body']}: gray {check['gray_range'][0]:.0f}-{check['gray_range'][1]:.0f} -> "
        f"{check['elevation_range_m'][0]:,.0f} m .. {check['elevation_range_m'][1]:,.0f} m "
        f"({check['metres_per_gray_level']:.1f} m per gray level, datum: {check['datum']})",
    ]
    for key in ("highest", "lowest"):
        e = check[key]
        lines.append(
            f"  {key:<7} = {e['landmark']} ({e['real_elevation_m']:,.0f} m). Map's {'lightest' if key == 'highest' else 'darkest'} "
            f"pixel is at {e['map_pixel_lat']:.1f}, {e['map_pixel_lon']:.1f} "
            f"({e['map_pixel_offset_km']:,.0f} km from the real one); map reads "
            f"{e['map_elevation_at_real_location_m']:,.0f} m at the real location."
        )
    return "\n".join(lines)


def _fmt_latlon(lat: float, lon: float) -> str:
    return f"{abs(lat):5.1f}{'N' if lat >= 0 else 'S'} {abs(lon):6.1f}{'E' if lon >= 0 else 'W'}"


def format_matches(matches: list[Match]) -> str:
    if not matches:
        return "(no matches)"
    a_name, b_name = matches[0].a.body, matches[0].b.body
    header = (
        f"{'#':>3}  {'sim%':>5}  {a_name + ' location':<15} {a_name + ' elev':>11}  "
        f"{b_name + ' location':<15} {b_name + ' elev':>11}  {'diff':>8}"
    )
    lines = [header, "-" * len(header)]
    for m in matches:
        lines.append(
            f"{m.rank:>3}  {m.similarity:5.1f}  {_fmt_latlon(m.a.lat, m.a.lon):<15} {m.a.elevation_m:>9,.0f} m  "
            f"{_fmt_latlon(m.b.lat, m.b.lon):<15} {m.b.elevation_m:>9,.0f} m  {m.elevation_difference_m:>+7,.0f}m"
        )
    return "\n".join(lines)


def write_csv(matches: list[Match], path: Path) -> None:
    if not matches:
        path.write_text("")
        return
    feat_names = list(matches[0].a.features)
    a_name, b_name = matches[0].a.body.lower(), matches[0].b.body.lower()
    fields = ["rank", "similarity", "distance", "elevation_difference_m"]
    for side in (a_name, b_name):
        fields += [f"{side}_lat", f"{side}_lon", f"{side}_row", f"{side}_col", f"{side}_gray"]
        fields += [f"{side}_{n}" for n in feat_names]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for m in matches:
            row = [m.rank, round(m.similarity, 3), round(m.distance, 5), round(m.elevation_difference_m, 1)]
            for p in (m.a, m.b):
                row += [round(p.lat, 4), round(p.lon, 4), p.row, p.col, p.gray]
                row += [round(p.features[n], 3) for n in feat_names]
            writer.writerow(row)


def write_json(matcher: TerrainMatcher, matches: list[Match], checks: list[dict], path: Path) -> None:
    payload = {
        "mode": matcher.config.mode,
        "features": matcher.active,
        "weights": matcher.weights,
        "scales_km": list(matcher.config.scales_km),
        "min_separation_deg": matcher.config.min_separation_deg,
        "calibration": checks,
        "matches": [m.to_dict() for m in matches],
    }
    path.write_text(json.dumps(payload, indent=2))


def plot_matches(matcher: TerrainMatcher, matches: list[Match], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    sides = (matcher.a.hm, matcher.b.hm)
    cmaps = ("gist_earth", "cividis")
    fig, axes = plt.subplots(2, 1, figsize=(14, 13), constrained_layout=True)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(matches), 1)))

    for ax, hm, cmap in zip(axes, sides, cmaps):
        extent = (*hm.lon_range, *hm.lat_range)
        im = ax.imshow(hm.elevation, cmap=cmap, extent=extent, aspect="auto", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
        cb.set_label("calibrated elevation (m)")
        for lm, marker in ((hm.body.highest, "^"), (hm.body.lowest, "v")):
            ax.scatter(lm.lon, lm.lat, marker=marker, s=120, c="red", edgecolors="white", zorder=5)
            ax.annotate(lm.name, (lm.lon, lm.lat), xytext=(6, 6), textcoords="offset points", color="red", fontsize=8)
        ax.set_title(
            f"{hm.body.name}: lightest = {hm.body.max_elevation_m:,.0f} m, darkest = {hm.body.min_elevation_m:,.0f} m"
        )
        ax.set_xlabel("longitude (deg)")
        ax.set_ylabel("latitude (deg)")

    for m, color in zip(matches, colors):
        for ax, p in ((axes[0], m.a), (axes[1], m.b)):
            ax.scatter(p.lon, p.lat, s=70, color=color, edgecolors="black", zorder=6)
            ax.annotate(str(m.rank), (p.lon, p.lat), xytext=(4, -10), textcoords="offset points", fontsize=8, weight="bold")
        con = ConnectionPatch(
            xyA=(m.a.lon, m.a.lat), coordsA=axes[0].transData,
            xyB=(m.b.lon, m.b.lat), coordsB=axes[1].transData,
            color=color, alpha=0.45, lw=1,
        )
        fig.add_artist(con)

    fig.suptitle(f"Most similar {sides[0].body.name}/{sides[1].body.name} terrain ({matcher.config.mode} mode)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_histograms(matcher: TerrainMatcher, matches: list[Match], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bins = np.linspace(-11500, 11500, 116)
    for side, color in ((matcher.a, "tab:blue"), (matcher.b, "tab:gray")):
        elev = side.raw[:, 0]
        ax.hist(elev, bins=bins, alpha=0.55, color=color, density=True, label=f"{side.hm.body.name} (area-weighted)")
    if matches:
        ax.scatter([m.a.elevation_m for m in matches], np.zeros(len(matches)), marker="|", s=400, c="tab:red",
                   label="matched elevations")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("elevation (m) relative to datum (Earth: sea level, Moon: mean radius)")
    ax.set_ylabel("fraction of surface")
    ax.legend()
    ax.set_title("Calibrated elevation distributions")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(matcher: TerrainMatcher, matches: list[Match], out_dir: Path, plots: bool = True) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = [calibration_check(matcher.a.hm), calibration_check(matcher.b.hm)]
    paths = {"csv": out_dir / "matches.csv", "json": out_dir / "matches.json", "summary": out_dir / "summary.txt"}
    write_csv(matches, paths["csv"])
    write_json(matcher, matches, checks, paths["json"])
    summary = "\n\n".join(
        ["Calibration", *(format_calibration(c) for c in checks), "Top matches", format_matches(matches)]
    )
    paths["summary"].write_text(summary + "\n")
    if plots:
        paths["map"] = out_dir / "matches_map.png"
        paths["histogram"] = out_dir / "elevation_histogram.png"
        plot_matches(matcher, matches, paths["map"])
        plot_histograms(matcher, matches, paths["histogram"])
    return paths
