"""Human-readable, step-by-step account of how a match was found."""

from __future__ import annotations

import numpy as np

from .heightmap import Heightmap
from .matcher import Match, SitePoint, TerrainMatcher

FEATURE_LABELS = {
    "elevation_m": "Elevation at the point",
    "mean": "Average elevation within {s} km",
    "relief": "Roughness (std-dev of elevation) within {s} km",
    "slope": "Regional slope over {s} km",
}


def feature_label(name: str) -> str:
    if name == "elevation_m":
        return FEATURE_LABELS[name]
    kind, scale = name.split("_")[0], name.split("_")[1].removesuffix("km")
    return FEATURE_LABELS[kind].format(s=scale)


def feature_unit(name: str) -> str:
    return "deg" if name.endswith("_deg") else "m"


def describe_surface(body: str, elevation_m: float) -> str:
    if body == "Earth":
        if elevation_m < 0:
            return f"underwater: sea floor {abs(elevation_m):,.0f} m below sea level"
        return f"on land: {elevation_m:,.0f} m above sea level"
    side = "above" if elevation_m >= 0 else "below"
    if body == "Mars":
        return f"{abs(elevation_m):,.0f} m {side} the Mars datum (MOLA areoid; Mars has no sea level today)"
    if body == "Moon":
        return f"{abs(elevation_m):,.0f} m {side} the mean lunar radius (the Moon has no sea level)"
    return f"{abs(elevation_m):,.0f} m {side} the reference surface"


def _value_step(hm: Heightmap, p: SitePoint) -> dict:
    if hm.source == "dem":
        return {
            "title": "Read the measured elevation",
            "lines": [
                f"Data set: {hm.source_name}.",
                f"Value at that cell: {p.elevation_m:,.1f} m. No conversion is needed because the data is already in metres.",
            ],
        }
    b = hm.body
    if hm.source == "image-fit":
        levels, metres = hm.lut
        i = int(np.argmin(np.abs(levels - p.gray)))
        return {
            "title": "Convert brightness to real elevation (curve fitted to measured data)",
            "lines": [
                f"Pixel brightness: {p.gray:.0f}.",
                f"For every brightness level the bot found all pixels with that brightness and took the median measured "
                f"elevation at the same coordinates in {hm.source_name.split(' fitted to ')[-1]}. The curve was then forced "
                "to rise with brightness.",
                f"Brightness {levels[i]:.0f} -> {metres[i]:,.0f} m, so this point is {p.elevation_m:,.0f} m.",
                f"The curve runs from {metres[0]:,.0f} m (brightness {levels[0]:.0f}) to {metres[-1]:,.0f} m "
                f"(brightness {levels[-1]:.0f}). It corrects for clipped or non-linear images, which a straight "
                f"{b.lowest.name}-to-{b.highest.name} scale can't.",
                *(
                    [
                        f"Warning: brightness {p.gray:.0f} is the image's "
                        f"{'brightest' if p.gray >= levels[-1] else 'darkest'} level, so the image is clipped here. The real "
                        f"terrain may be much {'higher' if p.gray >= levels[-1] else 'lower'}; use the measured data for this spot."
                    ]
                    if p.gray >= levels[-1] or p.gray <= levels[0]
                    else []
                ),
            ],
        }
    t = (p.gray - hm.gray_min) / (hm.gray_max - hm.gray_min)
    return {
        "title": "Convert brightness to real elevation",
        "lines": [
            f"Pixel brightness: {p.gray:.0f} (the map's darkest pixel is {hm.gray_min:.0f}, its lightest is {hm.gray_max:.0f}).",
            f"The darkest pixel is set to {b.lowest.name} ({b.min_elevation_m:,.0f} m) and the lightest to "
            f"{b.highest.name} ({b.max_elevation_m:,.0f} m).",
            f"elevation = {b.min_elevation_m:,.0f} + ({p.gray:.0f} - {hm.gray_min:.0f}) / ({hm.gray_max:.0f} - {hm.gray_min:.0f})"
            f" x {b.relief_m:,.0f} = {b.min_elevation_m:,.0f} + {t:.4f} x {b.relief_m:,.0f} = {p.elevation_m:,.0f} m",
            f"Each brightness step is worth {b.relief_m / (hm.gray_max - hm.gray_min):,.1f} m, "
            "so elevations are only accurate to about half of that.",
        ],
    }


def breakdown_lines(m: Match) -> list[str]:
    lines = []
    for row in m.breakdown:
        u = feature_unit(row["feature"])
        dp = 2 if u == "deg" else 1
        lines.append(
            f"{feature_label(row['feature'])}: {row['a']:,.{dp}f} vs {row['b']:,.{dp}f} {u} "
            f"(difference {row['difference']:+,.{dp}f} {u} = {row['z_difference']:+.3f} std-devs, "
            f"weight {row['weight_share']:.0%}, adds {row['contribution']:.5f})"
        )
    total = sum(r["contribution"] for r in m.breakdown)
    lines.append(
        f"distance = sqrt({total:.5f}) = {np.sqrt(total):.4f}; similarity = 100 x exp(-{m.distance:.4f}) = {m.similarity:.1f}%"
    )
    return lines


def explain_query(
    matcher: TerrainMatcher,
    body: str,
    origin: SitePoint,
    matches: list[Match],
    surface: str = "any",
    target: str | None = None,
) -> list[dict]:
    src, dst = matcher._pair(body, target)
    hm = src.hm
    km_ns, km_ew = hm.km_per_px
    steps = [
        {
            "title": "Locate the point on the map",
            "lines": [
                f"{origin.body} {origin.lat:.3f}, {origin.lon:.3f} falls in pixel row {origin.row}, column {origin.col} "
                f"of a {hm.shape[1]} x {hm.shape[0]} map.",
                f"One pixel covers about {km_ns:,.1f} km north-south and {km_ew * np.cos(np.radians(origin.lat)):,.1f} km "
                f"east-west at this latitude, so locations can't be resolved more finely than that.",
            ],
        },
        _value_step(hm, origin),
        {"title": "Above or below the surface datum", "lines": [describe_surface(origin.body, origin.elevation_m)]},
        {
            "title": "Describe the surrounding terrain",
            "lines": [
                "Height alone matches thousands of places, so the shape of the surroundings is measured too. The windows are "
                "Gaussian, measured in real ground kilometres on each body, and widened east-west by 1/cos(latitude):",
                *[
                    f"{feature_label(n)}: {v:,.2f} {feature_unit(n)}"
                    for n, v in origin.features.items()
                ],
            ],
        },
        {
            "title": "Put every feature on a common scale",
            "lines": [
                "Each feature is converted to standard deviations using the area-weighted spread over both bodies, then weighted:",
                *[
                    f"{feature_label(n)}: 1 std-dev = {matcher.std[i]:,.{2 if feature_unit(n) == 'deg' else 1}f} {feature_unit(n)}, weight {matcher.weight_share[i]:.1%}"
                    for i, n in enumerate(matcher.feature_names)
                ],
                "In elevation-only mode, height carries almost all of the weight and the other features only break ties."
                if matcher.config.mode == "elevation" else "",
            ],
        },
        {
            "title": f"Search every pixel on {dst.hm.body.name}",
            "lines": [
                f"Compared against all {matcher.search_size(dst.hm.body.name, surface):,} valid {dst.hm.body.name} pixels"
                + (f" ({surface} only)" if surface != "any" and dst.hm.body.name == "Earth" else "")
                + f" from {dst.hm.source_name}.",
                "An exact KD-tree nearest-neighbour search gives the same best match as checking every pixel one by one "
                "(this is verified in the tests).",
                "distance = sqrt(sum over features of weight x (difference / std-dev)^2)",
            ],
        },
    ]
    if matches:
        best = matches[0]
        steps.append(
            {
                "title": f"Best match: {best.b.body} {best.b.lat:.2f}, {best.b.lon:.2f}",
                "lines": [describe_surface(best.b.body, best.b.elevation_m), *breakdown_lines(best)],
            }
        )
        steps.append(
            {
                "title": "Keep the other results spread out",
                "lines": [
                    f"Results are taken in order of distance, skipping any within {matcher.config.min_separation_deg:g} deg of "
                    "a result already kept, so the list isn't the same crater repeated.",
                ],
            }
        )
    for s in steps:
        s["lines"] = [line for line in s["lines"] if line]
    return steps


def explain_top(matcher: TerrainMatcher, matches: list[Match], surface: str = "any") -> list[dict]:
    if matches:
        a, b = matcher.side(matches[0].a.body).hm, matcher.side(matches[0].b.body).hm
    else:
        a, b = matcher.a.hm, matcher.b.hm
    steps = [
        {
            "title": "Load real-world elevations",
            "lines": [f"{hm.body.name}: {hm.source_name}, {hm.shape[1]} x {hm.shape[0]} pixels" for hm in (a, b)],
        },
        {
            "title": "Describe every pixel",
            "lines": [
                "Each pixel gets its elevation, plus the average and roughness of the terrain within "
                + " and ".join(f"{s:g} km" for s in matcher.config.scales_km)
                + ", plus regional slope. All of these are measured in ground kilometres so both bodies are compared at the same scale."
            ],
        },
        {
            "title": "Compare every pixel with every pixel",
            "lines": [
                f"{matcher.search_size(a.body.name, surface):,} {a.body.name} pixels x {matcher.search_size(b.body.name, surface):,} "
                f"{b.body.name} pixels, using an exact KD-tree. Earth pixels: {surface}.",
                "distance = sqrt(sum over features of weight x (difference / std-dev)^2); similarity = 100 x exp(-distance)",
            ],
        },
        {
            "title": "Pick the closest pairs and spread them out",
            "lines": [
                f"Pairs are taken in order of distance, skipping any within {matcher.config.min_separation_deg:g} deg of an "
                "already chosen point on either body."
            ],
        },
    ]
    if matches:
        steps.append({"title": "Why #1 matched", "lines": breakdown_lines(matches[0])})
    return steps


def format_steps(steps: list[dict]) -> str:
    out = []
    for i, s in enumerate(steps, 1):
        out.append(f"Step {i}. {s['title']}")
        out.extend(f"   {line}" for line in s["lines"])
    return "\n".join(out)
