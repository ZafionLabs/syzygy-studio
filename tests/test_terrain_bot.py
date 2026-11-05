from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from terrain_bot import EARTH, MOON, MatchConfig, TerrainMatcher, heightmap_from_array, load_heightmap
from terrain_bot.bodies import Body
from terrain_bot.cli import handle_chat, main
from terrain_bot.features import compute_features

DATA = Path(__file__).resolve().parent.parent / "data"


def _synthetic_terrain(h=90, w=180, seed=0):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    z = gaussian_filter(rng.normal(size=(h, w)), sigma=3, mode="wrap")
    return (z - z.min()) / (z.max() - z.min()) * 255.0


def test_calibration_maps_extremes_to_real_elevations():
    gray = np.array([[0, 64, 128], [192, 255, 100]], dtype=float)
    earth = heightmap_from_array(gray, EARTH)
    moon = heightmap_from_array(gray, MOON)
    assert earth.elevation.max() == pytest.approx(8848.86)
    assert earth.elevation.min() == pytest.approx(-10935.0)
    assert moon.elevation.max() == pytest.approx(10786.0)
    assert moon.elevation.min() == pytest.approx(-9129.0)
    assert earth.elevation[0, 2] == pytest.approx(-10935.0 + 128 / 255 * EARTH.relief_m)


def test_transparent_pixels_are_ignored(tmp_path):
    rgba = np.zeros((4, 8, 4), dtype=np.uint8)
    rgba[..., :3] = 100
    rgba[1, 1, :3] = 200
    rgba[2, 2, :3] = 50
    rgba[..., 3] = 255
    rgba[0, :, :3] = 255
    rgba[0, :, 3] = 0
    path = tmp_path / "map.png"
    Image.fromarray(rgba, "RGBA").save(path)

    hm = load_heightmap(path, EARTH)
    assert not hm.valid[0].any()
    assert np.isnan(hm.elevation[0]).all()
    assert hm.elevation[1, 1] == pytest.approx(EARTH.max_elevation_m)
    assert hm.elevation[2, 2] == pytest.approx(EARTH.min_elevation_m)


def test_georeferencing_round_trip():
    hm = heightmap_from_array(_synthetic_terrain(), EARTH)
    for lat, lon in [(0.5, 0.5), (45.0, -120.0), (-89.0, 179.0), (27.9881, 86.925)]:
        r, c = hm.pixel_of(lat, lon)
        plat, plon = hm.latlon_of(r, c)
        assert abs(plat - lat) <= 1.0 and abs(plon - lon) <= 1.0
    assert hm.pixel_of(0.0, 190.0) == hm.pixel_of(0.0, -170.0)
    with pytest.raises(ValueError):
        hm.pixel_of(95.0, 0.0)


def test_features_use_physical_scale():
    flat = np.full((90, 180), 128.0)
    flat[0, 0] = 0
    flat[-1, -1] = 255
    fs = compute_features(heightmap_from_array(flat, EARTH))
    mid = fs.at(45, 90)
    assert mid["relief_75km_m"] == pytest.approx(0.0, abs=1e-6)
    assert mid["slope_75km_deg"] == pytest.approx(0.0, abs=1e-6)


def test_matcher_finds_planted_twin():
    twin_body = Body("Twin", EARTH.mean_radius_km, "test", EARTH.highest, EARTH.lowest)
    a_gray = _synthetic_terrain(seed=1)
    b_gray = _synthetic_terrain(seed=2)
    b_gray[30:60, 100:160] = a_gray[20:50, 20:80]
    for g in (a_gray, b_gray):
        g[0, 0], g[-1, -1] = 0.0, 255.0

    hm_a = heightmap_from_array(a_gray, EARTH)
    hm_b = heightmap_from_array(b_gray, twin_body)
    matcher = TerrainMatcher(hm_a, hm_b, MatchConfig(scales_km=(150.0, 400.0), min_separation_deg=10))

    lat, lon = hm_a.latlon_of(35, 50)
    origin, results = matcher.query("earth", lat, lon, k=3)
    best = results[0].b
    assert abs(best.row - 45) <= 1 and abs(best.col - 130) <= 1
    assert results[0].similarity > 95

    top = matcher.top_matches(5)
    assert len(top) == 5
    assert all(top[i].distance <= top[i + 1].distance for i in range(4))


def test_top_matches_respect_separation():
    hm_a = heightmap_from_array(_synthetic_terrain(seed=3), EARTH)
    hm_b = heightmap_from_array(_synthetic_terrain(seed=4), MOON)
    matcher = TerrainMatcher(hm_a, hm_b, MatchConfig(min_separation_deg=15))
    top = matcher.top_matches(8)
    from terrain_bot.heightmap import haversine_km

    for side in ("a", "b"):
        pts = [(getattr(m, side).lat, getattr(m, side).lon) for m in top]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                assert haversine_km(*pts[i], *pts[j], 180 / np.pi) >= 15 - 1e-6


@pytest.mark.skipif(not (DATA / "earth_heightmap.png").exists(), reason="bundled data missing")
def test_bundled_maps_end_to_end(tmp_path):
    assert main(["analyze", "--top", "5", "--out", str(tmp_path), "--no-plots"]) == 0
    assert (tmp_path / "matches.csv").read_text().count("\n") == 6

    earth = load_heightmap(DATA / "earth_heightmap.png", EARTH)
    moon = load_heightmap(DATA / "moon_heightmap.png", MOON)
    matcher = TerrainMatcher(earth, moon)
    reply = handle_chat(matcher, "everest", k=2)
    assert "Mount Everest" in reply and "Moon location" in reply
    assert "outside the map" in handle_chat(matcher, "moon 100 0")
