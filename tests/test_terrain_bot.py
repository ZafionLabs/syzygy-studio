from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from terrain_bot import EARTH, MARS, MOON, MatchConfig, TerrainMatcher, heightmap_from_array, load_heightmap
from terrain_bot.bodies import Body
from terrain_bot import dem
from terrain_bot.cli import handle_chat, main
from terrain_bot.explain import describe_surface, explain_query, format_steps
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
    matcher = TerrainMatcher([hm_a, hm_b], MatchConfig(scales_km=(150.0, 400.0), min_separation_deg=10))

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
    matcher = TerrainMatcher([hm_a, hm_b], MatchConfig(min_separation_deg=15))
    top = matcher.top_matches(8)
    from terrain_bot.heightmap import haversine_km

    for side in ("a", "b"):
        pts = [(getattr(m, side).lat, getattr(m, side).lon) for m in top]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                assert haversine_km(*pts[i], *pts[j], 180 / np.pi) >= 15 - 1e-6


def test_kdtree_search_equals_brute_force():
    hm_a = heightmap_from_array(_synthetic_terrain(seed=7), EARTH)
    hm_b = heightmap_from_array(_synthetic_terrain(seed=8), MOON)
    for mode in ("terrain", "elevation"):
        matcher = TerrainMatcher([hm_a, hm_b], MatchConfig(mode=mode))
        rng = np.random.default_rng(0)
        for lat, lon in zip(rng.uniform(-80, 80, 15), rng.uniform(-180, 180, 15)):
            for surface in ("any", "land", "underwater"):
                _, results = matcher.query("earth", lat, lon, k=1, surface=surface)
                idx, d = matcher.brute_force_best("earth", lat, lon, surface=surface)
                assert results[0].distance == pytest.approx(d, abs=1e-12)
                assert (results[0].b.row, results[0].b.col) == (matcher.b.rows[idx], matcher.b.cols[idx])
                total = sum(r["contribution"] for r in results[0].breakdown)
                assert np.sqrt(total) == pytest.approx(results[0].distance, rel=1e-9)


def test_top_matches_bounded_search_is_exact():
    hm_a = heightmap_from_array(_synthetic_terrain(seed=9), EARTH)
    hm_b = heightmap_from_array(_synthetic_terrain(seed=10), MOON)
    matcher = TerrainMatcher([hm_a, hm_b])
    a_idx, _ = matcher.a.subset("any")
    b_idx, b_tree = matcher.b.subset("any")
    bounded = matcher.top_matches(6)
    unbounded = matcher._top_within(matcher.a, matcher.b, a_idx, b_idx, b_tree, 6, np.inf)
    assert [(m.a.row, m.a.col, m.b.row, m.b.col) for m in bounded] == [
        (m.a.row, m.a.col, m.b.row, m.b.col) for m in unbounded
    ]


def test_surface_filter_selects_land_or_sea_floor():
    hm_a = heightmap_from_array(_synthetic_terrain(seed=11), EARTH)
    hm_b = heightmap_from_array(_synthetic_terrain(seed=12), MOON)
    matcher = TerrainMatcher([hm_a, hm_b])
    _, wet = matcher.query("moon", 0, 0, k=5, surface="underwater")
    _, dry = matcher.query("moon", 0, 0, k=5, surface="land")
    assert all(m.b.elevation_m < 0 for m in wet)
    assert all(m.b.elevation_m >= 0 for m in dry)
    assert all(m.a.elevation_m < 0 for m in matcher.top_matches(4, surface="underwater"))


def test_explanation_steps_show_the_working():
    hm_a = heightmap_from_array(_synthetic_terrain(seed=13), EARTH)
    hm_b = heightmap_from_array(_synthetic_terrain(seed=14), MOON)
    matcher = TerrainMatcher([hm_a, hm_b])
    origin, results = matcher.query("earth", 10, 20, k=3)
    steps = explain_query(matcher, "earth", origin, results)
    text = format_steps(steps)
    assert "Convert brightness to real elevation" in text
    assert f"{origin.gray:.0f}" in text and "Challenger Deep" in text
    assert "Search every pixel" in text and "similarity = 100 x exp" in text
    assert describe_surface("Earth", -4100) == "underwater: sea floor 4,100 m below sea level"
    assert "no sea level" in describe_surface("Moon", -100)


def test_sidecar_sets_image_bounds(tmp_path):
    path = tmp_path / "map.png"
    Image.fromarray((_synthetic_terrain(20, 40)).astype(np.uint8), "L").save(path)
    (tmp_path / "map.json").write_text('{"lat_range": [-60, 80]}')
    hm = load_heightmap(path, EARTH)
    assert hm.lat_range == (-60, 80)
    assert load_heightmap(path, EARTH, lat_range=(-90, 90)).lat_range == (-90, 90)


@pytest.mark.skipif(not (DATA / "earth_heightmap.png").exists(), reason="bundled data missing")
def test_bundled_maps_end_to_end(tmp_path):
    assert main(["analyze", "--source", "images", "--top", "5", "--out", str(tmp_path), "--no-plots"]) == 0
    assert (tmp_path / "matches.csv").read_text().count("\n") == 6

    earth = load_heightmap(DATA / "earth_heightmap.png", EARTH)
    moon = load_heightmap(DATA / "moon_heightmap.png", MOON)
    matcher = TerrainMatcher([earth, moon])
    reply = handle_chat(matcher, "everest steps", k=2)
    assert "Mount Everest" in reply and "Moon location" in reply and "Step 1." in reply
    assert "outside the map" in handle_chat(matcher, "moon 100 0")
    assert "sea" in handle_chat(matcher, "moon 0 0 underwater", k=2)


@pytest.mark.skipif(not dem.dem_available("earth") or not dem.dem_available("moon"), reason="DEMs not downloaded")
def test_real_dem_data():
    earth, moon = dem.load_dem("earth"), dem.load_dem("moon")
    assert earth.elevation_at(31.5, 35.5) == pytest.approx(-430, abs=100)  # Dead Sea shore
    assert earth.elevation_at(11.37, 142.59) < -8000  # Mariana Trench
    assert earth.elevation_at(0, -150) < -3000  # Pacific abyssal plain
    assert moon.elevation_at(5.41, -158.63) > 10000  # Selenean summit
    assert moon.elevation_at(-70.36, -172.49) < -8000  # Antoniadi crater

    moon_cmp = dem.compare_to_dem(load_heightmap(DATA / "moon_heightmap.png", MOON), moon)
    assert moon_cmp["spearman"] > 0.95
    earth_cmp = dem.compare_to_dem(load_heightmap(DATA / "earth_heightmap.png", EARTH), earth)
    assert earth_cmp["spearman"] > 0.95
    assert earth_cmp["land_water_agreement"] > 0.9


@pytest.mark.skipif(not dem.dem_available("mars"), reason="Mars DEM not downloaded")
def test_mars_real_data_and_fitted_image():
    mars = dem.load_dem("mars")
    assert mars.elevation_at(18.65, -133.8) > 19000  # Olympus Mons
    assert mars.elevation_at(-32.85, 62.21) < -7500  # Badwater crater, Hellas
    assert mars.latlon_of(*mars.extreme_pixels()["highest"])[1] == pytest.approx(-133.1, abs=0.5)

    image = load_heightmap(DATA / "mars_heightmap.png", MARS)
    assert image.elevation.max() == pytest.approx(21229.0)
    assert image.elevation.min() == pytest.approx(-8201.0)
    linear = dem.compare_to_dem(image, mars)
    fitted = dem.compare_to_dem(dem.fit_image_to_dem(image, mars), mars)
    assert linear["spearman"] > 0.95
    assert fitted["median_abs_error_m"] < 300 < linear["median_abs_error_m"]


def test_three_bodies_any_pair():
    maps = [heightmap_from_array(_synthetic_terrain(seed=s), b) for s, b in ((20, EARTH), (21, MOON), (22, MARS))]
    matcher = TerrainMatcher(maps)
    assert matcher.default_target("earth") == "moon"
    assert matcher.default_target("mars") == "earth"
    _, res = matcher.query("mars", 10, 20, k=2, target="moon")
    assert {m.b.body for m in res} == {"Moon"}
    top = matcher.top_matches(3, a="moon", b="mars")
    assert top[0].a.body == "Moon" and top[0].b.body == "Mars"
    idx, d = matcher.brute_force_best("mars", 10, 20, target="moon")
    assert res[0].distance == pytest.approx(d)
    with pytest.raises(ValueError):
        matcher.query("mars", 0, 0, target="mars")
    reply = handle_chat(matcher, "olympus on moon steps", k=1)
    assert "Olympus Mons" in reply and "Moon location" in reply and "Mars datum" in reply


def test_web_api_round_trip():
    import json
    import threading
    from urllib.error import HTTPError
    from urllib.request import urlopen

    from terrain_bot.web import TerrainApp, make_handler
    from http.server import ThreadingHTTPServer

    hm_a = heightmap_from_array(_synthetic_terrain(seed=5), EARTH)
    hm_b = heightmap_from_array(_synthetic_terrain(seed=6), MOON)
    app = TerrainApp({"images": {"earth": hm_a, "moon": hm_b}}, MatchConfig(), default_source="images")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urlopen(base + "/").read()
        assert b"Terrain Bot" in page
        assert b"Find closest match" in page and b"Compare two bodies" in page
        assert b"<select id=\"searchMode\">" not in page
        assert urlopen(base + "/map/images/moon.png").read().startswith(b"\x89PNG")
        layers = json.load(urlopen(base + "/api/layers"))
        assert [item["label"] for item in layers["layers"]] == [
            "Elevation", "Temperature", "Ice and water", "Geology", "Weather", "Light", "Radiation",
        ]
        assert layers["bodies"]["moon"]["weather"]["available"] is True
        assert layers["bodies"]["moon"]["weather"]["citation"]
        assert layers["bodies"]["mars"]["radiation"]["available"] is True
        assert urlopen(base + "/map/layer/earth/temperature.png").read().startswith(b"\x89PNG")
        assert urlopen(base + "/map/layer/mars/weather.png").read().startswith(b"\x89PNG")
        info = json.load(urlopen(base + "/api/info"))
        assert info["sources"]["images"]["bodies"]["earth"]["color_max_m"] == pytest.approx(8848.86)
        res = json.load(urlopen(base + "/api/query?body=moon&lat=10&lon=20&k=3&mode=elevation&surface=underwater"))
        assert len(res["matches"]) == 3 and res["matches"][0]["b"]["body"] == "Earth"
        assert all(m["b"]["underwater"] for m in res["matches"])
        assert res["steps"][0]["title"] == "Locate the point on the map"
        assert len(res["matches"][0]["breakdown"]) == 6
        top = json.load(urlopen(base + "/api/top?n=4"))
        assert len(top["matches"]) == 4 and top["steps"]
        with pytest.raises(HTTPError) as err:
            urlopen(base + "/api/height?body=earth&lat=120&lon=0")
        assert err.value.code == 400
    finally:
        server.shutdown()


def test_place_names_cover_earth_ocean_moon_and_mars():
    import json
    names = json.loads((Path(__file__).resolve().parents[1] / "frontend" / "public" / "names.json").read_text())
    assert len(names["earth"]) > 5000
    assert len(names["ocean"]) > 4000
    assert len(names["moon"]) > 8000
    assert len(names["mars"]) > 1500
    found = {body: {row[0] for row in names[body]} for body in ("earth", "ocean", "moon", "mars")}
    assert "Tokyo" in found["earth"] and "Mount Everest" in found["earth"]
    assert "Mariana" in found["ocean"]
    assert "Tycho" in found["moon"] and "Mare Imbrium" in found["moon"]
    assert "Olympus Mons" in found["mars"] and "Hellas Planitia" in found["mars"]
    assert "Natural Earth" in names["citation"]["earth"]
    assert "GEBCO" in names["citation"]["ocean"]
    assert "International Astronomical Union" in names["citation"]["moon"]
