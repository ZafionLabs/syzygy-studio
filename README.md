# terrain-bot: Earth vs Moon look-alike terrain

`terrain-bot` turns grayscale Earth and Moon heightmaps into **real elevations** and
finds the places on each body whose terrain is most alike.

## Calibration to real data

Each map's brightness is mapped linearly onto that body's true elevation range:

| Body  | Lightest pixel becomes                         | Darkest pixel becomes                               | Datum                       |
|-------|------------------------------------------------|-----------------------------------------------------|-----------------------------|
| Earth | Mount Everest, **+8,848.86 m** (2020 survey)   | Challenger Deep, Mariana Trench, **-10,935 m** (2021) | mean sea level              |
| Moon  | Selenean summit, **+10,786 m** (LRO LOLA)      | Point Trieste, Antoniadi crater, **-9,129 m** (LOLA) | mean lunar radius 1737.4 km |

With the bundled 8-bit maps that is ~91 m per gray level on Earth and ~82 m on the Moon.
`terrain-bot calibrate` also checks the map against the real landmark locations, e.g. the
Moon map's lightest pixel lands 13 km from the real Selenean summit.

## How similarity is measured

For every pixel the bot computes, in real units:

* `elevation_m`: calibrated height
* `mean_<s>km_m` / `relief_<s>km_m`: mean and standard deviation of elevation in a
  Gaussian window of *s* ground kilometres (default 75 km and 250 km)
* `slope_75km_deg`: regional slope

Windows are defined in kilometres (not pixels) using each body's radius and widened by
1/cos(latitude), so Earth (~48 km/pixel here) and Moon (~12 km/pixel) are compared at the same
physical scale. Features are z-scored over both bodies, weighted (elevation x3, slope x0.5,
others x1), and matched with a KD-tree. Reported matches are spread out with a minimum arc
separation (default 5 degrees) on both bodies. `similarity = 100 * exp(-weighted RMS z-distance)`.

Use `--mode elevation` to compare height alone.

## Usage

```bash
pip install -r requirements.txt

# Globally most similar Earth/Moon pairs -> results/ (CSV, JSON, summary, map, histogram)
python -m terrain_bot analyze --top 20

# Places on the Moon most like a point on Earth (or vice versa)
python -m terrain_bot query --body earth --lat 36.1 --lon -112.1
python -m terrain_bot query --body moon --lat -70.36 --lon -172.49

# Calibration and landmark sanity check
python -m terrain_bot calibrate

# Interactive bot: "everest", "earth 27.99 86.93", "height moon 5.4 -158.6", "top 10"
python -m terrain_bot chat

# Browser UI at http://localhost:8000: hover to read elevations, click a map to find look-alikes
python -m terrain_bot serve
```

To share the browser UI temporarily, run `cloudflared tunnel --url http://localhost:8000`, which prints a
public `https://<random>.trycloudflare.com` link.

Use your own images with `--earth path.png --moon path.png`. Maps are assumed to be
equirectangular, covering -180 to 180 longitude and -90 to 90 latitude. Override with
`--earth-lat-range S N`, `--earth-lon-range W E` (same for `--moon-*`). Transparent pixels are
ignored. 16-bit PNG/TIFF heightmaps are read at full precision. `--clip-percentiles 0.01 99.99`
stops a few noisy pixels from setting the extremes, and `--gamma` handles non-linear encodings.

## Caveats

* The bundled Earth image is a shaded-relief rendering, not a pure DEM, and its projection is only
  roughly equirectangular. Its lightest pixel is in Tibet (~800 km from Everest) and trenches are
  not the darkest pixels, so treat Earth elevations as approximate. For precise results, use real
  DEMs exported to grayscale (e.g. NOAA ETOPO 2022 for Earth, LRO LOLA LDEM for the Moon).
* Linear calibration assumes brightness is proportional to height.

## Tests

```bash
python -m pytest
```
