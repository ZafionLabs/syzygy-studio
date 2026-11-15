# terrain-bot: Earth, Moon and Mars look-alike terrain

`terrain-bot` finds the places on Earth, the Moon and Mars (any pair) whose terrain is most alike,
using **real elevations**. Earth's underwater sea floor is included, and every answer comes with the
step-by-step working.

## Data

| Source (`--source`) | Earth | Moon | Mars |
|---|---|---|---|
| `dem` (default): real measured data in `data/dem/` | NOAA **ETOPO1** (land + ocean bathymetry), sampled every 0.25 deg | NASA **LRO LOLA LDEM_4**, 4 px/deg, relative to the 1737.4 km mean radius | NASA **MGS MOLA MEGDR**, 4 px/deg, relative to the MOLA areoid |
| `images`: your grayscale maps in `data/`, lightest = highest, darkest = lowest | Everest **+8,848.86 m** / Challenger Deep **-10,935 m** | Selenean summit **+10,786 m** / Point Trieste (Antoniadi) **-9,129 m** | Olympus Mons **+21,229 m** / Badwater crater (Hellas) **-8,201 m** |
| `images-fit`: the same images through a brightness curve learned from the DEM | | | |

`images-fit` takes, for every brightness level, the median measured elevation of the pixels with
that brightness, then forces the curve to rise with brightness. Run
`python -m terrain_bot download-dem` to refetch the DEMs (about 5 MB total).

### How accurate are the images? (`python -m terrain_bot validate`)

Each calibrated image is compared pixel by pixel with the real data at the same coordinates:

| Image | Rank correlation | Median error, lowest-to-highest scale | Median error, fitted curve | Notes |
|---|---|---|---|---|
| Moon | **0.989** | 400 m | **125 m** | faithful elevation map |
| Mars | **0.980** | 5,533 m | **120 m** | clipped: brightness 255 covers everything above about +7 km and 0 covers everything below about -6.4 km |
| Earth | **0.995** | 1,202 m | **54 m** | faithful elevation + bathymetry map; land/water agrees on 94% (98% after fit) |

The Moon image is a faithful elevation map: its lightest pixel is 13 km from the real Selenean
summit. The Mars image has the right geography but saturates at both ends, so Olympus Mons is as
white as the Tharsis plains. The straight Olympus-to-Hellas scale therefore reads about 5 km too
high, but the fitted curve fixes that except in the clipped areas, which the steps flag. The Earth
image is a full-globe grayscale heightmap (lightest = highest, darkest = deepest ocean), so the
linear Everest-to-Challenger scale already tracks ETOPO1 closely; the fitted curve brings median
error down to about 54 m. Measured DEMs remain the default for precision at individual peaks and
trench bottoms.

## How a match is found (shown by `query`, the chat bot's `steps`, and the web UI)

1. **Locate** the coordinate's pixel. Resolution limits precision: about 28 km per pixel for Earth
   and 7.6 km for the Moon with the DEMs.
2. **Read the elevation.** DEM values are already in metres. For images,
   `elevation = min + (gray - darkest) / (lightest - darkest) x (max - min)`.
3. **Classify** the point: Earth land or underwater (sea level = 0 m). The Moon (mean radius) and
   Mars (areoid) have no sea level.
4. **Describe the surroundings** in real ground kilometres: average and roughness (standard
   deviation) of elevation within 75 km and 250 km, plus regional slope. The windows are widened by
   1/cos(latitude).
5. **Normalise** each feature by its area-weighted standard deviation over both bodies and weight
   it: elevation 40%, slope 7%, the others 13% each. In `--mode elevation`, height carries all of
   the weight and the other features only break ties.
6. **Search every pixel** of the other body (about 1 million) with an exact KD-tree. The tests
   check that it returns the same best match as brute force.
7. **Score**: `distance = sqrt(sum weight x (difference / std)^2)` and
   `similarity = 100 x exp(-distance)`. The per-feature contributions are listed.
8. **Spread out** the results so no two are within 5 deg of each other.

`--surface land|underwater` (and the web UI's "Earth points" selector) restricts the Earth side to
dry land or sea floor.

## Other maps

In Find closest match, a Moon / Mars toggle shows one of those maps at a time, with Earth beside it. The web app can turn on elevation, temperature, ice and water, geology, weather, light, and radiation together. Each selected map is drawn on its own, one under the other. Hover a selected point or a found match to see its name, coordinates, a picture of the spot, the nearest landmark and its distance, and the distance from the center of the nearest crater. Coordinates can be typed instead of clicking a map. Under the maps, the match uses the earlier place cards: both places, their pictures, the overall match percentage, a percentage for every result, and a percentage for each measurement. High risk is marked on hazards, sunlight, shadows, or radiation when that spot is dangerous for that factor. Sliders set how much each one counts. A click still searches by elevation. The images are in `data/layers/` and are built by `python3 scripts/build_layers.py`.

| Map | Earth | Moon | Mars |
|---|---|---|---|
| Temperature | MERRA-2 air temperature, July 2024 | Diviner hottest temperature, full globe | TES afternoon temperature, Mars Year 25 |
| Ice and water | Ocean from ETOPO1, AMSR2 sea ice, MODIS snow | Diviner coldest temperature (ice below ~110 K) | Odyssey epithermal neutrons (hydrogen) |
| Geology | Geological Survey of Canada units, 1995 | USGS unified geologic units, 2020 | USGS SIM 3292 units, Tanaka and others 2014 |
| Weather | IMERG rain, 15 July 2024, on the Blue Marble | Diviner day–night temperature swing on the WAC mosaic (no atmosphere) | MY34 global dust storm (Montabone et al. 2020), same smooth color field as the Moon |
| Light | Blue Marble, August 2004 | LRO WAC mosaic | Viking MDIM color mosaic |
| Radiation | CERES outgoing longwave, July 2018 | Diviner midnight heat radiation | Odyssey fast neutrons |

The source line for the map on screen is written under **Data sources and landmarks**. `python3 scripts/replace_layers.py` rebuilds the Moon and Mars maps other than the two mosaics.

## Usage

```bash
pip install -r requirements.txt
npm install

python -m terrain_bot serve --port 8010        # Python terrain API
npm start                                      # TypeScript UI at http://localhost:8000
                                                 # default: click the Moon or Mars, matches land on Earth
                                                 # every search is stored in the database
python -m terrain_bot query --body moon --lat -70.36 --lon -172.49 --surface underwater
python -m terrain_bot query --body mars --lat 18.65 --lon -133.8 --target moon
python -m terrain_bot analyze --bodies earth mars --top 20   # results/: CSV, JSON, summary, map, histogram
python -m terrain_bot chat                     # "olympus on moon steps", "moon 0 0 underwater", "top 10 earth mars"
python -m terrain_bot validate                 # image accuracy vs real data
python -m terrain_bot calibrate                # data ranges and landmark checks
```

Add `--source images` or `--source images-fit` to use the grayscale images. Use your own images
with `--earth`, `--moon` and `--mars`. They are assumed to be equirectangular. Set the bounds with
`--earth-lat-range S N` (and the related flags) or with a `<image>.json` sidecar. Transparent
pixels are ignored, and 16-bit PNG/TIFF files are read at full precision.

To share the UI temporarily, run `cloudflared tunnel --url http://localhost:8000`.

## Caveats

* The quarter-degree DEM cells can't capture single peaks or trench bottoms. Everest's cell reads
  5,380 m and the Mariana cell reads -8,810 m.
* Image calibration is linear (or an isotonic fit for `images-fit`) and assumes brightness tracks
  height. That holds for the bundled Earth and Moon maps; Mars saturates at the extremes.

## Tests

```bash
python -m pytest
```
