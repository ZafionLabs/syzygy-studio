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
| Earth | 0.670 | 1,517 m | 401 m | shaded relief; land/water agrees on only 72% |

The Moon image is a faithful elevation map: its lightest pixel is 13 km from the real Selenean
summit. The Mars image has the right geography but saturates at both ends, so Olympus Mons is as
white as the Tharsis plains. The straight Olympus-to-Hellas scale therefore reads about 5 km too
high, but the fitted curve fixes that except in the clipped areas, which the steps flag. The Earth image is a shaded-relief picture, so brightness isn't height. Lowlands like the
Amazon are as dark as abyssal plains, and continental shelves are as bright as hills. No
brightness-to-height curve can fix that, which is why measured data is the default. The image's
bounds (84 N to 77 S) were fitted against ETOPO1 and stored in `data/earth_heightmap.json`.

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

## Usage

```bash
pip install -r requirements.txt

python -m terrain_bot serve                    # browser UI at http://localhost:8000
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
* Image calibration is linear and assumes brightness is proportional to height.

## Tests

```bash
python -m pytest
```
