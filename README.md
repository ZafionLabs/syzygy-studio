# terrain-bot: Earth vs Moon look-alike terrain

`terrain-bot` finds the places on Earth and the Moon whose terrain is most alike, using **real
elevations**. Earth's underwater sea floor is included, and every answer comes with the
step-by-step working.

## Data

| Source (`--source`) | Earth | Moon |
|---|---|---|
| `dem` (default): real measured data in `data/dem/` | NOAA **ETOPO1** 1 arc-minute global relief (land + ocean bathymetry, ice surface), sampled every 0.25 deg | NASA **LRO LOLA LDEM_4**, 4 px/deg, heights relative to the 1737.4 km mean radius |
| `images`: your grayscale maps in `data/` | lightest pixel = Mount Everest **+8,848.86 m**, darkest = Challenger Deep (Mariana Trench) **-10,935 m** | lightest = Selenean summit **+10,786 m**, darkest = Point Trieste (Antoniadi crater) **-9,129 m** |

Run `python -m terrain_bot download-dem` to refetch the DEMs (about 3.5 MB total).

### How accurate are the images? (`python -m terrain_bot validate`)

Each calibrated image is compared pixel by pixel with the real data at the same coordinates:

| Image | Rank correlation | Median error | Land/water |
|---|---|---|---|
| Moon | **0.989** | 400 m | n/a |
| Earth | 0.670 | 1,517 m | agrees on 72% (the image says 86% is underwater; reality is 72%) |

The Moon image is a faithful elevation map: its lightest pixel is 13 km from the real Selenean
summit. The Earth image is a shaded-relief picture, so brightness isn't height. Lowlands like the
Amazon are as dark as abyssal plains, and continental shelves are as bright as hills. No
brightness-to-height curve can fix that, which is why measured data is the default. The image's
bounds (84 N to 77 S) were fitted against ETOPO1 and stored in `data/earth_heightmap.json`.

## How a match is found (shown by `query`, the chat bot's `steps`, and the web UI)

1. **Locate** the coordinate's pixel. Resolution limits precision: about 28 km per pixel for Earth
   and 7.6 km for the Moon with the DEMs.
2. **Read the elevation.** DEM values are already in metres. For images,
   `elevation = min + (gray - darkest) / (lightest - darkest) x (max - min)`.
3. **Classify** the point: Earth land or underwater (sea level = 0 m). The Moon has no sea level,
   so its heights are relative to the mean radius.
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
python -m terrain_bot analyze --top 20         # results/: CSV, JSON, summary, map, histogram
python -m terrain_bot chat                     # "everest steps", "moon 0 0 underwater", "top 10"
python -m terrain_bot validate                 # image accuracy vs real data
python -m terrain_bot calibrate                # data ranges and landmark checks
```

Add `--source images` to use the grayscale images instead of the DEMs. Use your own images with
`--earth path.png --moon path.png`. They are assumed to be equirectangular. Set the bounds with
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
