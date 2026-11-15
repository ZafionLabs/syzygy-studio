# Replacement maps

Files in this folder are used instead of the bundled maps.

| File | Replaces |
|---|---|
| `earth_elevation.npz`, `moon_elevation.npz`, `mars_elevation.npz` | The elevation grid for that body |
| `earth_elevation.npy`, `moon_elevation.npy`, `mars_elevation.npy` | The same, as a full-globe grid from −90 to 90 and −180 to 180 (upload this; the app stores an `.npz`) |
| `{body}_temperature.png`, `{body}_ice-water.png`, `{body}_geology.png`, `{body}_weather.png`, `{body}_light.png`, `{body}_radiation.png` | That science map |

An elevation `.npz` needs an `elevation` array in metres. `lat_range` and `lon_range` are optional
and default to the whole globe. A science map is a PNG in the same simple-cylindrical frame as the
built-in maps: north at the top, 180° west at the left.

You can also send the file from the **New map** control in the app. The originals in `data/dem`
and `data/layers` stay as they are.
