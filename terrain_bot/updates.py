"""Replacement maps dropped in by hand or uploaded from the app.

A file in ``data/updates`` is used instead of the bundled map for that body.
Elevation is a NumPy ``.npz`` with an ``elevation`` array in metres, plus
optional ``lat_range`` and ``lon_range``. Every other map is a PNG in the same
simple-cylindrical frame as the built-in maps: north at the top, 180°W at the left,
twice as wide as it is tall. A photograph or any other picture is refused.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

UPDATE_DIR = Path(__file__).resolve().parents[1] / "data" / "updates"
BODIES = ("earth", "moon", "mars")
KINDS = ("elevation", "temperature", "ice-water", "geology", "weather", "light", "radiation")
MAX_BYTES = 80_000_000


def elevation_file(body: str, update_dir: Path | None = None) -> Path | None:
    path = (UPDATE_DIR if update_dir is None else update_dir) / f"{body}_elevation.npz"
    return path if path.is_file() else None


def layer_file(body: str, layer: str, update_dir: Path | None = None) -> Path | None:
    path = (UPDATE_DIR if update_dir is None else update_dir) / f"{body}_{layer}.png"
    return path if path.is_file() else None


def active_updates(update_dir: Path | None = None) -> list[dict]:
    update_dir = UPDATE_DIR if update_dir is None else update_dir
    found = []
    if not update_dir.is_dir():
        return found
    for body in BODIES:
        if elevation_file(body, update_dir):
            found.append({"body": body, "kind": "elevation", "file": f"{body}_elevation.npz"})
        for kind in KINDS:
            if kind == "elevation":
                continue
            if layer_file(body, kind, update_dir):
                found.append({"body": body, "kind": kind, "file": f"{body}_{kind}.png"})
    return found


def _check_body_kind(body: str, kind: str) -> tuple[str, str]:
    body = body.lower()
    kind = kind.lower()
    if body not in BODIES:
        raise ValueError("body must be earth, moon, or mars")
    if kind not in KINDS:
        raise ValueError(f"map must be one of {', '.join(KINDS)}")
    return body, kind


def _elevation_npz(raw: bytes) -> bytes:
    if len(raw) > MAX_BYTES:
        raise ValueError("map is larger than 80 MB")
    try:
        with np.load(io.BytesIO(raw)) as packed:
            elev = np.asarray(packed["elevation"], dtype=np.float64)
            lat_range = tuple(float(x) for x in packed["lat_range"]) if "lat_range" in packed.files else (-90.0, 90.0)
            lon_range = tuple(float(x) for x in packed["lon_range"]) if "lon_range" in packed.files else (-180.0, 180.0)
    except Exception as exc:
        raise ValueError("elevation file must be a .npz with an elevation array in metres") from exc
    if elev.ndim != 2 or elev.shape[0] < 2 or elev.shape[1] < 2:
        raise ValueError("elevation array must be a grid with at least 2 rows and 2 columns")
    if not np.isfinite(elev).any():
        raise ValueError("elevation array has no finite values")
    if len(lat_range) != 2 or len(lon_range) != 2 or not (-90 <= lat_range[0] < lat_range[1] <= 90):
        raise ValueError("lat_range must run from a lower latitude to a higher one, inside -90 to 90")
    if not (-180 <= lon_range[0] < lon_range[1] <= 180):
        raise ValueError("lon_range must run from a lower longitude to a higher one, inside -180 to 180")
    out = io.BytesIO()
    np.savez_compressed(out, elevation=elev, lat_range=np.array(lat_range), lon_range=np.array(lon_range))
    return out.getvalue()


def _looks_like_picture(raw: bytes) -> bool:
    return raw.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a")) or (
        raw.startswith(b"RIFF") and raw[8:12] == b"WEBP"
    )


def _layer_png(raw: bytes) -> bytes:
    if len(raw) > MAX_BYTES:
        raise ValueError("map is larger than 80 MB")
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        if _looks_like_picture(raw):
            raise ValueError(
                "That file is a picture, not a map. Drop a PNG of the whole body, twice as wide as it is tall, "
                "with north at the top and 180° west at the left."
            )
        raise ValueError("That file is not a PNG map.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise ValueError("that file is not an image the bot can read") from exc
    if image.width > 8192 or image.height > 8192:
        raise ValueError("map image must be at most 8192 pixels on each side")
    ratio = image.width / image.height
    if image.height < 64 or image.width < 128 or abs(ratio - 2.0) > 0.05:
        raise ValueError(
            "That picture is not a global map. It has to be a PNG twice as wide as it is tall, "
            "at least 128 by 64 pixels, with north at the top and 180° west at the left."
        )
    out = io.BytesIO()
    image.convert("RGB").save(out, format="PNG")
    return out.getvalue()


def _as_elevation_npz(raw: bytes, filename: str) -> bytes:
    if _looks_like_picture(raw):
        raise ValueError("That is a picture. Elevation has to be a .npz or .npy grid of heights in metres.")
    if filename.lower().endswith(".npy"):
        try:
            elev = np.load(io.BytesIO(raw), allow_pickle=False)
        except Exception as exc:
            raise ValueError("that .npy file is not a numeric elevation grid") from exc
        packed = io.BytesIO()
        np.savez_compressed(packed, elevation=np.asarray(elev), lat_range=np.array((-90.0, 90.0)), lon_range=np.array((-180.0, 180.0)))
        raw = packed.getvalue()
    return _elevation_npz(raw)


def prepare_map(body: str, kind: str, raw: bytes, filename: str = "") -> dict:
    """Validate ``raw`` and return the bytes to use for one browser visit."""
    body, kind = _check_body_kind(body, kind)
    if kind == "elevation":
        payload = _as_elevation_npz(raw, filename)
        name = f"{body}_elevation.npz"
    else:
        payload = _layer_png(raw)
        name = f"{body}_{kind}.png"
    return {"body": body, "kind": kind, "file": name, "bytes": len(payload), "payload": payload}


def install_map(body: str, kind: str, raw: bytes, filename: str = "", update_dir: Path | None = None) -> dict:
    """Validate ``raw`` and save it so this body uses it instead of the bundled map."""
    prepared = prepare_map(body, kind, raw, filename)
    update_dir = UPDATE_DIR if update_dir is None else update_dir
    update_dir.mkdir(parents=True, exist_ok=True)
    payload = prepared.pop("payload")
    name = prepared["file"]
    path = update_dir / name
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {"body": body, "kind": kind, "file": name, "bytes": len(payload)}
