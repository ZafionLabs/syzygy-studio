"""Replacement maps dropped in by hand or uploaded from the app.

A file in ``data/updates`` is used instead of the bundled map for that body.
Elevation is a NumPy ``.npz`` with an ``elevation`` array in metres, plus
optional ``lat_range`` and ``lon_range``, or an image of the whole body whose
brightness is scaled from that body's lowest point to its highest. Every other
map is an image in the same simple-cylindrical frame: north at the top, 180°W
at the left, twice as wide as it is tall. PNG, JPEG, and any other image format
the server can read are accepted. A picture that is not that shape is refused, and so
is a map of a different body.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from .bodies import get_body
from .heightmap import calibrate

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


_DATA = Path(__file__).resolve().parents[1] / "data"
_PATTERN_SIZE = (180, 90)
_BODY_NAMES = {"earth": "Earth", "moon": "Moon", "mars": "Mars"}
_references: dict[tuple[str, str], np.ndarray | None] = {}


def _pattern(values: np.ndarray) -> np.ndarray | None:
    """Coarse brightness pattern, or ``None`` when the grid is flat."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    clean = np.where(finite, values, np.nanmean(values))
    image = Image.fromarray(clean.astype(np.float32), mode="F")
    image = image.resize(_PATTERN_SIZE, Image.Resampling.BOX)
    pattern = np.asarray(image, dtype=np.float64)
    pattern -= pattern.mean()
    scale = float(pattern.std())
    if scale < 1e-6:
        return None
    return pattern / scale


def _reference_pattern(body: str, kind: str) -> np.ndarray | None:
    key = (body, kind)
    if key not in _references:
        path = _DATA / f"{body}_heightmap.png" if kind == "elevation" else _DATA / "layers" / f"{body}_{kind}.png"
        if path.is_file():
            _references[key] = _pattern(np.asarray(Image.open(path).convert("L"), dtype=np.float64))
        else:
            _references[key] = None
    return _references[key]


def _same_body(pattern: np.ndarray | None, body: str, kind: str) -> None:
    """Refuse a map whose broad shape matches a different body."""
    if pattern is None:
        return
    scores: dict[str, float] = {}
    for other in BODIES:
        reference = _reference_pattern(other, kind)
        if reference is None:
            continue
        direct = float((pattern * reference).mean())
        shifted = float((np.roll(pattern, pattern.shape[1] // 2, axis=1) * reference).mean())
        scores[other] = max(direct, shifted)
    if body not in scores or len(scores) < 2:
        return
    match = max(scores, key=scores.get)
    if match != body and scores[match] >= 0.70 and scores[match] - scores[body] >= 0.06:
        found = "the Moon" if match == "moon" else _BODY_NAMES[match]
        wanted = "the Moon" if body == "moon" else _BODY_NAMES[body]
        raise ValueError(f"That map looks like {found}, not {wanted}. Place it on the {_BODY_NAMES[match]} spot.")


def _elevation_npz(raw: bytes, body: str) -> bytes:
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
    _same_body(_pattern(elev), body, "elevation")
    out = io.BytesIO()
    np.savez_compressed(out, elevation=elev, lat_range=np.array(lat_range), lon_range=np.array(lon_range))
    return out.getvalue()


def _looks_like_picture(raw: bytes) -> bool:
    return raw.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a")) or (
        raw.startswith(b"RIFF") and raw[8:12] == b"WEBP"
    )


def _open_image(raw: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise ValueError("That file is not an image the bot can read.") from exc
    return image


def _require_map_image(image: Image.Image) -> None:
    if image.width > 8192 or image.height > 8192:
        raise ValueError("map image must be at most 8192 pixels on each side")
    ratio = image.width / image.height
    if image.height < 64 or image.width < 128 or abs(ratio - 2.0) > 0.05:
        raise ValueError(
            "That picture is not a global map. It has to be twice as wide as it is tall, "
            "at least 128 by 64 pixels, with north at the top and 180° west at the left."
        )


def _layer_png(raw: bytes, body: str, kind: str) -> bytes:
    if len(raw) > MAX_BYTES:
        raise ValueError("map is larger than 80 MB")
    image = _open_image(raw)
    _require_map_image(image)
    _same_body(_pattern(np.asarray(image.convert("L"), dtype=np.float64)), body, kind)
    out = io.BytesIO()
    image.convert("RGB").save(out, format="PNG")
    return out.getvalue()


def _image_elevation_npz(raw: bytes, body: str) -> bytes:
    image = _open_image(raw)
    _require_map_image(image)
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    _same_body(_pattern(gray), body, "elevation")
    try:
        elevation, _, _ = calibrate(gray, np.ones(gray.shape, dtype=bool), get_body(body))
    except ValueError as exc:
        raise ValueError("That image has no height variation, so it cannot be used as an elevation map.") from exc
    out = io.BytesIO()
    np.savez_compressed(
        out,
        elevation=elevation,
        lat_range=np.array((-90.0, 90.0)),
        lon_range=np.array((-180.0, 180.0)),
    )
    return out.getvalue()


def _as_elevation_npz(raw: bytes, filename: str, body: str) -> bytes:
    if _looks_like_picture(raw):
        return _image_elevation_npz(raw, body)
    if filename.lower().endswith(".npy"):
        try:
            elev = np.load(io.BytesIO(raw), allow_pickle=False)
        except Exception as exc:
            raise ValueError("that .npy file is not a numeric elevation grid") from exc
        packed = io.BytesIO()
        np.savez_compressed(packed, elevation=np.asarray(elev), lat_range=np.array((-90.0, 90.0)), lon_range=np.array((-180.0, 180.0)))
        raw = packed.getvalue()
    return _elevation_npz(raw, body)


def prepare_map(body: str, kind: str, raw: bytes, filename: str = "") -> dict:
    """Validate ``raw`` and return the bytes to use for one browser visit."""
    body, kind = _check_body_kind(body, kind)
    if kind == "elevation":
        payload = _as_elevation_npz(raw, filename, body)
        name = f"{body}_elevation.npz"
    else:
        payload = _layer_png(raw, body, kind)
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
