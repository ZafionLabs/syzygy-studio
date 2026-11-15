"""Extra science maps (temperature, ice, geology, weather, light, radiation)."""

from __future__ import annotations

import json
from pathlib import Path

LAYER_DIR = Path(__file__).resolve().parents[1] / "data" / "layers"


def catalog() -> dict:
    return json.loads((LAYER_DIR / "catalog.json").read_text())


def layer_path(body: str, layer: str) -> Path:
    from .updates import layer_file

    custom = layer_file(body, layer)
    path = custom or (LAYER_DIR / f"{body}_{layer}.png")
    if not path.is_file():
        raise ValueError(f"no {layer} map for {body}")
    return path


def layer_png(body: str, layer: str) -> bytes:
    return layer_path(body, layer).read_bytes()
