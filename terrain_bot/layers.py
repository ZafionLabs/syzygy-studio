"""Extra science maps (temperature, ice, geology, weather, light, radiation)."""

from __future__ import annotations

import json
from pathlib import Path

LAYER_DIR = Path(__file__).resolve().parents[1] / "data" / "layers"


def catalog() -> dict:
    return json.loads((LAYER_DIR / "catalog.json").read_text())


def layer_png(body: str, layer: str) -> bytes:
    path = LAYER_DIR / f"{body}_{layer}.png"
    if not path.is_file():
        raise ValueError(f"no {layer} map for {body}")
    return path.read_bytes()
