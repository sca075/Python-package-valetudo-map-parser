"""
Material overlay tests for the Python renderer (config/material.py).

Checks tile generation, tile_block wrapping, that the overlay really changes the
rendered map, and logs (does not assert) rendering time and memory growth.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import pytest


# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from SCR.valetudo_map_parser.config.colors import ColorsManagement
from SCR.valetudo_map_parser.config.material import (
    MaterialColors,
    MaterialTileRenderer,
)
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.hypfer_handler import HypferMapImageHandler


_LOGGER = logging.getLogger(__name__)

# ----- Test Configuration -----
TEST_FILE = "X40_carpet.json"  # contains wood_horizontal, wood_vertical and tile rooms
FRAME_COUNT = 20  # Frames rendered for the timing/memory report
WOOD_RGBA = (255, 0, 0, 200)
TILE_RGBA = (0, 0, 255, 200)
# ------------------------------

TEST_COLORS = MaterialColors(wood_rgba=WOOD_RGBA, tile_rgba=TILE_RGBA)

# material name -> (cells per tile, expected seam color)
MATERIALS = {
    "wood_horizontal": (36, WOOD_RGBA),
    "wood_vertical": (36, WOOD_RGBA),
    "tile": (4, TILE_RGBA),
}


@pytest.mark.parametrize("pixel_size", [5, 7, 8])
@pytest.mark.parametrize("material", list(MATERIALS))
def test_get_tile_shape_and_color(material: str, pixel_size: int):
    cells, expected_rgba = MATERIALS[material]
    tile = MaterialTileRenderer.get_tile(material, pixel_size, TEST_COLORS)

    assert tile is not None
    assert tile.dtype == np.uint8
    assert tile.shape == (cells * pixel_size, cells * pixel_size, 4)

    # Seams are drawn only in the material color; everything else is transparent.
    seam_pixels = tile[tile[..., 3] > 0]
    assert len(seam_pixels) > 0
    assert np.unique(seam_pixels, axis=0).tolist() == [list(expected_rgba)]


def test_get_tile_invalid_input_returns_none():
    assert MaterialTileRenderer.get_tile("carpet", 5, TEST_COLORS) is None
    assert MaterialTileRenderer.get_tile("tile", 0, TEST_COLORS) is None


def test_get_tile_default_colors():
    tile = MaterialTileRenderer.get_tile("tile", 5)
    seam_pixels = tile[tile[..., 3] > 0]
    expected = list(MaterialColors().tile_rgba)
    assert np.unique(seam_pixels, axis=0).tolist() == [expected]


def test_tile_block_wraps_around_tile():
    tile = MaterialTileRenderer.get_tile("tile", 5, TEST_COLORS)
    th, tw, _ = tile.shape

    # A block that crosses the tile border must repeat the tile (modulo indexing).
    r0, c0 = th - 2, tw - 3
    block = MaterialTileRenderer.tile_block(tile, r0, r0 + 5, c0, c0 + 5)

    assert block.shape == (5, 5, 4)
    for i in range(5):
        for j in range(5):
            assert (block[i, j] == tile[(r0 + i) % th, (c0 + j) % tw]).all()


def _load_json() -> dict:
    with open(Path(__file__).parent / TEST_FILE, encoding="utf-8") as f:
        return json.load(f)


def _make_handler(file_name: str, use_material: bool) -> HypferMapImageHandler:
    device_info = {
        "rand256": False,
        "image_rotate": 0,
        "image_auto_zoom": True,
        "user_language": "en",
        "disable_carpets": False,
        "disable_material_overlay": not use_material,
        "color_material_wood": list(WOOD_RGBA[:3]),
        "alpha_material_wood": WOOD_RGBA[3],
        "color_material_tile": list(TILE_RGBA[:3]),
        "alpha_material_tile": TILE_RGBA[3],
    }
    shared = CameraSharedManager(file_name, device_info).get_instance()
    shared.vacuum_state = "docked"
    shared.vacuum_connection = True
    shared.vacuum_battery = 100
    ColorsManagement(shared).set_initial_colours(device_info)
    return HypferMapImageHandler(shared)


async def _render(handler: HypferMapImageHandler, json_data: dict) -> np.ndarray:
    result = await handler.async_get_image(json_data)
    pil_image = result[0] if isinstance(result, tuple) else result
    array = np.array(pil_image)
    pil_image.close()
    return array


def test_material_overlay_changes_rendered_map():
    json_data = _load_json()
    with_material = asyncio.run(
        _render(_make_handler("material_on", True), json_data)
    )
    without_material = asyncio.run(
        _render(_make_handler("material_off", False), json_data)
    )

    assert with_material.shape == without_material.shape
    changed_pixels = (with_material != without_material).any(axis=-1).sum()
    assert changed_pixels > 0, "Material overlay did not change the rendered map"


async def _measure(handler: HypferMapImageHandler, json_data: dict) -> dict:
    process = psutil.Process()
    times = []
    mem_start = process.memory_info().rss / 1024 / 1024  # MB

    # Reuse the same handler across frames, like the integration does.
    for _ in range(FRAME_COUNT):
        gc.collect()
        start = time.perf_counter()
        await _render(handler, json_data)
        times.append((time.perf_counter() - start) * 1000)

    mem_growth = process.memory_info().rss / 1024 / 1024 - mem_start
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "mem_growth_mb": mem_growth,
    }


def test_material_overlay_performance_report():
    """Log render time and memory with and without materials (informational only)."""
    json_data = _load_json()
    results = {
        "with materials": asyncio.run(
            _measure(_make_handler("perf_material_on", True), json_data)
        ),
        "without materials": asyncio.run(
            _measure(_make_handler("perf_material_off", False), json_data)
        ),
    }

    for name, r in results.items():
        _LOGGER.info(
            "%s (%d frames): avg %.1fms, min %.1fms, max %.1fms, memory %+.1fMB",
            name,
            FRAME_COUNT,
            r["avg_ms"],
            r["min_ms"],
            r["max_ms"],
            r["mem_growth_mb"],
        )
        assert r["avg_ms"] > 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_material_overlay_performance_report()
