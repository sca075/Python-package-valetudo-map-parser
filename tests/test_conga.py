"""
Tests for Conga vacuum support:
- Detection via ImageData.is_conga_map and BaseHandler.detect_vacuum_type
- Image generation via CongaMapImageHandler
- Room extraction with congaPixels data
- Calibration data (4 vacuum↔map point pairs)
- Rooms data structure (name, centre coordinates, outline)
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SCR.valetudo_map_parser.config.colors import ColorsManagement
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.config.utils import BaseHandler
from SCR.valetudo_map_parser.conga_handler import CongaMapImageHandler
from SCR.valetudo_map_parser.map_data import ImageData


logging.basicConfig(level=logging.WARNING)
_LOGGER = logging.getLogger(__name__)

CONGA_JSON = os.path.join(os.path.dirname(__file__), "concaduto.json")
HYPFER_JSON = os.path.join(os.path.dirname(__file__), "test.json")


@pytest.fixture(scope="module")
def conga_data() -> dict:
    with open(CONGA_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def hypfer_data() -> dict:
    with open(HYPFER_JSON) as f:
        return json.load(f)


@pytest.fixture
def conga_shared():
    device_info = {
        "platform": "linux",
        "user_language": "en",
        "robot_size": 25,
        "margins": 50,
        "image_rotate": "0",
        "image_flip": "None",
        "aspect_ratio": "None",
        "zoom_lock_ratio": True,
        "trim_top": 0,
        "trim_bottom": 0,
        "trim_left": 0,
        "trim_right": 0,
        "show_vac_status": False,
        "color_charger": [0, 128, 0],
        "color_move": [238, 247, 255],
        "color_wall": [255, 255, 0],
        "color_robot": [255, 255, 204],
        "color_go_to": [0, 255, 0],
        "color_no_go": [255, 0, 0],
        "color_zone_clean": [255, 255, 255],
        "color_background": [25, 25, 112],
        "color_text": [255, 255, 255],
        "color_mop_move": [100, 100, 200],
        "mop_path_width": 16,
        "obstacle_link_ip": "",
        "obstacle_link_port": 0,
        "obstacle_link_protocol": "http",
        "alpha_background": 255.0,
        "alpha_charger": 255.0,
        "alpha_move": 255.0,
        "alpha_robot": 255.0,
        "alpha_text": 255.0,
        "alpha_wall": 255.0,
        "alpha_zone_clean": 255.0,
        "alpha_no_go": 255.0,
        "alpha_go_to": 255.0,
        "alpha_mop_move": 255.0,
        "alpha_carpet": 255.0,
        "color_carpet": [150, 100, 120],
        "disable_carpets": False,
        "disable_material_overlay": True,
        "color_material_wood": [40, 40, 40],
        "alpha_material_wood": 25,
        "color_material_tile": [40, 40, 40],
        "alpha_material_tile": 40,
    }
    shared_mgr = CameraSharedManager("test_conga", device_info)
    shared = shared_mgr.get_instance()
    shared.vacuum_state = "cleaning"
    shared.vacuum_connection = True
    shared.vacuum_battery = 85
    ColorsManagement(shared).set_initial_colours(device_info)
    return shared


# --- Image generation test ---


@pytest.mark.asyncio
async def test_conga_image_generation(conga_shared, conga_data):
    handler = CongaMapImageHandler(conga_shared)
    image, _ = await handler.async_get_image(conga_data)
    print("frame number ", handler.get_frame_number())
    print("rooms: ", conga_shared.map_rooms)
    assert image is not None, "CongaMapImageHandler must return a valid image"
    w, h = image.size
    assert w > 0 and h > 0, "Image dimensions must be positive"
    image.show()


# --- Room extraction test ---


@pytest.mark.asyncio
async def test_conga_room_extraction(conga_shared, conga_data):
    handler = CongaMapImageHandler(conga_shared)
    await handler.async_get_image(conga_data)
    rooms = await handler.async_get_rooms_attributes()
    assert rooms, "Room extraction must return at least one room"
    _LOGGER.warning("Conga rooms extracted: %s", list(rooms.keys()))


# --- Calibration data test ---


@pytest.mark.asyncio
async def test_conga_calibration_data(conga_shared, conga_data):
    """Calibration data must contain 4 vacuum↔map point pairs after image generation."""
    handler = CongaMapImageHandler(conga_shared)
    await handler.async_get_image(conga_data)

    cal = conga_shared.attr_calibration_points
    assert cal is not None, (
        "Calibration points must be populated after image generation"
    )
    assert len(cal) == 4, f"Expected 4 calibration points, got {len(cal)}"

    for idx, point in enumerate(cal):
        assert "vacuum" in point, f"Point {idx} missing 'vacuum' key"
        assert "map" in point, f"Point {idx} missing 'map' key"
        for coord_key in ("x", "y"):
            assert coord_key in point["vacuum"], (
                f"Point {idx} vacuum missing '{coord_key}'"
            )
            assert coord_key in point["map"], f"Point {idx} map missing '{coord_key}'"

    _LOGGER.warning("Conga calibration points: %s", cal)


# --- Rooms data structure test ---


@pytest.mark.asyncio
async def test_conga_rooms_data(conga_shared, conga_data):
    """Each room must have a name, centre coordinates, and a non-empty outline."""
    handler = CongaMapImageHandler(conga_shared)
    await handler.async_get_image(conga_data)
    rooms = await handler.async_get_rooms_attributes()

    assert rooms, "Rooms data must not be empty"

    for room_id, props in rooms.items():
        assert "name" in props, f"Room {room_id} missing 'name'"
        assert "x" in props, f"Room {room_id} missing 'x' centre coordinate"
        assert "y" in props, f"Room {room_id} missing 'y' centre coordinate"
        assert "outline" in props and props["outline"], (
            f"Room {room_id} must have a non-empty outline"
        )
        _LOGGER.warning(
            "Room %s: name=%s centre=(%s, %s) outline_points=%s",
            room_id,
            props["name"],
            props["x"],
            props["y"],
            len(props["outline"]),
        )
