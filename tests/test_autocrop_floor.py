"""Test auto crop and floor data functionality."""

from __future__ import annotations

import json
import logging
import os
import sys


# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SCR.valetudo_map_parser.config.colors import ColorsManagement
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.config.types import FloorData, TrimsData
from SCR.valetudo_map_parser.hypfer_handler import HypferMapImageHandler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)],
)

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)

# Test configuration
TEST_FILE = "X40_carpet.json"


def test_autocrop_and_floor_data():
    """Test auto crop and floor data handling."""
    print("\n" + "=" * 70)
    print("Auto Crop and Floor Data Test")
    print("=" * 70)
    _LOGGER.info("=" * 70)
    _LOGGER.info("Auto Crop and Floor Data Test")
    _LOGGER.info("=" * 70)

    # Load test JSON
    test_json_path = os.path.join(os.path.dirname(__file__), TEST_FILE)
    with open(test_json_path, encoding="utf-8") as f:
        test_data = json.load(f)

    _LOGGER.info(f"Loaded test JSON: {TEST_FILE}")

    # Device info with trims initialized to 0
    device_info = {
        "vacuum_connection_string": "mqtt://localhost:1883",
        "vacuum_identifiers": "test_vacuum",
        "vacuum_model": "Dreame.vacuum.r2228o",
        "auto_zoom": True,
        "margins": 100,
        "rotate_image": 270,
        "aspect_ratio": "16, 9",
        "trims_data": {
            "floor": "floor_0",
            "trim_up": 2400,
            "trim_left": 2945,
            "trim_down": 3654,
            "trim_right": 3704,
        },
    }

    # 270{'floor': 'floor_0', 'trim_up': 2400, 'trim_left': 2950, 'trim_down': 3649, 'trim_right': 3699}
    # 0 {'floor': 'floor_0', 'trim_up': 2400, 'trim_left': 2950, 'trim_down': 3649, 'trim_right': 3699}
    # "trims_data": {
    #     "floor": "floor_0",
    #     "trim_left": 1650,
    #     "trim_up": 2950,
    #     "trim_right": 3474,
    #     "trim_down": 3974,
    # },

    # Initialize shared data
    file_name = "test_autocrop"
    shared_data = CameraSharedManager(file_name, device_info)
    shared = shared_data.get_instance()
    shared.vacuum_state = "docked"
    shared.vacuum_connection = True
    shared.vacuum_battery = 100


    # Initialize colors
    colors = ColorsManagement(shared)
    colors.set_initial_colours(device_info)

    # Create handler
    handler = HypferMapImageHandler(shared)

    import asyncio

    # First image generation
    print(f"\n--- Generating Image 1 --- {shared.image_rotate}")
    img1, data1 = asyncio.run(handler.async_get_image(test_data, bytes_format=False))
    image1 = img1.copy() if img1 else None  # Copy to prevent it from being closed
    print(f"Trims {shared.file_name} image 1: {shared.trims.to_dict()}")
    #shared.reset_trims()

    #print(f"After Trims reset of {shared.file_name} image 1: {shared.trims}")
    print(f"Trims floor data image 1: {shared.floors_trims}")
    print(f"Robot position (vacuum coords): {shared.current_room}")
    print(f"Calibration points: {shared.attr_calibration_points}")
    print(f"Current imagesize: {image1.size}")

    # Second image generation
    shared.image_rotate = 0
    print(f"\n--- Generating Image 2 ---{shared.image_rotate}")

    img2, data2 = asyncio.run(handler.async_get_image(test_data, bytes_format=False))
    image2 = img2.copy() if img2 else None  # Copy to prevent it from being closed
    print(f"Trims after image 2: {shared.trims.to_dict()}")
    print(f"Robot position (vacuum coords): {shared.current_room}")
    print(f"Calibration points: {shared.attr_calibration_points}")
    print(f"Current imagesize: {image2.size}")

    # Third image generation
    shared.image_rotate = 90
    print(f"\n--- Generating Image 3 ---{shared.image_rotate}")

    img3, data3 = asyncio.run(handler.async_get_image(test_data, bytes_format=False))
    image3 = img3.copy() if img3 else None  # Copy to prevent it from being closed
    print(f"Trims after image 3: {shared.trims.to_dict()}")
    print(f"Robot position (vacuum coords): {shared.current_room}")
    print(f"Calibration points: {shared.attr_calibration_points}")
    print(f"Current imagesize: {image3.size}")

    # Third image generation
    shared.image_rotate = 180
    print(f"\n--- Generating Image 4 ---{shared.image_rotate}")
    img4, data4 = asyncio.run(handler.async_get_image(test_data, bytes_format=False))
    image4 = img4.copy() if img4 else None  # Copy to prevent it from being closed
    print(f"Trims after image 4: {shared.trims.to_dict()}")
    print(f"Robot position (vacuum coords): {shared.current_room}")
    print(f"Calibration points: {shared.attr_calibration_points}")
    print(f"Current imagesize: {image4.size}")
    # Save and display images
    if image1:
        image1.save("/tmp/autocrop_rotation_270.png")
        print(f"✅ Saved Image 1 (270°) to /tmp/autocrop_rotation_270.png")
        image1.show()
    if image2:
        image2.save("/tmp/autocrop_rotation_0.png")
        print(f"✅ Saved Image 2 (0°) to /tmp/autocrop_rotation_0.png")
        image2.show()
    if image3:
        image3.save("/tmp/autocrop_rotation_90.png")
        print(f"✅ Saved Image 3 (90°) to /tmp/autocrop_rotation_90.png")
        image3.show()
    if image4:
        image4.save("/tmp/autocrop_rotation_90.png")
        print(f"✅ Saved Image 4 (180°) to /tmp/autocrop_rotation_180.png")
        image4.show()

if __name__ == "__main__":
    test_autocrop_and_floor_data()
