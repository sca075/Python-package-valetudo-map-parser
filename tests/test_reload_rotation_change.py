"""Test reload scenario when rotation changes with saved floor data."""

from __future__ import annotations

import json
import logging
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SCR.valetudo_map_parser.config.colors import ColorsManagement
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
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

TEST_FILE = "X40_carpet.json"


def test_reload_with_rotation_change():
    """Test reload scenario: saved trims from 270° rotation, but config changed to 0°."""
    print("\n" + "=" * 70)
    print("RELOAD TEST: Rotation Change with Saved Floor Data")
    print("=" * 70)
    _LOGGER.info("=" * 70)
    _LOGGER.info("RELOAD TEST: Rotation Change with Saved Floor Data")
    _LOGGER.info("=" * 70)

    # Load test JSON
    test_json_path = os.path.join(os.path.dirname(__file__), TEST_FILE)
    with open(test_json_path, encoding="utf-8") as f:
        test_data = json.load(f)

    _LOGGER.info(f"Loaded test JSON: {TEST_FILE}")

    # SCENARIO 1: Initial run at 270° rotation
    print("\n" + "=" * 70)
    print("SCENARIO 1: Initial run at 270° - Generate and save trims")
    print("=" * 70)

    device_info_270 = {
        "vacuum_connection_string": "mqtt://localhost:1883",
        "vacuum_identifiers": "test_vacuum",
        "vacuum_model": "Dreame.vacuum.r2228o",
        "auto_zoom": True,
        "margins": 100,
        "rotate_image": 270,
        "aspect_ratio": "1, 1",
    }

    shared_data_270 = CameraSharedManager("test_reload_270", device_info_270)
    shared_270 = shared_data_270.get_instance()
    shared_270.vacuum_state = "docked"
    shared_270.vacuum_connection = True
    shared_270.vacuum_battery = 100

    colors_270 = ColorsManagement(shared_270)
    colors_270.set_initial_colours(device_info_270)

    handler_270 = HypferMapImageHandler(shared_270)

    import asyncio

    img_270, data_270 = asyncio.run(handler_270.async_get_image(test_data, bytes_format=False))
    image_270 = img_270.copy() if img_270 else None

    print(f"✅ Generated image at 270°")
    print(f"   Trims: {shared_270.trims.to_dict()}")
    print(f"   Calibration points: {shared_270.attr_calibration_points}")
    print(f"   Image size: {image_270.size if image_270 else 'None'}")

    # Save the trims as if they were persisted to config
    saved_trims_270 = shared_270.trims.to_dict()
    saved_floors_data = shared_270.floors_trims.copy()

    print(f"\n📝 Saved to config (simulating HA persistence):")
    print(f"   trims_data: {saved_trims_270}")
    print(f"   floors_data: {saved_floors_data}")

    # Clean up
    del shared_data_270, shared_270, handler_270, colors_270
    if image_270:
        image_270.close()

    # SCENARIO 2: Reload with rotation changed to 0° but old trims
    print("\n" + "=" * 70)
    print("SCENARIO 2: Reload at 0° with OLD trims from 270°")
    print("=" * 70)

    device_info_0_with_old_trims = {
        "vacuum_connection_string": "mqtt://localhost:1883",
        "vacuum_identifiers": "test_vacuum",
        "vacuum_model": "Dreame.vacuum.r2228o",
        "auto_zoom": True,
        "margins": 100,
        "rotate_image": 0,  # ⚠️ CHANGED to 0°
        "aspect_ratio": "1, 1",
        "trims_data": saved_trims_270,  # ⚠️ OLD trims from 270°
        "floors_data": saved_floors_data,  # ⚠️ OLD floor data
    }

    print(f"⚠️  Config has rotation=0° but trims from 270°:")
    print(f"   rotate_image: {device_info_0_with_old_trims['rotate_image']}")
    print(f"   trims_data: {device_info_0_with_old_trims['trims_data']}")

    shared_data_0 = CameraSharedManager("test_reload_0", device_info_0_with_old_trims)
    shared_0 = shared_data_0.get_instance()
    shared_0.vacuum_state = "docked"
    shared_0.vacuum_connection = True
    shared_0.vacuum_battery = 100

    colors_0 = ColorsManagement(shared_0)
    colors_0.set_initial_colours(device_info_0_with_old_trims)

    handler_0 = HypferMapImageHandler(shared_0)

    img_0, data_0 = asyncio.run(handler_0.async_get_image(test_data, bytes_format=False))
    image_0 = img_0.copy() if img_0 else None

    print(f"\n✅ Generated image at 0° with old trims:")
    print(f"   NEW Trims (after auto-crop): {shared_0.trims.to_dict()}")
    print(f"   NEW Calibration points: {shared_0.attr_calibration_points}")
    print(f"   Image size: {image_0.size if image_0 else 'None'}")

    # Save images for visual inspection
    if image_0:
        image_0.save("/tmp/reload_test_0_with_old_trims.png")
        print(f"\n💾 Saved image to /tmp/reload_test_0_with_old_trims.png")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\n✅ If image looks correct and calibration points are different,")
    print("   the reload scenario handles rotation changes properly!")


if __name__ == "__main__":
    test_reload_with_rotation_change()

