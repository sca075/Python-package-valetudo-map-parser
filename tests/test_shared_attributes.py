"""Test to check shared.to_dict()['attributes'] output."""
from __future__ import annotations

import json
import logging
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SCR.valetudo_map_parser.config.shared import CameraSharedManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

_LOGGER = logging.getLogger(__name__)


def test_shared_attributes():
    """Test shared.to_dict()['attributes'] output."""
    
    # Device info from test.py
    device_info = {
        "platform": "mqtt_vacuum_camera",
        "unique_id": "rockrobo_camera",
        "vacuum_config_entry": "9abe4e81972b00d1682c2363d3584041",
        "vacuum_map": "valetudo/rockrobo",
        "vacuum_identifiers": {("mqtt", "rockrobo")},
        "is_rand256": False,
        "alpha_background": 255.0,
        "alpha_charger": 255.0,
        "alpha_go_to": 255.0,
        "alpha_move": 50.0,
        "alpha_mop_move": 25.0,
        "alpha_no_go": 125.0,
        "alpha_robot": 255.0,
        "alpha_text": 255.0,
        "alpha_wall": 150.0,
        "alpha_zone_clean": 125.0,
        "aspect_ratio": "16: 9",
        "auto_zoom": True,
        "zoom_lock_ratio": True,
        "color_background": [0, 125, 255],
        "color_charger": [255, 128, 0],
        "color_go_to": [0, 255, 0],
        "color_move": [238, 247, 255],
        "color_mop_move": [238, 225, 255],
        "color_no_go": [255, 0, 0],
        "color_robot": [255, 255, 204],
        "color_text": [164, 25, 25],
        "color_wall": [255, 255, 0],
        "color_zone_clean": [255, 255, 255],
        "color_room_0": [135, 206, 250],
        "color_room_1": [176, 226, 255],
        "robot_size": 18,
        "mop_path_width": 10,
        "offset_top": 0,
        "offset_bottom": 0,
        "offset_left": 0,
        "offset_right": 0,
        "rotate_image": "270",
        "margins": "100",
        "show_vac_status": True,
        "vac_status_font": "SCR/valetudo_map_parser/config/fonts/FiraSans.ttf",
        "vac_status_position": True,
        "vac_status_size": 50.0,
        "enable_www_snapshots": False,
        "get_svg_file": False,
        "trims_data": {
            "floor": "floor_0",
            "trim_up": 2400,
            "trim_left": 2945,
            "trim_down": 3654,
            "trim_right": 3740,
        },
        "image_format": "image/jpeg",  # TEST: Set image format via device_info
    }
    
    _LOGGER.info("=" * 80)
    _LOGGER.info("TEST: image_format from device_info")
    _LOGGER.info("=" * 80)
    
    # Initialize shared data
    file_name = "test_shared_attributes"
    shared_data = CameraSharedManager(file_name, device_info)
    shared = shared_data.get_instance()
    
    # Set some vacuum state
    shared.vacuum_state = "docked"
    shared.dock_state = "mop cleaning"
    shared.vacuum_connection = True
    shared.vacuum_battery = 100
    shared.set_content_type("jpeg")
    
    # Check what image_format was set to
    _LOGGER.info(f"shared.get_content_type() = {shared.get_content_type()}")
    _LOGGER.info(f"Expected: 'image/jpeg'")

    # Get the to_dict output
    result = shared.to_dict()

    _LOGGER.info("\n" + "=" * 80)
    _LOGGER.info("COMPLETE shared.to_dict() OUTPUT")
    _LOGGER.info("=" * 80)
    _LOGGER.info(f"\n{json.dumps(result, indent=2, default=str)}")

    _LOGGER.info("\n" + "=" * 80)
    _LOGGER.info("VERIFICATION")
    _LOGGER.info("=" * 80)
    content_type = result['attributes']['content_type']
    _LOGGER.info(f"content_type in attributes: {content_type}")

    if content_type == "image/jpeg":
        _LOGGER.info("✅ SUCCESS: set_content_type('jpeg') is working!")
    else:
        _LOGGER.error(f"❌ FAIL: Expected 'image/jpeg', got '{content_type}'")
    
    return result


if __name__ == "__main__":
    test_shared_attributes()

