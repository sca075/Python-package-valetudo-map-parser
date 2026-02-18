"""
Test bytes_format parameter consistency.

Verifies that data["image"] structure is always populated regardless of bytes_format value.
This fixes the API inconsistency discovered in mqtt_vacuum_camera PR #419.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SCR.valetudo_map_parser.config.colors import ColorsManagement
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.hypfer_handler import HypferMapImageHandler

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


async def test_bytes_format_consistency():
    """Test that data structure is consistent for both bytes_format values."""

    # Load test JSON
    json_path = Path(__file__).parent / "X40_carpet.json"
    with open(json_path) as f:
        json_data = json.load(f)

    # Setup device_info (minimal config)
    device_info = {
        "image_rotate": 0,
        "image_auto_zoom": True,
        "image_ref_height": 1080,
        "image_ref_width": 1920,
        "user_language": "en",
        "aspect_ratio": "None",
    }

    # Test with bytes_format=True
    _LOGGER.info("=" * 60)
    _LOGGER.info("Testing with bytes_format=True")
    _LOGGER.info("=" * 60)

    # Initialize shared data and handler
    file_name = "test_bytes_format_true"
    shared_data_true = CameraSharedManager(file_name, device_info)
    shared_true = shared_data_true.get_instance()
    colors_true = ColorsManagement(shared_true)
    colors_true.set_initial_colours(device_info)

    handler_true = HypferMapImageHandler(shared_true)
    image_true, data_true = await handler_true.async_get_image(
        m_json=json_data,
        bytes_format=True
    )
    
    _LOGGER.info(f"Image type: {type(image_true)}")
    _LOGGER.info(f"Data keys: {list(data_true.keys())}")
    _LOGGER.info(f"Data['image'] keys: {list(data_true.get('image', {}).keys())}")
    
    assert "image" in data_true, "❌ FAIL: data['image'] missing with bytes_format=True"
    assert "binary" in data_true["image"], "❌ FAIL: data['image']['binary'] missing"
    assert "size" in data_true["image"], "❌ FAIL: data['image']['size'] missing"
    assert isinstance(data_true["image"]["binary"], bytes), "❌ FAIL: binary should be bytes"
    assert isinstance(data_true["image"]["size"], tuple), "❌ FAIL: size should be tuple"
    
    _LOGGER.info(f"✅ Binary type: {type(data_true['image']['binary'])}")
    _LOGGER.info(f"✅ Binary size: {len(data_true['image']['binary'])} bytes")
    _LOGGER.info(f"✅ Image size: {data_true['image']['size']}")
    
    # Test with bytes_format=False
    _LOGGER.info("")
    _LOGGER.info("=" * 60)
    _LOGGER.info("Testing with bytes_format=False")
    _LOGGER.info("=" * 60)

    # Initialize shared data and handler for second test
    file_name = "test_bytes_format_false"
    shared_data_false = CameraSharedManager(file_name, device_info)
    shared_false = shared_data_false.get_instance()
    colors_false = ColorsManagement(shared_false)
    colors_false.set_initial_colours(device_info)

    handler_false = HypferMapImageHandler(shared_false)
    image_false, data_false = await handler_false.async_get_image(
        m_json=json_data,
        bytes_format=False
    )
    
    _LOGGER.info(f"Image type: {type(image_false)}")
    _LOGGER.info(f"Data keys: {list(data_false.keys())}")
    _LOGGER.info(f"Data['image'] keys: {list(data_false.get('image', {}).keys())}")
    
    # This is the critical test - data["image"] should exist even with bytes_format=False
    assert "image" in data_false, "❌ FAIL: data['image'] missing with bytes_format=False"
    assert "binary" in data_false["image"], "❌ FAIL: data['image']['binary'] missing"
    assert "size" in data_false["image"], "❌ FAIL: data['image']['size'] missing"
    assert isinstance(data_false["image"]["size"], tuple), "❌ FAIL: size should be tuple"
    
    _LOGGER.info(f"✅ Binary type: {type(data_false['image']['binary'])}")
    if isinstance(data_false['image']['binary'], bytes):
        _LOGGER.info(f"✅ Binary size: {len(data_false['image']['binary'])} bytes")
    _LOGGER.info(f"✅ Image size: {data_false['image']['size']}")
    
    # Verify sizes match
    assert data_true["image"]["size"] == data_false["image"]["size"], \
        "❌ FAIL: Image sizes don't match between bytes_format=True and False"
    
    _LOGGER.info("")
    _LOGGER.info("=" * 60)
    _LOGGER.info("✅ SUCCESS: API consistency verified!")
    _LOGGER.info("=" * 60)
    _LOGGER.info(f"Both bytes_format values return data['image'] structure")
    _LOGGER.info(f"Image size is consistent: {data_true['image']['size']}")
    _LOGGER.info("")
    _LOGGER.info("This fix enables mqtt_vacuum_camera PR #419 optimization:")
    _LOGGER.info("  - Can now use bytes_format=False to get PIL Image directly")
    _LOGGER.info("  - Enables direct PIL → JPEG conversion (1 step)")
    _LOGGER.info("  - Avoids PIL → PNG → PIL → JPEG (2 conversions)")


if __name__ == "__main__":
    asyncio.run(test_bytes_format_consistency())

