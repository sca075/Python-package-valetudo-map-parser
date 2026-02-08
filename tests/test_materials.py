"""Test material overlay functionality."""

import asyncio
import json
import logging
import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.hypfer_handler import HypferMapImageHandler
from SCR.valetudo_map_parser.map_data import ImageData


logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


async def test_materials():
    """Test that materials are extracted and applied."""
    # Load test JSON
    json_path = Path(__file__).parent / "X40_carpet.json"
    with open(json_path) as f:
        json_data = json.load(f)

    # Test material extraction
    layers, active_zones, materials = ImageData.find_layers(
        json_data["layers"], None, None, None
    )

    _LOGGER.info("Materials extracted: %s", materials)
    assert materials == {
        "1": "wood_horizontal",
        "2": "tile",
        "3": "wood_vertical",
        "4": "wood_horizontal",
        "5": "wood_vertical",
        "6": "wood_vertical",
    }, f"Expected 6 materials, got {materials}"

    # Test image generation with materials
    device_info = {
        "rand256": False,
        "image_rotate": 0,
        "image_auto_zoom": False,
    }

    shared = CameraSharedManager("test_materials")
    shared.update_shared_data(device_info)

    handler = HypferMapImageHandler(shared)
    image_data = await handler.async_get_image_from_json(json_data)

    assert image_data is not None, "Image generation failed"
    assert handler.json_data is not None, "JSON data not stored"
    assert hasattr(handler.json_data, "materials"), "Materials not in json_data"
    assert handler.json_data.materials == materials, "Materials not stored correctly"

    _LOGGER.info("✅ Material extraction and storage test passed!")
    _LOGGER.info("Materials in handler: %s", handler.json_data.materials)

    # Save test image
    if image_data:
        output_path = Path("/tmp/test_materials_output.png")
        image_data.save(output_path)
        _LOGGER.info("✅ Test image saved to %s", output_path)
        _LOGGER.info("Image size: %s", image_data.size)

    return True


if __name__ == "__main__":
    result = asyncio.run(test_materials())
    if result:
        print("\n✅ All material tests passed!")
    else:
        print("\n❌ Material tests failed!")
        sys.exit(1)
