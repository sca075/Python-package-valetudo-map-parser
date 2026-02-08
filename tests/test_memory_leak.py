"""
Memory leak investigation test.
Simplified test to identify where memory is leaking.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import sys
import tracemalloc
from pathlib import Path

import psutil


# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from SCR.valetudo_map_parser.config.colors import ColorsManagement
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.hypfer_handler import HypferMapImageHandler


# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_LOGGER = logging.getLogger(__name__)

# Test Configuration
TEST_FILE = "X40_carpet.json"
FRAME_COUNT = 100


def format_bytes(bytes_val):
    """Format bytes to human readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}TB"


async def test_memory_leak():
    """Test for memory leaks in image generation."""
    # Load test data
    json_path = Path(__file__).parent / TEST_FILE
    with open(json_path) as f:
        json_data = json.load(f)

    # Setup device_info
    device_info = {
        "rand256": False,
        "image_rotate": 0,
        "image_auto_zoom": True,
        "image_ref_height": 1080,
        "image_ref_width": 1920,
        "user_language": "en",
        "disable_carpets": False,
        "disable_material_overlay": False,
    }

    # Initialize shared data
    shared_data = CameraSharedManager("test_memory", device_info)
    shared = shared_data.get_instance()
    shared.vacuum_state = "docked"
    shared.vacuum_connection = True
    shared.vacuum_battery = 100

    # Initialize colors
    colors = ColorsManagement(shared)
    colors.set_initial_colours(device_info)

    # Start memory tracking
    tracemalloc.start()
    process = psutil.Process()

    print(f"\n{'=' * 70}")
    print(f"Memory Leak Investigation - {FRAME_COUNT} frames")
    print(f"{'=' * 70}\n")

    mem_start = process.memory_info().rss / 1024 / 1024
    snapshot_start = tracemalloc.take_snapshot()

    print(f"Starting memory: {mem_start:.1f}MB\n")

    # Create handler ONCE and reuse it across frames
    handler = HypferMapImageHandler(shared)

    for frame_num in range(FRAME_COUNT):
        gc.collect()

        # Generate image using the same handler instance
        image = await handler.async_get_image(json_data)

        # Clean up
        if image and isinstance(image, tuple) and len(image) > 0:
            pil_image = image[0]
            if pil_image and hasattr(pil_image, "close"):
                pil_image.close()
        del image

        # Check memory every frame
        mem_current = process.memory_info().rss / 1024 / 1024
        mem_growth = mem_current - mem_start

        if frame_num % 5 == 4 or frame_num == 0:
            print(
                f"Frame {frame_num + 1:2d}: {mem_current:.1f}MB (growth: +{mem_growth:.1f}MB)"
            )

    gc.collect()
    mem_end = process.memory_info().rss / 1024 / 1024
    total_growth = mem_end - mem_start

    # Take final snapshot and compare
    snapshot_end = tracemalloc.take_snapshot()
    top_stats = snapshot_end.compare_to(snapshot_start, "lineno")

    print(f"\n{'=' * 70}")
    print(f"Final memory: {mem_end:.1f}MB")
    print(
        f"Total growth: +{total_growth:.1f}MB ({total_growth / FRAME_COUNT:.1f}MB per frame)"
    )
    print(f"{'=' * 70}\n")

    print("Top 10 memory allocations:")
    for stat in top_stats[:10]:
        print(f"{stat}")

    tracemalloc.stop()


if __name__ == "__main__":
    asyncio.run(test_memory_leak())
