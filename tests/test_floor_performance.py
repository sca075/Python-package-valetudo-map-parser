"""
Performance test comparing auto-crop calculation vs pre-stored trims.
Tests floor switching scenario with two different maps.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from SCR.valetudo_map_parser.config.colors import ColorsManagement
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.hypfer_handler import HypferMapImageHandler


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Test configuration
ITERATIONS = 5  # Number of iterations for averaging


def load_json_file(filename: str) -> dict:
    """Load JSON test data."""
    json_path = Path(__file__).parent / filename
    with open(json_path, "r") as f:
        return json.load(f)


def create_device_info(trims_data: dict = None, floors_data: dict = None) -> dict:
    """Create device_info configuration."""
    device_info = {
        "vacuum_platform": "hypfer",
        "vacuum_connection_string": "mqtt://localhost:1883",
        "vacuum_identifiers": "test_vacuum",
        "vacuum_map_name": "test_map",
        "rotate_image": 0,
        "margins": 100,
        "aspect_ratio": "None",
        "auto_crop": True,
    }

    if trims_data:
        device_info["trims_data"] = trims_data

    if floors_data:
        device_info["floors_data"] = floors_data
        device_info["current_floor"] = "floor_0"

    return device_info


async def measure_image_generation(handler, test_data: bytes) -> float:
    """Measure time to generate a single image."""
    start_time = time.perf_counter()
    image, data = await handler.async_get_image(test_data, bytes_format=True)
    end_time = time.perf_counter()

    if image:
        image.close()  # Clean up

    return end_time - start_time


async def calculate_optimal_trims(json_file: str) -> dict:
    """Calculate optimal trims for a JSON file by running with 0 trims."""
    print(f"\n=== Calculating optimal trims for {json_file} ===")

    # Load test data
    test_json = load_json_file(json_file)
    test_data = json.dumps(test_json).encode("utf-8")

    # Create device_info with 0 trims
    device_info = create_device_info(
        trims_data={
            "floor": "floor_0",
            "trim_left": 0,
            "trim_up": 0,
            "trim_right": 0,
            "trim_down": 0,
        }
    )

    # Initialize handler
    shared = CameraSharedManager.create_instance(device_info)
    colors = ColorsManagement()
    drawables = Drawable()

    handler = HypferMapImageHandler(
        shared_manager=shared,
        colors=colors,
        drawables=drawables,
        image_config=image_config,
    )

    # Generate image to calculate trims
    image, data = await handler.async_get_image(test_data, bytes_format=True)

    if image:
        image.close()

    # Get calculated trims
    calculated_trims = shared.trims.to_dict()
    print(f"Calculated trims: {calculated_trims}")

    return calculated_trims


async def test_scenario(scenario_name: str, json_file: str, trims_data: dict) -> list:
    """Test a single scenario and return list of generation times."""
    print(f"\n--- Testing: {scenario_name} ---")

    # Load test data
    test_json = load_json_file(json_file)
    test_data = json.dumps(test_json).encode("utf-8")

    # Create device_info
    device_info = create_device_info(trims_data=trims_data)

    # Initialize handler
    shared = CameraSharedManager.create_instance(device_info)
    colors = ColorsPalette()
    drawables = Drawable()
    image_config = ImageConfig()

    handler = HypferMapImageHandler(
        shared_manager=shared,
        colors=colors,
        drawables=drawables,
        image_config=image_config,
    )

    # Run iterations
    times = []
    for i in range(ITERATIONS):
        gen_time = await measure_image_generation(handler, test_data)
        times.append(gen_time)
        print(f"  Iteration {i + 1}: {gen_time:.4f}s")

    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time:.4f}s")

    return times


async def main():
    """Main test function."""
    print("=" * 70)
    print("FLOOR SWITCHING PERFORMANCE TEST")
    print("Comparing auto-crop calculation vs pre-stored trims")
    print("=" * 70)

    # Known trims for both floors
    floor_0_trims = {
        "floor": "floor_0",
        "trim_up": 2950,
        "trim_left": 2400,
        "trim_down": 3699,
        "trim_right": 3649,
    }
    floor_1_trims = {
        "floor": "floor_1",
        "trim_up": 1980,
        "trim_left": 1650,
        "trim_down": 3974,
        "trim_right": 3474,
    }

    print(f"\nFloor 0 (X40_carpet.json) trims: {floor_0_trims}")
    print(f"Floor 1 (test.json) trims: {floor_1_trims}")

    # Zero trims for auto-crop testing
    zero_trims_floor_0 = {
        "floor": "floor_0",
        "trim_left": 0,
        "trim_up": 0,
        "trim_right": 0,
        "trim_down": 0,
    }

    zero_trims_floor_1 = {
        "floor": "floor_1",
        "trim_left": 0,
        "trim_up": 0,
        "trim_right": 0,
        "trim_down": 0,
    }

    # Step 1: Start with 0 trims, generate X40 (auto-crop calculates)
    print("\n\n### STEP 1: X40 with Auto-Crop (0 trims) ###")
    times_x40_zero = await test_scenario(
        "X40 - Auto-crop (0 trims)", "X40_carpet.json", zero_trims_floor_0
    )

    # Step 2: Reset to 0 trims, generate L10/test.json (auto-crop calculates)
    print("\n\n### STEP 2: L10 with Auto-Crop (0 trims) ###")
    times_l10_zero = await test_scenario(
        "L10 - Auto-crop (0 trims)", "test.json", zero_trims_floor_1
    )

    # Step 3: Load pre-stored trims and generate L10
    print("\n\n### STEP 3: L10 with Pre-Stored Trims ###")
    times_l10_stored = await test_scenario(
        "L10 - Pre-stored trims", "test.json", floor_1_trims
    )

    # Step 4: Load pre-stored trims and generate X40
    print("\n\n### STEP 4: X40 with Pre-Stored Trims ###")
    times_x40_stored = await test_scenario(
        "X40 - Pre-stored trims", "X40_carpet.json", floor_0_trims
    )

    # Summary
    print("\n\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    avg_x40_zero = sum(times_x40_zero) / len(times_x40_zero)
    avg_x40_stored = sum(times_x40_stored) / len(times_x40_stored)
    avg_l10_zero = sum(times_l10_zero) / len(times_l10_zero)
    avg_l10_stored = sum(times_l10_stored) / len(times_l10_stored)

    print(f"\nX40 (X40_carpet.json):")
    print(f"  Auto-crop (0 trims):    {avg_x40_zero:.4f}s")
    print(f"  Pre-stored trims:       {avg_x40_stored:.4f}s")
    print(
        f"  Time saved:             {(avg_x40_zero - avg_x40_stored):.4f}s ({((avg_x40_zero - avg_x40_stored) / avg_x40_zero * 100):.1f}%)"
    )

    print(f"\nL10 (test.json):")
    print(f"  Auto-crop (0 trims):    {avg_l10_zero:.4f}s")
    print(f"  Pre-stored trims:       {avg_l10_stored:.4f}s")
    print(
        f"  Time saved:             {(avg_l10_zero - avg_l10_stored):.4f}s ({((avg_l10_zero - avg_l10_stored) / avg_l10_zero * 100):.1f}%)"
    )

    total_zero = avg_x40_zero + avg_l10_zero
    total_stored = avg_x40_stored + avg_l10_stored

    print(f"\nTotal (both floors):")
    print(f"  Auto-crop (0 trims):    {total_zero:.4f}s")
    print(f"  Pre-stored trims:       {total_stored:.4f}s")
    print(
        f"  Total time saved:       {(total_zero - total_stored):.4f}s ({((total_zero - total_stored) / total_zero * 100):.1f}%)"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
