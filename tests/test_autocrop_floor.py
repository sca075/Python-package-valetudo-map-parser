"""Test auto crop and floor data functionality."""

from __future__ import annotations

import json
import logging
import os
import sys


# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import ImageDraw, ImageFont

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


def draw_calibration_overlay(image, calibration_points, rotation):
    """Draw calibration points and rectangle overlay on the image."""
    # Create a copy to draw on
    img_with_overlay = image.copy()
    draw = ImageDraw.Draw(img_with_overlay)

    # Extract map and vacuum points
    map_points = [p["map"] for p in calibration_points]
    vacuum_points = [p["vacuum"] for p in calibration_points]

    # Draw the calibration rectangle on the image
    # Connect the map points to show the calibration area
    for i in range(4):
        p1 = map_points[i]
        p2 = map_points[(i + 1) % 4]
        draw.line([(p1["x"], p1["y"]), (p2["x"], p2["y"])], fill="red", width=3)

    # Draw calibration points with labels
    colors = ["red", "cyan", "blue", "yellow"]
    for i, (mp, vp, color) in enumerate(zip(map_points, vacuum_points, colors)):
        # Draw circle at map position
        x, y = mp["x"], mp["y"]
        radius = 8
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline="black",
            width=2,
        )

        # Draw label with vacuum coordinates
        label = f"P{i}\nV:({vp['x']},{vp['y']})\nM:({mp['x']},{mp['y']})"
        # Position label outside the point
        label_x = x + 15 if x < image.width / 2 else x - 15
        label_y = y + 15 if y < image.height / 2 else y - 15

        # Draw text background
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((label_x, label_y), label, font=font)
        draw.rectangle(bbox, fill="white", outline="black")
        draw.text((label_x, label_y), label, fill="black", font=font)

    # Add title
    title = f"Rotation {rotation}° - Calibration Overlay"
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        title_font = ImageFont.load_default()

    draw.text((10, 10), title, fill="red", font=title_font)

    return img_with_overlay


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
    cal_points_270 = (
        shared.attr_calibration_points.copy()
    )  # Save calibration points immediately
    print(f"Trims {shared.file_name} image 1: {shared.trims.to_dict()}")
    # shared.reset_trims()

    # print(f"After Trims reset of {shared.file_name} image 1: {shared.trims}")
    print(f"Trims floor data image 1: {shared.floors_trims}")
    print(f"Robot position (vacuum coords): {shared.current_room}")
    print(f"Calibration points: {shared.attr_calibration_points}")
    print(f"Current imagesize: {image1.size}")

    # Second image generation
    shared.image_rotate = 0
    print(f"\n--- Generating Image 2 ---{shared.image_rotate}")

    img2, data2 = asyncio.run(handler.async_get_image(test_data, bytes_format=False))
    image2 = img2.copy() if img2 else None  # Copy to prevent it from being closed
    cal_points_0 = (
        shared.attr_calibration_points.copy()
    )  # Save calibration points immediately
    print(f"Trims after image 2: {shared.trims.to_dict()}")
    print(f"Robot position (vacuum coords): {shared.current_room}")
    print(f"Calibration points: {shared.attr_calibration_points}")
    print(f"Current imagesize: {image2.size}")

    # Third image generation
    shared.image_rotate = 90
    print(f"\n--- Generating Image 3 ---{shared.image_rotate}")

    img3, data3 = asyncio.run(handler.async_get_image(test_data, bytes_format=False))
    image3 = img3.copy() if img3 else None  # Copy to prevent it from being closed
    cal_points_90 = (
        shared.attr_calibration_points.copy()
    )  # Save calibration points immediately
    print(f"Trims after image 3: {shared.trims.to_dict()}")
    print(f"Robot position (vacuum coords): {shared.current_room}")
    print(f"Calibration points: {shared.attr_calibration_points}")
    print(f"Current imagesize: {image3.size}")

    # Fourth image generation
    shared.image_rotate = 180
    print(f"\n--- Generating Image 4 ---{shared.image_rotate}")
    img4, data4 = asyncio.run(handler.async_get_image(test_data, bytes_format=False))
    image4 = img4.copy() if img4 else None  # Copy to prevent it from being closed
    cal_points_180 = (
        shared.attr_calibration_points.copy()
    )  # Save calibration points immediately
    print(f"Trims after image 4: {shared.trims.to_dict()}")
    print(f"Robot position (vacuum coords): {shared.current_room}")
    print(f"Calibration points: {shared.attr_calibration_points}")
    print(f"Current imagesize: {image4.size}")
    # Save and display images with calibration overlay
    if image1:
        image1.save("/tmp/autocrop_rotation_270.png")
        print(f"✅ Saved Image 1 (270°) to /tmp/autocrop_rotation_270.png")
        # Use the calibration points saved immediately after generation
        overlay1 = draw_calibration_overlay(image1, cal_points_270, 270)
        overlay1.save("/tmp/autocrop_rotation_270_overlay.png")
        print(f"✅ Saved Image 1 overlay to /tmp/autocrop_rotation_270_overlay.png")
        overlay1.show()

    if image2:
        image2.save("/tmp/autocrop_rotation_0.png")
        print(f"✅ Saved Image 2 (0°) to /tmp/autocrop_rotation_0.png")
        # Use the calibration points saved immediately after generation
        overlay2 = draw_calibration_overlay(image2, cal_points_0, 0)
        overlay2.save("/tmp/autocrop_rotation_0_overlay.png")
        print(f"✅ Saved Image 2 overlay to /tmp/autocrop_rotation_0_overlay.png")
        overlay2.show()

    if image3:
        image3.save("/tmp/autocrop_rotation_90.png")
        print(f"✅ Saved Image 3 (90°) to /tmp/autocrop_rotation_90.png")
        # Use the calibration points saved immediately after generation
        overlay3 = draw_calibration_overlay(image3, cal_points_90, 90)
        overlay3.save("/tmp/autocrop_rotation_90_overlay.png")
        print(f"✅ Saved Image 3 overlay to /tmp/autocrop_rotation_90_overlay.png")
        overlay3.show()

    if image4:
        image4.save("/tmp/autocrop_rotation_180.png")
        print(f"✅ Saved Image 4 (180°) to /tmp/autocrop_rotation_180.png")
        # Use the calibration points saved immediately after generation
        overlay4 = draw_calibration_overlay(image4, cal_points_180, 180)
        overlay4.save("/tmp/autocrop_rotation_180_overlay.png")
        print(f"✅ Saved Image 4 overlay to /tmp/autocrop_rotation_180_overlay.png")
        overlay4.show()


if __name__ == "__main__":
    test_autocrop_and_floor_data()
