"""Regression tests for floors_data-based crop changes at rotations 0 and 270."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SCR.valetudo_map_parser.config.colors import ColorsManagement
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.hypfer_handler import HypferMapImageHandler


TEST_FILE = Path(__file__).with_name("X40_carpet.json")
BASE_TRIMS = {
    "floor": "floor_0",
    "trim_up": 2945,
    "trim_left": 2400,
    "trim_down": 3704,
    "trim_right": 3654,
}


def _load_test_json() -> dict:
    with TEST_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def _build_device_info(rotation: int) -> dict:
    return {
        "vacuum_connection_string": "mqtt://localhost:1883",
        "vacuum_identifiers": "test_vacuum",
        "vacuum_model": "Dreame.vacuum.r2228o",
        "rotate_image": rotation,
        "auto_zoom": False,
        "margins": 100,
        "current_floor": "floor_0",
        "floors_data": {
            "floor_0": {
                "trims": copy.deepcopy(BASE_TRIMS),
                "map_name": "floor_0",
                "rotation": rotation,
            }
        },
    }


def _render_image(file_name: str, device_info: dict, test_json: dict):
    shared_manager = CameraSharedManager(file_name, device_info)
    shared = shared_manager.get_instance()
    shared.vacuum_state = "docked"
    shared.vacuum_connection = True
    shared.vacuum_battery = 100

    colors = ColorsManagement(shared)
    colors.set_initial_colours(device_info)

    handler = HypferMapImageHandler(shared)
    loop = asyncio.new_event_loop()
    try:
        image, _ = loop.run_until_complete(
            handler.async_get_image(test_json, bytes_format=False)
        )
    finally:
        loop.close()
    assert image is not None
    return image.copy(), shared


def _run_trim_up_plus_500_comparison(rotation: int) -> dict:
    test_json = _load_test_json()
    base_device_info = _build_device_info(rotation)
    raised_trim_device_info = copy.deepcopy(base_device_info)
    raised_trim_device_info["floors_data"]["floor_0"]["trims"]["trim_up"] += 500

    assert "trims_data" not in base_device_info
    assert "trims_data" not in raised_trim_device_info

    base_image, base_shared = _render_image(
        "test_rotate_crop_base", base_device_info, test_json
    )
    raised_image, raised_shared = _render_image(
        "test_rotate_crop_raised", raised_trim_device_info, test_json
    )

    return {
        "rotation": rotation,
        "base_image": base_image,
        "raised_image": raised_image,
        "base_shared": base_shared,
        "raised_shared": raised_shared,
    }


def _close_comparison_images(comparison: dict) -> None:
    comparison["base_image"].close()
    comparison["raised_image"].close()


def show_three_trim_reference_images() -> None:
    """Show 0° base, 0° trim_up +500, and 270° trim_up +500 for visual comparison."""
    comparison_0 = _run_trim_up_plus_500_comparison(rotation=0)
    comparison_270 = _run_trim_up_plus_500_comparison(rotation=270)
    output_dir = Path(tempfile.gettempdir()) / "valetudo_rotate_crop"
    output_dir.mkdir(exist_ok=True)

    base_0_path = output_dir / "rotate_0_base.png"
    raised_0_path = output_dir / "rotate_0_trim_up_plus_500.png"
    raised_270_path = output_dir / "rotate_270_trim_up_plus_500.png"

    comparison_0["base_image"].save(base_0_path)
    comparison_0["raised_image"].save(raised_0_path)
    comparison_270["raised_image"].save(raised_270_path)

    comparison_0["base_image"].show(title="rotation 0 base")
    comparison_0["raised_image"].show(title="rotation 0 trim_up +500")
    comparison_270["raised_image"].show(title="rotation 270 trim_up +500")

    print(f"Saved rotation 0 base image to {base_0_path}")
    print(f"Saved rotation 0 trim_up +500 image to {raised_0_path}")
    print(f"Saved rotation 270 trim_up +500 image to {raised_270_path}")

    _close_comparison_images(comparison_0)
    _close_comparison_images(comparison_270)


def test_rotation_zero_with_floor_data_and_trim_up_plus_500(tmp_path):
    """The same floor_data at rotation 0 must crop higher when trim_up is increased."""
    comparison = _run_trim_up_plus_500_comparison(rotation=0)
    base_image = comparison["base_image"]
    raised_image = comparison["raised_image"]
    base_shared = comparison["base_shared"]
    raised_shared = comparison["raised_shared"]

    base_image.save(tmp_path / "rotate_crop_base.png")
    raised_image.save(tmp_path / "rotate_crop_trim_up_plus_500.png")

    assert base_shared.current_floor == "floor_0"
    assert raised_shared.current_floor == "floor_0"
    assert base_shared.image_rotate == 0
    assert raised_shared.image_rotate == 0

    assert base_shared.floors_trims["floor_0"].rotation == 0
    assert raised_shared.floors_trims["floor_0"].rotation == 0

    assert base_shared.trims.trim_left == BASE_TRIMS["trim_left"]
    assert base_shared.trims.trim_up == BASE_TRIMS["trim_up"]
    assert base_shared.trims.trim_right == BASE_TRIMS["trim_right"]
    assert base_shared.trims.trim_down == BASE_TRIMS["trim_down"]

    assert raised_shared.trims.trim_left == BASE_TRIMS["trim_left"]
    assert raised_shared.trims.trim_up == BASE_TRIMS["trim_up"] + 500
    assert raised_shared.trims.trim_right == BASE_TRIMS["trim_right"]
    assert raised_shared.trims.trim_down == BASE_TRIMS["trim_down"]

    assert base_image.width == raised_image.width
    assert base_image.height > raised_image.height
    assert base_image.height - raised_image.height == 500

    base_calibration = base_shared.attr_calibration_points
    raised_calibration = raised_shared.attr_calibration_points

    assert base_calibration is not None
    assert raised_calibration is not None
    assert raised_calibration[0]["vacuum"]["y"] - base_calibration[0]["vacuum"]["y"] == 500
    assert raised_calibration[1]["vacuum"]["y"] - base_calibration[1]["vacuum"]["y"] == 500

    _close_comparison_images(comparison)


def test_rotation_270_with_floor_data_and_trim_up_plus_500(tmp_path):
    """At rotation 270, trim_up +500 must shrink the displayed width, not the height."""
    comparison = _run_trim_up_plus_500_comparison(rotation=270)
    base_image = comparison["base_image"]
    raised_image = comparison["raised_image"]
    base_shared = comparison["base_shared"]
    raised_shared = comparison["raised_shared"]

    base_image.save(tmp_path / "rotate_crop_270_base.png")
    raised_image.save(tmp_path / "rotate_crop_270_trim_up_plus_500.png")

    assert base_shared.current_floor == "floor_0"
    assert raised_shared.current_floor == "floor_0"
    assert base_shared.image_rotate == 270
    assert raised_shared.image_rotate == 270

    assert base_shared.floors_trims["floor_0"].rotation == 270
    assert raised_shared.floors_trims["floor_0"].rotation == 270

    assert base_shared.trims.trim_up == BASE_TRIMS["trim_up"]
    assert raised_shared.trims.trim_up == BASE_TRIMS["trim_up"] + 500

    assert base_image.height == raised_image.height
    assert base_image.width > raised_image.width
    assert base_image.width - raised_image.width == 500

    base_calibration = base_shared.attr_calibration_points
    raised_calibration = raised_shared.attr_calibration_points

    assert base_calibration is not None
    assert raised_calibration is not None
    assert raised_calibration[1]["vacuum"]["y"] - base_calibration[1]["vacuum"]["y"] == 500
    assert raised_calibration[2]["vacuum"]["y"] - base_calibration[2]["vacuum"]["y"] == 500

    _close_comparison_images(comparison)


if __name__ == "__main__":
    show_three_trim_reference_images()