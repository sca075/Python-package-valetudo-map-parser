"""
Test simple floor data management.
This test validates the simplified floor data structure and management.
"""

import sys
from pathlib import Path


# Add SCR to path
sys.path.insert(0, str(Path(__file__).parent.parent / "SCR"))

import pytest

from valetudo_map_parser.config.shared import CameraShared, CameraSharedManager
from valetudo_map_parser.config.types import TrimsData


def test_floor_data_structure():
    """Test that floors_data uses simple dict structure."""
    device_info = {
        "floors_data": {
            "flat": {  # HA floor ID
                "trim_up": 100,
                "trim_left": 50,
                "trim_down": 200,
                "trim_right": 150,
            },
            "cellar": {  # HA floor ID
                "trim_up": 1000,
                "trim_left": 2000,
                "trim_down": 3000,
                "trim_right": 4000,
            },
        },
        "current_floor": "flat",
    }

    # Initialize shared
    shared_manager = CameraSharedManager("test_floor", device_info)
    shared = shared_manager.get_instance()

    # Verify current floor trims are loaded
    assert shared.current_floor == "flat"
    assert shared.trims.trim_up == 100
    assert shared.trims.trim_left == 50
    assert shared.trims.trim_down == 200
    assert shared.trims.trim_right == 150

    print("✅ Floor data structure test passed!")


def test_floor_switching():
    """Test switching between floors updates shared.trims."""
    device_info = {
        "floors_data": {
            "flat": {
                "trim_up": 100,
                "trim_left": 50,
                "trim_down": 200,
                "trim_right": 150,
            },
            "cellar": {
                "trim_up": 10,
                "trim_left": 20,
                "trim_down": 30,
                "trim_right": 40,
            },
        },
        "current_floor": "flat",
    }

    # Initialize shared
    shared_manager = CameraSharedManager("test_floor", device_info)
    shared = shared_manager.get_instance()

    # Verify initial floor
    assert shared.current_floor == "flat"
    assert shared.trims.trim_up == 100

    # Switch to cellar floor
    device_info["current_floor"] = "cellar"
    shared_manager.update_shared_data(device_info)

    # Verify floor switched and trims updated
    assert shared.current_floor == "cellar"
    assert shared.trims.trim_up == 10
    assert shared.trims.trim_left == 20
    assert shared.trims.trim_down == 30
    assert shared.trims.trim_right == 40

    print("✅ Floor switching test passed!")


def test_init_with_cellar_data():
    """Test initializing a new instance with cellar floor data."""
    device_info = {
        "floors_data": {
            "flat": {
                "trim_up": 100,
                "trim_left": 50,
                "trim_down": 200,
                "trim_right": 150,
            },
            "cellar": {
                "trim_up": 10,
                "trim_left": 20,
                "trim_down": 30,
                "trim_right": 40,
            },
        },
        "current_floor": "cellar",  # Start with cellar
    }

    # Initialize NEW instance with cellar as current floor
    shared_manager = CameraSharedManager("test_cellar", device_info)
    shared = shared_manager.get_instance()

    # Verify cellar floor trims are loaded on init
    assert shared.current_floor == "cellar"
    assert shared.trims.trim_up == 10
    assert shared.trims.trim_left == 20
    assert shared.trims.trim_down == 30
    assert shared.trims.trim_right == 40

    print("✅ Init with cellar data test passed!")


def test_update_floor_trims():
    """Test updating trims for a specific floor."""
    device_info = {
        "floors_data": {
            "flat": {"trim_up": 0, "trim_left": 0, "trim_down": 0, "trim_right": 0}
        },
        "current_floor": "flat",
    }

    # Initialize shared
    shared_manager = CameraSharedManager("test_floor", device_info)
    shared = shared_manager.get_instance()

    # Simulate auto-crop calculating new trims
    new_trims = TrimsData(
        floor="flat", trim_up=2950, trim_left=2400, trim_down=3699, trim_right=3649
    )

    # Update shared.trims (this would be done by update_trims() after auto-crop)
    shared.trims = new_trims

    # Verify trims updated
    assert shared.trims.trim_up == 2950
    assert shared.trims.trim_left == 2400

    print("✅ Update floor trims test passed!")


def test_backward_compatibility():
    """Test that old trims_data format still works."""
    device_info = {
        "trims_data": {
            "floor": "floor_0",
            "trim_up": 100,
            "trim_left": 50,
            "trim_down": 200,
            "trim_right": 150,
        }
    }

    # Initialize shared
    shared_manager = CameraSharedManager("test_floor", device_info)
    shared = shared_manager.get_instance()

    # Verify old format still works
    assert shared.current_floor == "floor_0"
    assert shared.trims.trim_up == 100
    assert shared.trims.trim_left == 50

    print("✅ Backward compatibility test passed!")


def test_new_floor_methods():
    """Test the new add_floor, update_floor, remove_floor methods."""
    from valetudo_map_parser.config.types import FloorData

    device_info = {
        "floors_data": {
            "flat": {
                "trim_up": 100,
                "trim_left": 50,
                "trim_down": 200,
                "trim_right": 150,
            }
        },
        "current_floor": "flat",
    }

    # Initialize shared
    shared_manager = CameraSharedManager("test_new_methods", device_info)
    shared = shared_manager.get_instance()

    # Test add_floor
    new_floor_data = FloorData(
        trims=TrimsData(trim_up=10, trim_left=20, trim_down=30, trim_right=40),
        map_name="Cellar",
    )
    shared.add_floor("cellar", new_floor_data)
    assert "cellar" in shared.floors_trims
    assert shared.floors_trims["cellar"].trims.trim_up == 10
    assert shared.floors_trims["cellar"].map_name == "Cellar"

    # Test update_floor
    new_trims = TrimsData(trim_up=999, trim_left=888, trim_down=777, trim_right=666)
    shared.update_floor("cellar", new_trims)
    assert shared.floors_trims["cellar"].trims.trim_up == 999
    assert shared.floors_trims["cellar"].trims.trim_left == 888

    # Test FloorData.update_trims
    another_trims = TrimsData(trim_up=111, trim_left=222, trim_down=333, trim_right=444)
    shared.floors_trims["cellar"].update_trims(another_trims)
    assert shared.floors_trims["cellar"].trims.trim_up == 111

    # Test remove_floor
    shared.remove_floor("cellar")
    assert "cellar" not in shared.floors_trims

    # Test FloorData.clear
    shared.add_floor(
        "test",
        FloorData(
            trims=TrimsData(trim_up=100, trim_left=200, trim_down=300, trim_right=400),
            map_name="Test Floor",
        ),
    )
    shared.floors_trims["test"].clear()
    assert shared.floors_trims["test"].trims.trim_up == 0
    assert shared.floors_trims["test"].map_name == ""

    print("✅ New floor methods test passed!")


if __name__ == "__main__":
    test_floor_data_structure()
    test_floor_switching()
    test_init_with_cellar_data()
    test_update_floor_trims()
    test_backward_compatibility()
    test_new_floor_methods()
    print("\n✅ All floor data tests passed!")
