"""Pytest configuration and fixtures for valetudo_map_parser tests."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from PIL import Image

# Add SCR directory to path to import from local source instead of installed package
sys.path.insert(0, str(Path(__file__).parent.parent / "SCR"))

from valetudo_map_parser.config.shared import CameraShared, CameraSharedManager
from valetudo_map_parser.config.types import RoomProperty, RoomStore


# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / "tests"
HYPFER_JSON_SAMPLES = [
    "test.json",
    "glossyhardtofindnarwhal.json",
    "l10_carpet.json",
]
RAND256_BIN_SAMPLES = [
    "map_data_20250728_185945.bin",
    "map_data_20250728_193950.bin",
    "map_data_20250729_084141.bin",
]


@pytest.fixture
def test_data_dir():
    """Return the test data directory path."""
    return TEST_DATA_DIR


@pytest.fixture
def hypfer_json_path(test_data_dir):
    """Return path to a Hypfer JSON test file."""
    return test_data_dir / "test.json"


@pytest.fixture
def hypfer_json_data(hypfer_json_path):
    """Load and return Hypfer JSON test data."""
    with open(hypfer_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def rand256_bin_path(test_data_dir):
    """Return path to a Rand256 binary test file."""
    return test_data_dir / "map_data_20250728_185945.bin"


@pytest.fixture
def rand256_bin_data(rand256_bin_path):
    """Load and return Rand256 binary test data."""
    with open(rand256_bin_path, "rb") as f:
        return f.read()


@pytest.fixture(params=HYPFER_JSON_SAMPLES)
def all_hypfer_json_files(request, test_data_dir):
    """Parametrized fixture providing all Hypfer JSON test files."""
    json_path = test_data_dir / request.param
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return request.param, json.load(f)
    pytest.skip(f"Test file {request.param} not found")


@pytest.fixture(params=RAND256_BIN_SAMPLES)
def all_rand256_bin_files(request, test_data_dir):
    """Parametrized fixture providing all Rand256 binary test files."""
    bin_path = test_data_dir / request.param
    if bin_path.exists():
        with open(bin_path, "rb") as f:
            return request.param, f.read()
    pytest.skip(f"Test file {request.param} not found")


@pytest.fixture
def device_info():
    """Return sample device info dictionary."""
    return {
        "identifiers": {("mqtt_vacuum_camera", "test_vacuum")},
        "name": "Test Vacuum",
        "manufacturer": "Valetudo",
        "model": "Test Model",
    }


@pytest.fixture
def vacuum_id():
    """Return a test vacuum ID."""
    return "test_vacuum_001"


@pytest.fixture
def camera_shared(vacuum_id, device_info):
    """Create and return a CameraShared instance."""
    manager = CameraSharedManager(vacuum_id, device_info)
    return manager.get_instance()


@pytest.fixture
def sample_room_data():
    """Return sample room data for testing."""
    return {
        "16": {
            "number": 16,
            "outline": [(100, 100), (200, 100), (200, 200), (100, 200)],
            "name": "Living Room",
            "x": 150,
            "y": 150,
        },
        "17": {
            "number": 17,
            "outline": [(300, 100), (400, 100), (400, 200), (300, 200)],
            "name": "Kitchen",
            "x": 350,
            "y": 150,
        },
    }


@pytest.fixture
def room_store(vacuum_id, sample_room_data):
    """Create and return a RoomStore instance."""
    return RoomStore(vacuum_id, sample_room_data)


@pytest.fixture
def test_image():
    """Create and return a test PIL Image."""
    img = Image.new("RGBA", (800, 600), (255, 255, 255, 255))
    yield img
    img.close()


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Clean up singleton instances after each test."""
    yield
    # Clean up RoomStore instances
    RoomStore._instances.clear()

