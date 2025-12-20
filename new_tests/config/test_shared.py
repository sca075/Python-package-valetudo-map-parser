"""Tests for config/shared.py module."""

import pytest
from PIL import Image

from valetudo_map_parser.config.shared import CameraShared, CameraSharedManager
from valetudo_map_parser.config.types import CameraModes, TrimsData


class TestCameraShared:
    """Tests for CameraShared class."""

    def test_initialization(self, vacuum_id):
        """Test CameraShared initialization."""
        shared = CameraShared(vacuum_id)
        assert shared.file_name == vacuum_id
        assert shared.camera_mode == CameraModes.MAP_VIEW
        assert shared.frame_number == 0
        assert shared.destinations == []
        assert shared.rand256_active_zone == []
        assert shared.rand256_zone_coordinates == []
        assert shared.is_rand is False
        assert isinstance(shared.last_image, Image.Image)
        assert shared.new_image is None
        assert shared.binary_image is None

    def test_vacuum_bat_charged_not_docked(self, camera_shared):
        """Test vacuum_bat_charged when not docked."""
        camera_shared.vacuum_state = "cleaning"
        camera_shared.vacuum_battery = 50
        result = camera_shared.vacuum_bat_charged()
        assert result is False

    def test_vacuum_bat_charged_docked_charging(self, camera_shared):
        """Test vacuum_bat_charged when docked and charging."""
        camera_shared.vacuum_state = "docked"
        camera_shared.vacuum_battery = 50
        result = camera_shared.vacuum_bat_charged()
        assert result is True

    def test_vacuum_bat_charged_docked_full(self, camera_shared):
        """Test vacuum_bat_charged when docked and fully charged."""
        camera_shared.vacuum_state = "docked"
        camera_shared.vacuum_battery = 100
        camera_shared._battery_state = "charging_done"
        result = camera_shared.vacuum_bat_charged()
        assert result is False  # Not charging anymore

    def test_compose_obstacle_links_valid(self):
        """Test composing obstacle links with valid data."""
        obstacles = [
            {"label": "shoe", "points": [100, 200], "id": "obstacle_1"},
            {"label": "sock", "points": [150, 250], "id": "obstacle_2"},
        ]
        result = CameraShared._compose_obstacle_links("192.168.1.100", obstacles)
        assert len(result) == 2
        assert result[0]["label"] == "shoe"
        assert result[0]["point"] == [100, 200]
        assert "192.168.1.100" in result[0]["link"]
        assert "obstacle_1" in result[0]["link"]

    def test_compose_obstacle_links_no_id(self):
        """Test composing obstacle links without image ID."""
        obstacles = [{"label": "shoe", "points": [100, 200], "id": "None"}]
        result = CameraShared._compose_obstacle_links("192.168.1.100", obstacles)
        assert len(result) == 1
        assert "link" not in result[0]
        assert result[0]["label"] == "shoe"

    def test_compose_obstacle_links_empty(self):
        """Test composing obstacle links with empty data."""
        result = CameraShared._compose_obstacle_links("192.168.1.100", [])
        assert result is None

    def test_compose_obstacle_links_no_ip(self):
        """Test composing obstacle links without IP."""
        obstacles = [{"label": "shoe", "points": [100, 200], "id": "obstacle_1"}]
        result = CameraShared._compose_obstacle_links("", obstacles)
        assert result is None

    def test_update_user_colors(self, camera_shared):
        """Test updating user colors."""
        new_colors = {"wall": (255, 0, 0), "floor": (0, 255, 0)}
        camera_shared.update_user_colors(new_colors)
        assert camera_shared.user_colors == new_colors

    def test_get_user_colors(self, camera_shared):
        """Test getting user colors."""
        colors = camera_shared.get_user_colors()
        assert colors is not None

    def test_update_rooms_colors(self, camera_shared):
        """Test updating rooms colors."""
        new_colors = {"room_1": (255, 0, 0), "room_2": (0, 255, 0)}
        camera_shared.update_rooms_colors(new_colors)
        assert camera_shared.rooms_colors == new_colors

    def test_get_rooms_colors(self, camera_shared):
        """Test getting rooms colors."""
        colors = camera_shared.get_rooms_colors()
        assert colors is not None

    def test_reset_trims(self, camera_shared):
        """Test resetting trims to default."""
        camera_shared.trims = TrimsData(floor="floor_1", trim_up=10, trim_left=20, trim_down=30, trim_right=40)
        result = camera_shared.reset_trims()
        assert isinstance(result, TrimsData)
        assert camera_shared.trims.trim_up == 0
        assert camera_shared.trims.trim_left == 0

    @pytest.mark.asyncio
    async def test_batch_update(self, camera_shared):
        """Test batch updating attributes."""
        await camera_shared.batch_update(
            vacuum_battery=75, vacuum_state="cleaning", image_rotate=90, frame_number=5
        )
        assert camera_shared.vacuum_battery == 75
        assert camera_shared.vacuum_state == "cleaning"
        assert camera_shared.image_rotate == 90
        assert camera_shared.frame_number == 5

    @pytest.mark.asyncio
    async def test_batch_get(self, camera_shared):
        """Test batch getting attributes."""
        camera_shared.vacuum_battery = 80
        camera_shared.vacuum_state = "docked"
        camera_shared.frame_number = 10
        result = await camera_shared.batch_get("vacuum_battery", "vacuum_state", "frame_number")
        assert result["vacuum_battery"] == 80
        assert result["vacuum_state"] == "docked"
        assert result["frame_number"] == 10

    def test_generate_attributes(self, camera_shared):
        """Test generating attributes dictionary."""
        camera_shared.vacuum_battery = 90
        camera_shared.vacuum_state = "docked"
        attrs = camera_shared.generate_attributes()
        assert isinstance(attrs, dict)
        # Should contain various attributes
        assert len(attrs) > 0


class TestCameraSharedManager:
    """Tests for CameraSharedManager singleton class."""

    def test_singleton_behavior(self, vacuum_id, device_info):
        """Test that CameraSharedManager creates instances (not strict singleton)."""
        manager1 = CameraSharedManager(vacuum_id, device_info)
        manager2 = CameraSharedManager(vacuum_id, device_info)
        # CameraSharedManager doesn't implement strict singleton pattern
        # Each call creates a new manager instance
        assert manager1 is not manager2
        # But they should both return CameraShared instances
        shared1 = manager1.get_instance()
        shared2 = manager2.get_instance()
        assert isinstance(shared1, CameraShared)
        assert isinstance(shared2, CameraShared)

    def test_different_vacuum_ids(self, device_info):
        """Test that different vacuum IDs get different managers."""
        manager1 = CameraSharedManager("vacuum_1", device_info)
        manager2 = CameraSharedManager("vacuum_2", device_info)
        assert manager1 is not manager2

    def test_get_instance(self, vacuum_id, device_info):
        """Test getting CameraShared instance from manager."""
        manager = CameraSharedManager(vacuum_id, device_info)
        shared = manager.get_instance()
        assert isinstance(shared, CameraShared)
        assert shared.file_name == vacuum_id

