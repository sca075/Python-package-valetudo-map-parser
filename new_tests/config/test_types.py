"""Tests for config/types.py module."""

import asyncio
import json
import threading

import pytest

from valetudo_map_parser.config.types import (
    FloorData,
    RoomStore,
    SnapshotStore,
    TrimCropData,
    TrimsData,
    UserLanguageStore,
)


class TestTrimCropData:
    """Tests for TrimCropData dataclass."""

    def test_initialization(self):
        """Test TrimCropData initialization."""
        trim = TrimCropData(trim_left=10, trim_up=20, trim_right=30, trim_down=40)
        assert trim.trim_left == 10
        assert trim.trim_up == 20
        assert trim.trim_right == 30
        assert trim.trim_down == 40

    def test_to_dict(self):
        """Test conversion to dictionary."""
        trim = TrimCropData(trim_left=10, trim_up=20, trim_right=30, trim_down=40)
        result = trim.to_dict()
        assert result == {
            "trim_left": 10,
            "trim_up": 20,
            "trim_right": 30,
            "trim_down": 40,
        }

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"trim_left": 10, "trim_up": 20, "trim_right": 30, "trim_down": 40}
        trim = TrimCropData.from_dict(data)
        assert trim.trim_left == 10
        assert trim.trim_up == 20
        assert trim.trim_right == 30
        assert trim.trim_down == 40

    def test_to_list(self):
        """Test conversion to list."""
        trim = TrimCropData(trim_left=10, trim_up=20, trim_right=30, trim_down=40)
        result = trim.to_list()
        assert result == [10, 20, 30, 40]

    def test_from_list(self):
        """Test creation from list."""
        data = [10, 20, 30, 40]
        trim = TrimCropData.from_list(data)
        assert trim.trim_left == 10
        assert trim.trim_up == 20
        assert trim.trim_right == 30
        assert trim.trim_down == 40


class TestTrimsData:
    """Tests for TrimsData dataclass."""

    def test_initialization_defaults(self):
        """Test TrimsData initialization with defaults."""
        trims = TrimsData()
        assert trims.floor == ""
        assert trims.trim_up == 0
        assert trims.trim_left == 0
        assert trims.trim_down == 0
        assert trims.trim_right == 0

    def test_initialization_with_values(self):
        """Test TrimsData initialization with values."""
        trims = TrimsData(floor="floor_1", trim_up=10, trim_left=20, trim_down=30, trim_right=40)
        assert trims.floor == "floor_1"
        assert trims.trim_up == 10
        assert trims.trim_left == 20
        assert trims.trim_down == 30
        assert trims.trim_right == 40

    def test_to_json(self):
        """Test conversion to JSON string."""
        trims = TrimsData(floor="floor_1", trim_up=10, trim_left=20, trim_down=30, trim_right=40)
        json_str = trims.to_json()
        data = json.loads(json_str)
        assert data["floor"] == "floor_1"
        assert data["trim_up"] == 10
        assert data["trim_left"] == 20
        assert data["trim_down"] == 30
        assert data["trim_right"] == 40

    def test_from_json(self):
        """Test creation from JSON string."""
        json_str = '{"floor": "floor_1", "trim_up": 10, "trim_left": 20, "trim_down": 30, "trim_right": 40}'
        trims = TrimsData.from_json(json_str)
        assert trims.floor == "floor_1"
        assert trims.trim_up == 10
        assert trims.trim_left == 20
        assert trims.trim_down == 30
        assert trims.trim_right == 40

    def test_to_dict(self):
        """Test conversion to dictionary."""
        trims = TrimsData(floor="floor_1", trim_up=10, trim_left=20, trim_down=30, trim_right=40)
        result = trims.to_dict()
        assert result["floor"] == "floor_1"
        assert result["trim_up"] == 10

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"floor": "floor_1", "trim_up": 10, "trim_left": 20, "trim_down": 30, "trim_right": 40}
        trims = TrimsData.from_dict(data)
        assert trims.floor == "floor_1"
        assert trims.trim_up == 10

    def test_from_list(self):
        """Test creation from list."""
        crop_area = [10, 20, 30, 40]
        trims = TrimsData.from_list(crop_area, floor="floor_1")
        assert trims.trim_up == 10
        assert trims.trim_left == 20
        assert trims.trim_down == 30
        assert trims.trim_right == 40
        assert trims.floor == "floor_1"

    def test_clear(self):
        """Test clearing all trims."""
        trims = TrimsData(floor="floor_1", trim_up=10, trim_left=20, trim_down=30, trim_right=40)
        result = trims.clear()
        assert trims.floor == ""
        assert trims.trim_up == 0
        assert trims.trim_left == 0
        assert trims.trim_down == 0
        assert trims.trim_right == 0
        assert result["floor"] == ""


class TestFloorData:
    """Tests for FloorData dataclass."""

    def test_initialization(self):
        """Test FloorData initialization."""
        trims = TrimsData(floor="floor_1", trim_up=10, trim_left=20, trim_down=30, trim_right=40)
        floor_data = FloorData(trims=trims, map_name="Test Map")
        assert floor_data.trims == trims
        assert floor_data.map_name == "Test Map"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "trims": {"floor": "floor_1", "trim_up": 10, "trim_left": 20, "trim_down": 30, "trim_right": 40},
            "map_name": "Test Map",
        }
        floor_data = FloorData.from_dict(data)
        assert floor_data.trims.floor == "floor_1"
        assert floor_data.map_name == "Test Map"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        trims = TrimsData(floor="floor_1", trim_up=10, trim_left=20, trim_down=30, trim_right=40)
        floor_data = FloorData(trims=trims, map_name="Test Map")
        result = floor_data.to_dict()
        assert result["map_name"] == "Test Map"
        assert result["trims"]["floor"] == "floor_1"


class TestRoomStore:
    """Tests for RoomStore singleton class."""

    def test_singleton_behavior(self, vacuum_id, sample_room_data):
        """Test that RoomStore implements singleton pattern per vacuum_id."""
        store1 = RoomStore(vacuum_id, sample_room_data)
        store2 = RoomStore(vacuum_id)
        assert store1 is store2

    def test_different_vacuum_ids(self, sample_room_data):
        """Test that different vacuum IDs get different instances."""
        store1 = RoomStore("vacuum_1", sample_room_data)
        store2 = RoomStore("vacuum_2", sample_room_data)
        assert store1 is not store2

    def test_initialization_with_data(self, vacuum_id, sample_room_data):
        """Test initialization with room data."""
        store = RoomStore(vacuum_id, sample_room_data)
        assert store.vacuum_id == vacuum_id
        assert store.vacuums_data == sample_room_data
        assert store.rooms_count == 2

    def test_get_rooms(self, vacuum_id, sample_room_data):
        """Test getting all rooms data."""
        store = RoomStore(vacuum_id, sample_room_data)
        rooms = store.get_rooms()
        assert rooms == sample_room_data

    def test_set_rooms(self, vacuum_id, sample_room_data):
        """Test setting rooms data."""
        store = RoomStore(vacuum_id)
        store.set_rooms(sample_room_data)
        assert store.vacuums_data == sample_room_data
        assert store.rooms_count == 2

    def test_get_rooms_count(self, vacuum_id, sample_room_data):
        """Test getting room count."""
        store = RoomStore(vacuum_id, sample_room_data)
        assert store.get_rooms_count() == 2

    def test_get_rooms_count_empty(self, vacuum_id):
        """Test getting room count when no rooms."""
        store = RoomStore(vacuum_id, {})
        assert store.get_rooms_count() == 1  # DEFAULT_ROOMS

    def test_room_names_property(self, vacuum_id, sample_room_data):
        """Test room_names property returns correct format."""
        store = RoomStore(vacuum_id, sample_room_data)
        names = store.room_names
        assert "room_0_name" in names
        assert "room_1_name" in names
        assert "16: Living Room" in names.values()
        assert "17: Kitchen" in names.values()

    def test_room_names_max_16_rooms(self, vacuum_id):
        """Test that room_names supports maximum 16 rooms."""
        # Create 20 rooms
        rooms_data = {str(i): {"number": i, "outline": [], "name": f"Room {i}", "x": 0, "y": 0} for i in range(20)}
        store = RoomStore(vacuum_id, rooms_data)
        names = store.room_names
        # Should only have 16 rooms
        assert len(names) == 16

    def test_room_names_empty_data(self, vacuum_id):
        """Test room_names with empty data returns defaults."""
        store = RoomStore(vacuum_id, {})
        names = store.room_names
        assert isinstance(names, dict)
        assert len(names) > 0  # Should return DEFAULT_ROOMS_NAMES

    def test_get_all_instances(self, sample_room_data):
        """Test getting all RoomStore instances."""
        store1 = RoomStore("vacuum_1", sample_room_data)
        store2 = RoomStore("vacuum_2", sample_room_data)
        all_instances = RoomStore.get_all_instances()
        assert "vacuum_1" in all_instances
        assert "vacuum_2" in all_instances
        assert all_instances["vacuum_1"] is store1
        assert all_instances["vacuum_2"] is store2

    def test_thread_safety(self, sample_room_data):
        """Test thread-safe singleton creation."""
        instances = []

        def create_instance():
            store = RoomStore("thread_test", sample_room_data)
            instances.append(store)

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All instances should be the same
        assert all(inst is instances[0] for inst in instances)


class TestUserLanguageStore:
    """Tests for UserLanguageStore singleton class."""

    @pytest.mark.asyncio
    async def test_singleton_behavior(self):
        """Test that UserLanguageStore implements singleton pattern."""
        store1 = UserLanguageStore()
        store2 = UserLanguageStore()
        assert store1 is store2

    @pytest.mark.asyncio
    async def test_set_and_get_user_language(self):
        """Test setting and getting user language."""
        store = UserLanguageStore()
        await store.set_user_language("user_1", "en")
        language = await store.get_user_language("user_1")
        assert language == "en"

    @pytest.mark.asyncio
    async def test_get_nonexistent_user_language(self):
        """Test getting language for non-existent user returns empty string."""
        store = UserLanguageStore()
        language = await store.get_user_language("nonexistent_user")
        assert language == ""

    @pytest.mark.asyncio
    async def test_get_all_languages(self):
        """Test getting all user languages."""
        store = UserLanguageStore()
        store.user_languages.clear()  # Clear for clean test
        await store.set_user_language("user_1", "en")
        await store.set_user_language("user_2", "it")
        languages = await store.get_all_languages()
        assert "en" in languages
        assert "it" in languages

    @pytest.mark.asyncio
    async def test_get_all_languages_empty(self):
        """Test getting all languages when empty returns default."""
        store = UserLanguageStore()
        store.user_languages.clear()
        languages = await store.get_all_languages()
        assert languages == ["en"]

    @pytest.mark.asyncio
    async def test_update_user_language(self):
        """Test updating existing user language."""
        store = UserLanguageStore()
        await store.set_user_language("user_1", "en")
        await store.set_user_language("user_1", "it")
        language = await store.get_user_language("user_1")
        assert language == "it"


class TestSnapshotStore:
    """Tests for SnapshotStore singleton class."""

    @pytest.mark.asyncio
    async def test_singleton_behavior(self):
        """Test that SnapshotStore implements singleton pattern."""
        store1 = SnapshotStore()
        store2 = SnapshotStore()
        assert store1 is store2

    @pytest.mark.asyncio
    async def test_set_and_get_snapshot_save_data(self):
        """Test setting and getting snapshot save data."""
        store = SnapshotStore()
        await store.async_set_snapshot_save_data("vacuum_1", True)
        result = await store.async_get_snapshot_save_data("vacuum_1")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_nonexistent_snapshot_save_data(self):
        """Test getting snapshot data for non-existent vacuum returns False."""
        store = SnapshotStore()
        result = await store.async_get_snapshot_save_data("nonexistent_vacuum")
        assert result is False

    @pytest.mark.asyncio
    async def test_set_and_get_vacuum_json(self):
        """Test setting and getting vacuum JSON data."""
        store = SnapshotStore()
        test_json = {"test": "data", "value": 123}
        await store.async_set_vacuum_json("vacuum_1", test_json)
        result = await store.async_get_vacuum_json("vacuum_1")
        assert result == test_json

    @pytest.mark.asyncio
    async def test_get_nonexistent_vacuum_json(self):
        """Test getting JSON for non-existent vacuum returns empty dict."""
        store = SnapshotStore()
        result = await store.async_get_vacuum_json("nonexistent_vacuum")
        assert result == {}

    @pytest.mark.asyncio
    async def test_update_vacuum_json(self):
        """Test updating existing vacuum JSON data."""
        store = SnapshotStore()
        json1 = {"test": "data1"}
        json2 = {"test": "data2"}
        await store.async_set_vacuum_json("vacuum_1", json1)
        await store.async_set_vacuum_json("vacuum_1", json2)
        result = await store.async_get_vacuum_json("vacuum_1")
        assert result == json2

