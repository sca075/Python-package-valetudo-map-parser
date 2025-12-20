"""
Test suite for RoomStore singleton behavior.

This test file validates:
1. Singleton pattern per vacuum_id
2. Instance caching and reuse
3. Data persistence and updates
4. Room counting and naming
5. Edge cases (empty data, max rooms, etc.)
6. Type safety with Dict[str, RoomProperty]

The RoomStore uses proper type hints without runtime validation overhead.
"""

import importlib.util
import logging
import sys
from pathlib import Path

# Add SCR/valetudo_map_parser to path so relative imports work
valetudo_path = Path(__file__).parent.parent / "SCR" / "valetudo_map_parser"
if str(valetudo_path.parent) not in sys.path:
    sys.path.insert(0, str(valetudo_path.parent))

# Load const module first
const_path = valetudo_path / "const.py"
const_spec = importlib.util.spec_from_file_location("valetudo_map_parser.const", const_path)
const_module = importlib.util.module_from_spec(const_spec)
sys.modules["valetudo_map_parser.const"] = const_module
const_spec.loader.exec_module(const_module)

# Now load types module
types_path = valetudo_path / "config" / "types.py"
spec = importlib.util.spec_from_file_location("valetudo_map_parser.config.types", types_path)
types = importlib.util.module_from_spec(spec)
sys.modules["valetudo_map_parser.config.types"] = types
spec.loader.exec_module(types)

RoomStore = types.RoomStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_LOGGER = logging.getLogger(__name__)


def test_room_store_singleton():
    """Test that RoomStore maintains singleton per vacuum_id."""
    _LOGGER.info("=" * 60)
    _LOGGER.info("Testing RoomStore Singleton Behavior")
    _LOGGER.info("=" * 60)

    # Test 1: Create first instance with initial data
    _LOGGER.info("\n1. Creating first instance for vacuum_1")
    initial_data = {"1": {"name": "Living Room"}, "2": {"name": "Kitchen"}}
    store1 = RoomStore("vacuum_1", initial_data)
    _LOGGER.info(f"   Instance ID: {id(store1)}")
    _LOGGER.info(f"   Rooms: {store1.get_rooms()}")
    _LOGGER.info(f"   Room count: {store1.rooms_count}")

    # Test 2: Get same instance without new data (should return cached)
    _LOGGER.info("\n2. Getting same instance for vacuum_1 (no new data)")
    store2 = RoomStore("vacuum_1")
    _LOGGER.info(f"   Instance ID: {id(store2)}")
    _LOGGER.info(f"   Rooms: {store2.get_rooms()}")
    _LOGGER.info(f"   Same instance? {store1 is store2}")
    assert store1 is store2, "Should return the same instance"
    assert store2.get_rooms() == initial_data, "Should preserve initial data"

    # Test 3: Update existing instance with new data
    _LOGGER.info("\n3. Updating vacuum_1 with new data")
    updated_data = {
        "1": {"name": "Living Room"},
        "2": {"name": "Kitchen"},
        "3": {"name": "Bedroom"},
    }
    store3 = RoomStore("vacuum_1", updated_data)
    _LOGGER.info(f"   Instance ID: {id(store3)}")
    _LOGGER.info(f"   Rooms: {store3.get_rooms()}")
    _LOGGER.info(f"   Room count: {store3.rooms_count}")
    _LOGGER.info(f"   Same instance? {store1 is store3}")
    assert store1 is store3, "Should return the same instance"
    assert store3.get_rooms() == updated_data, "Should update with new data"
    assert store3.rooms_count == 3, "Room count should be updated"

    # Test 4: Create different instance for different vacuum
    _LOGGER.info("\n4. Creating instance for vacuum_2")
    vacuum2_data = {"10": {"name": "Office"}}
    store4 = RoomStore("vacuum_2", vacuum2_data)
    _LOGGER.info(f"   Instance ID: {id(store4)}")
    _LOGGER.info(f"   Rooms: {store4.get_rooms()}")
    _LOGGER.info(f"   Different instance? {store1 is not store4}")
    assert store1 is not store4, "Should be different instance for different vacuum"
    assert store4.get_rooms() == vacuum2_data, "Should have its own data"

    # Test 5: Verify vacuum_1 data is still intact
    _LOGGER.info("\n5. Verifying vacuum_1 data is still intact")
    store5 = RoomStore("vacuum_1")
    _LOGGER.info(f"   Rooms: {store5.get_rooms()}")
    assert store5.get_rooms() == updated_data, "vacuum_1 data should be unchanged"

    # Test 6: Test set_rooms method
    _LOGGER.info("\n6. Testing set_rooms method")
    new_data = {"1": {"name": "Updated Living Room"}}
    store1.set_rooms(new_data)
    _LOGGER.info(f"   Rooms after set_rooms: {store1.get_rooms()}")
    assert store1.get_rooms() == new_data, "set_rooms should update data"

    # Test 7: Test room_names property
    _LOGGER.info("\n7. Testing room_names property")
    test_data = {
        "16": {"name": "Living Room"},
        "17": {"name": "Kitchen"},
        "18": {"name": "Bedroom"},
    }
    store6 = RoomStore("vacuum_3", test_data)
    room_names = store6.room_names
    _LOGGER.info(f"   Room names: {room_names}")
    assert "room_0_name" in room_names, "Should have room_0_name"
    assert "16: Living Room" in room_names["room_0_name"], "Should format correctly"

    # Test 8: Test get_all_instances
    _LOGGER.info("\n8. Testing get_all_instances")
    all_instances = RoomStore.get_all_instances()
    _LOGGER.info(f"   Total instances: {len(all_instances)}")
    _LOGGER.info(f"   Vacuum IDs: {list(all_instances.keys())}")
    assert len(all_instances) >= 3, "Should have at least 3 instances"
    assert "vacuum_1" in all_instances, "Should contain vacuum_1"
    assert "vacuum_2" in all_instances, "Should contain vacuum_2"
    assert "vacuum_3" in all_instances, "Should contain vacuum_3"

    _LOGGER.info("\n" + "=" * 60)
    _LOGGER.info("✅ All singleton tests passed!")
    _LOGGER.info("=" * 60)


def test_room_store_edge_cases():
    """Test edge cases and error conditions."""
    _LOGGER.info("\n" + "=" * 60)
    _LOGGER.info("Testing RoomStore Edge Cases and Error Conditions")
    _LOGGER.info("=" * 60)

    # Test 1: Create instance with no data (None)
    _LOGGER.info("\n1. Creating instance with no data (None)")
    store1 = RoomStore("vacuum_no_data", None)
    _LOGGER.info(f"   Rooms: {store1.get_rooms()}")
    _LOGGER.info(f"   Room count: {store1.rooms_count}")
    assert store1.get_rooms() == {}, "Should have empty dict"
    assert store1.rooms_count == 1, "Should default to 1 room"

    # Test 2: Create instance with empty dict
    _LOGGER.info("\n2. Creating instance with empty dict")
    store2 = RoomStore("vacuum_empty", {})
    _LOGGER.info(f"   Rooms: {store2.get_rooms()}")
    _LOGGER.info(f"   Room count: {store2.rooms_count}")
    assert store2.get_rooms() == {}, "Should have empty dict"
    assert store2.rooms_count == 1, "Should default to 1 room"

    # Test 3: Vacuum that doesn't support rooms (empty data)
    _LOGGER.info("\n3. Vacuum without room support")
    store3 = RoomStore("vacuum_no_rooms")
    _LOGGER.info(f"   Rooms: {store3.get_rooms()}")
    _LOGGER.info(f"   Room count: {store3.rooms_count}")
    _LOGGER.info(f"   Room names: {store3.room_names}")
    assert store3.get_rooms() == {}, "Should have empty dict"
    assert store3.rooms_count == 1, "Should default to 1 room"
    assert len(store3.room_names) == 15, "Should return DEFAULT_ROOMS_NAMES (15 rooms)"
    assert "room_0_name" in store3.room_names, "Should have room_0_name"
    assert store3.room_names["room_0_name"] == "Room 1", "Should use default name"

    # Test 4: Update from empty to having rooms
    _LOGGER.info("\n4. Updating from no rooms to having rooms")
    store3_updated = RoomStore("vacuum_no_rooms", {"1": {"name": "New Room"}})
    _LOGGER.info(f"   Rooms: {store3_updated.get_rooms()}")
    _LOGGER.info(f"   Room count: {store3_updated.rooms_count}")
    _LOGGER.info(f"   Same instance? {store3 is store3_updated}")
    assert store3 is store3_updated, "Should be same instance"
    assert store3_updated.rooms_count == 1, "Should have 1 room now"
    assert store3_updated.get_rooms() == {"1": {"name": "New Room"}}, (
        "Should update data"
    )

    # Test 5: Set rooms to empty (simulate rooms removed)
    _LOGGER.info("\n5. Setting rooms to empty (rooms removed)")
    store3.set_rooms({})
    _LOGGER.info(f"   Rooms: {store3.get_rooms()}")
    _LOGGER.info(f"   Room count: {store3.rooms_count}")
    assert store3.get_rooms() == {}, "Should be empty"
    assert store3.rooms_count == 1, "Should default to 1"

    # Test 6: Maximum rooms (16 rooms)
    _LOGGER.info("\n6. Testing maximum rooms (16)")
    max_rooms_data = {str(i): {"name": f"Room {i}"} for i in range(1, 17)}
    store4 = RoomStore("vacuum_max_rooms", max_rooms_data)
    _LOGGER.info(f"   Room count: {store4.rooms_count}")
    _LOGGER.info(f"   Room names count: {len(store4.room_names)}")
    assert store4.rooms_count == 16, "Should have 16 rooms"
    assert len(store4.room_names) == 16, "Should have 16 room names"

    # Test 7: More than 16 rooms (should only process first 16)
    _LOGGER.info("\n7. Testing more than 16 rooms (should cap at 16)")
    too_many_rooms = {str(i): {"name": f"Room {i}"} for i in range(1, 21)}
    store5 = RoomStore("vacuum_too_many", too_many_rooms)
    _LOGGER.info(f"   Room count: {store5.rooms_count}")
    _LOGGER.info(f"   Room names count: {len(store5.room_names)}")
    assert store5.rooms_count == 20, "Room count should be 20"
    assert len(store5.room_names) == 16, "Room names should cap at 16"

    # Test 8: Room data without name field
    _LOGGER.info("\n8. Testing room data without name field")
    no_name_data = {"5": {}, "10": {"other_field": "value"}}
    store6 = RoomStore("vacuum_no_names", no_name_data)
    room_names = store6.room_names
    _LOGGER.info(f"   Room names: {room_names}")
    assert "room_0_name" in room_names, "Should have room_0_name"
    assert "5: Room 5" in room_names["room_0_name"], "Should use default name"

    # Test 9: Type checking (no runtime validation - relies on type hints)
    _LOGGER.info("\n9. Type safety with proper types")
    _LOGGER.info(
        "   Note: Invalid types should be caught by type checkers (mypy, pylint)"
    )
    _LOGGER.info("   No runtime validation overhead - relying on static type checking")
    _LOGGER.info("   ✓ Type hints enforce Dict[str, RoomProperty]")

    # Test 10: Accessing room_names on empty store
    _LOGGER.info("\n10. Testing room_names property on empty store")
    empty_store = RoomStore("vacuum_empty_names", {})
    room_names = empty_store.room_names
    _LOGGER.info(f"   Room names: {room_names}")
    assert len(room_names) == 15, "Should return DEFAULT_ROOMS_NAMES (15 rooms)"
    assert room_names["room_0_name"] == "Room 1", "Should use default names"

    # Test 11: Floor attribute (should be None by default)
    _LOGGER.info("\n11. Testing floor attribute")
    store8 = RoomStore("vacuum_floor_test")
    _LOGGER.info(f"   Floor: {store8.floor}")
    assert store8.floor is None, "Floor should be None by default"
    store8.floor = "ground_floor"
    _LOGGER.info(f"   Floor after setting: {store8.floor}")
    assert store8.floor == "ground_floor", "Floor should be updated"

    # Test 12: Proper typing with RoomProperty
    _LOGGER.info("\n12. Testing proper typed room data")
    store9 = RoomStore("vacuum_typed", {"1": {"name": "Typed Room"}})
    _LOGGER.info(f"   Rooms: {store9.get_rooms()}")
    _LOGGER.info(f"   Type: Dict[str, RoomProperty]")
    assert store9.get_rooms() == {"1": {"name": "Typed Room"}}, (
        "Should store typed data"
    )
    _LOGGER.info("   ✓ Proper type hints without runtime overhead")

    _LOGGER.info("\n" + "=" * 60)
    _LOGGER.info("✅ All edge case tests passed!")
    _LOGGER.info("=" * 60)


if __name__ == "__main__":
    test_room_store_singleton()
    test_room_store_edge_cases()
