"""Test FloorData and multi-floor support - Standalone version."""
from dataclasses import asdict, dataclass
from typing import List, Optional


# Replicate TrimsData for testing
@dataclass
class TrimsData:
    floor: str = ""
    trim_up: int = 0
    trim_left: int = 0
    trim_down: int = 0
    trim_right: int = 0

    @classmethod
    def from_list(cls, crop_area: List[int], floor: Optional[str] = None):
        return cls(
            trim_up=crop_area[0],
            trim_left=crop_area[1],
            trim_down=crop_area[2],
            trim_right=crop_area[3],
            floor=floor or "",
        )

    def to_dict(self):
        return asdict(self)


# Replicate FloorData for testing
@dataclass
class FloorData:
    trims: TrimsData
    map_name: str = ""

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            trims=TrimsData(**data.get("trims", {})),
            map_name=data.get("map_name", ""),
        )

    def to_dict(self):
        return {"trims": self.trims.to_dict(), "map_name": self.map_name}


# Replicate CameraShared for testing
class CameraShared:
    def __init__(self, file_name):
        self.file_name = file_name
        self.trims = TrimsData()
        self.floors_trims = {}
        self.current_floor = "floor_0"


def test_trims_data_from_list():
    """Test TrimsData.from_list() with crop_area."""
    print("\n=== Test 1: TrimsData.from_list() ===")
    
    # Simulate crop_area from AutoCrop: [left, top, right, bottom]
    crop_area = [790, 490, 3209, 2509]
    
    trims = TrimsData.from_list(crop_area, floor="Ground Floor")
    
    print(f"Input crop_area: {crop_area}")
    print(f"Created TrimsData: {trims}")
    print(f"  floor: {trims.floor}")
    print(f"  trim_up: {trims.trim_up}")
    print(f"  trim_left: {trims.trim_left}")
    print(f"  trim_down: {trims.trim_down}")
    print(f"  trim_right: {trims.trim_right}")
    
    assert trims.trim_up == 790
    assert trims.trim_left == 490
    assert trims.trim_down == 3209
    assert trims.trim_right == 2509
    assert trims.floor == "Ground Floor"
    
    print("✅ Test 1 passed!")


def test_floor_data():
    """Test FloorData creation and serialization."""
    print("\n=== Test 2: FloorData ===")
    
    # Create TrimsData
    trims = TrimsData.from_list([790, 490, 3209, 2509], floor="Ground Floor")
    
    # Create FloorData
    floor_data = FloorData(trims=trims, map_name="map_0")
    
    print(f"Created FloorData: {floor_data}")
    print(f"  map_name: {floor_data.map_name}")
    print(f"  trims: {floor_data.trims}")
    
    # Test to_dict
    floor_dict = floor_data.to_dict()
    print(f"FloorData.to_dict(): {floor_dict}")
    
    # Test from_dict
    floor_data2 = FloorData.from_dict(floor_dict)
    print(f"FloorData.from_dict(): {floor_data2}")
    
    assert floor_data2.map_name == "map_0"
    assert floor_data2.trims.trim_up == 790
    assert floor_data2.trims.floor == "Ground Floor"
    
    print("✅ Test 2 passed!")


def test_camera_shared_floors():
    """Test CameraShared with multiple floors."""
    print("\n=== Test 3: CameraShared Multi-Floor ===")
    
    shared = CameraShared("test_vacuum")
    
    print(f"Initial current_floor: {shared.current_floor}")
    print(f"Initial floors_trims: {shared.floors_trims}")
    
    # Add floor_0
    trims_0 = TrimsData.from_list([790, 490, 3209, 2509], floor="Ground Floor")
    floor_0 = FloorData(trims=trims_0, map_name="map_0")
    shared.floors_trims["floor_0"] = floor_0
    
    # Add floor_1
    trims_1 = TrimsData.from_list([650, 380, 2950, 2200], floor="First Floor")
    floor_1 = FloorData(trims=trims_1, map_name="map_1")
    shared.floors_trims["floor_1"] = floor_1
    
    print(f"\nAdded 2 floors:")
    print(f"  floor_0: {shared.floors_trims['floor_0']}")
    print(f"  floor_1: {shared.floors_trims['floor_1']}")
    
    # Test accessing floor data
    assert shared.floors_trims["floor_0"].map_name == "map_0"
    assert shared.floors_trims["floor_0"].trims.trim_up == 790
    assert shared.floors_trims["floor_1"].map_name == "map_1"
    assert shared.floors_trims["floor_1"].trims.trim_up == 650
    
    print("✅ Test 3 passed!")


def test_update_trims_simulation():
    """Simulate BaseHandler.update_trims() workflow."""
    print("\n=== Test 4: Simulate update_trims() ===")
    
    shared = CameraShared("test_vacuum")
    
    # Simulate AutoCrop calculating crop_area
    crop_area = [790, 490, 3209, 2509]
    print(f"AutoCrop calculated crop_area: {crop_area}")
    
    # Simulate BaseHandler.update_trims()
    shared.trims = TrimsData.from_list(crop_area, floor="Ground Floor")
    print(f"Updated shared.trims: {shared.trims}")
    
    # Store in floors_trims
    floor_data = FloorData(trims=shared.trims, map_name="map_0")
    shared.floors_trims["floor_0"] = floor_data
    
    print(f"Stored in floors_trims['floor_0']: {shared.floors_trims['floor_0']}")
    
    # Verify
    assert shared.floors_trims["floor_0"].trims.trim_up == 790
    assert shared.floors_trims["floor_0"].trims.trim_left == 490
    assert shared.floors_trims["floor_0"].map_name == "map_0"
    
    print("✅ Test 4 passed!")


def test_floor_switching():
    """Test switching between floors."""
    print("\n=== Test 5: Floor Switching ===")
    
    shared = CameraShared("test_vacuum")
    
    # Setup two floors
    trims_0 = TrimsData.from_list([790, 490, 3209, 2509], floor="Ground Floor")
    floor_0 = FloorData(trims=trims_0, map_name="map_0")
    shared.floors_trims["floor_0"] = floor_0
    
    trims_1 = TrimsData.from_list([650, 380, 2950, 2200], floor="First Floor")
    floor_1 = FloorData(trims=trims_1, map_name="map_1")
    shared.floors_trims["floor_1"] = floor_1
    
    # Start on floor_0
    shared.current_floor = "floor_0"
    shared.trims = shared.floors_trims["floor_0"].trims
    print(f"Current floor: {shared.current_floor}")
    print(f"Current trims: {shared.trims}")
    
    # Switch to floor_1
    shared.current_floor = "floor_1"
    shared.trims = shared.floors_trims["floor_1"].trims
    print(f"\nSwitched to floor: {shared.current_floor}")
    print(f"Current trims: {shared.trims}")
    
    assert shared.trims.trim_up == 650
    assert shared.trims.floor == "First Floor"
    
    # Switch back to floor_0
    shared.current_floor = "floor_0"
    shared.trims = shared.floors_trims["floor_0"].trims
    print(f"\nSwitched back to floor: {shared.current_floor}")
    print(f"Current trims: {shared.trims}")
    
    assert shared.trims.trim_up == 790
    assert shared.trims.floor == "Ground Floor"
    
    print("✅ Test 5 passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing FloorData and Multi-Floor Support")
    print("=" * 60)
    
    try:
        test_trims_data_from_list()
        test_floor_data()
        test_camera_shared_floors()
        test_update_trims_simulation()
        test_floor_switching()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

