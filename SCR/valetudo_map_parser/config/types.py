"""
This module contains type aliases for the project.
Version 0.11.1
"""

import asyncio
import json
import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, NotRequired, Optional, Tuple, TypedDict, Union

import numpy as np
from PIL import Image


DEFAULT_ROOMS = 1

LOGGER = logging.getLogger(__package__)


class Spot(TypedDict):
    """Type definition for a spot location."""

    name: str
    coordinates: List[int]  # [x, y]


class Zone(TypedDict):
    """Type definition for a zone area."""

    name: str
    coordinates: List[List[int]]  # [[x1, y1, x2, y2, repeats], ...]


class Room(TypedDict):
    """Type definition for a room."""

    name: str
    id: int


class Destinations(TypedDict, total=False):
    """Type definition for destinations including spots, zones, and rooms."""

    spots: NotRequired[Optional[List[Spot]]]
    zones: NotRequired[Optional[List[Zone]]]
    rooms: NotRequired[Optional[List[Room]]]
    updated: NotRequired[Optional[float | int]]


class RoomProperty(TypedDict):
    """Type definition for room properties including outline."""

    number: int
    outline: list[tuple[int, int]]
    name: str
    x: int
    y: int


# pylint: disable=no-member
@dataclass
class TrimCropData:
    """Dataclass for trim and crop data."""

    trim_left: int
    trim_up: int
    trim_right: int
    trim_down: int

    def to_dict(self) -> dict:
        """Convert dataclass to dictionary."""
        return {
            "trim_left": self.trim_left,
            "trim_up": self.trim_up,
            "trim_right": self.trim_right,
            "trim_down": self.trim_down,
        }

    @staticmethod
    def from_dict(data: dict):
        """Create dataclass from dictionary."""
        return TrimCropData(
            trim_left=data["trim_left"],
            trim_up=data["trim_up"],
            trim_right=data["trim_right"],
            trim_down=data["trim_down"],
        )

    def to_list(self) -> list:
        """Convert dataclass to list."""
        return [self.trim_left, self.trim_up, self.trim_right, self.trim_down]

    @staticmethod
    def from_list(data: list):
        """Create dataclass from list."""
        return TrimCropData(
            trim_left=data[0],
            trim_up=data[1],
            trim_right=data[2],
            trim_down=data[3],
        )


class RoomStore:
    """Singleton storage for room data per vacuum ID.

    Stores room properties in format: {segment_id: RoomProperty}
    Example: {"16": {"number": 16, "outline": [...], "name": "Living Room", "x": 100, "y": 200}}
    """

    _instances: Dict[str, "RoomStore"] = {}
    _lock = threading.Lock()

    def __new__(
        cls, vacuum_id: str, rooms_data: Optional[Dict[str, RoomProperty]] = None
    ) -> "RoomStore":
        with cls._lock:
            if vacuum_id not in cls._instances:
                instance = super(RoomStore, cls).__new__(cls)
                cls._instances[vacuum_id] = instance
            return cls._instances[vacuum_id]

    def __init__(
        self, vacuum_id: str, rooms_data: Optional[Dict[str, RoomProperty]] = None
    ) -> None:
        # Only initialize if this is a new instance (not yet initialized)
        if not hasattr(self, "vacuum_id"):
            self.vacuum_id: str = vacuum_id
            self.vacuums_data: Dict[str, RoomProperty] = rooms_data or {}
            self.rooms_count: int = self.get_rooms_count()
            self.floor: Optional[str] = None
        elif rooms_data is not None:
            # Update only if new data is provided
            self.vacuums_data = rooms_data
            self.rooms_count = self.get_rooms_count()

    def get_rooms(self) -> Dict[str, RoomProperty]:
        """Get all rooms data."""
        return self.vacuums_data

    def set_rooms(self, rooms_data: Dict[str, RoomProperty]) -> None:
        """Set rooms data and update room count."""
        self.vacuums_data = rooms_data
        self.rooms_count = self.get_rooms_count()

    def get_rooms_count(self) -> int:
        """Get the number of rooms, defaulting to 1 if no rooms are present."""
        if isinstance(self.vacuums_data, dict):
            count = len(self.vacuums_data)
            return count if count > 0 else DEFAULT_ROOMS
        return DEFAULT_ROOMS

    @property
    def room_names(self) -> dict:
        """Return room names in format {'room_0_name': 'SegmentID: RoomName', ...}.

        Maximum of 16 rooms supported.
        """
        result = {}
        if isinstance(self.vacuums_data, dict):
            for idx, (segment_id, room_data) in enumerate(self.vacuums_data.items()):
                if idx >= 16:  # Max 16 rooms supported
                    break
                room_name = room_data.get("name", f"Room {segment_id}")
                result[f"room_{idx}_name"] = f"{segment_id}: {room_name}"
        return result

    @classmethod
    def get_all_instances(cls) -> Dict[str, "RoomStore"]:
        """Get all RoomStore instances for all vacuum IDs."""
        return cls._instances


# pylint: disable=no-member
class UserLanguageStore:
    """Store the user language data."""

    _instance = None
    _lock = asyncio.Lock()
    _initialized = False

    def __init__(self):
        self.user_languages = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UserLanguageStore, cls).__new__(cls)
            cls._instance.user_languages = {}
        return cls._instance

    async def set_user_language(self, user_id: str, language: str) -> None:
        """Set the user language."""
        async with self._lock:
            self.user_languages[user_id] = language

    async def get_user_language(self, user_id: str) -> str:
        """Get the user language."""
        async with self._lock:
            return self.user_languages.get(user_id, "")

    async def get_all_languages(self):
        """Get all the user languages."""
        async with self._lock:
            if not self.user_languages:
                return ["en"]
            return list(self.user_languages.values())

    @classmethod
    async def is_initialized(cls):
        """Return if the instance is initialized."""
        async with cls._lock:
            return bool(cls._initialized)

    @classmethod
    async def initialize_if_needed(cls, other_instance=None):
        """Initialize the instance if needed by copying from another instance if available."""
        async with cls._lock:
            if not cls._initialized and other_instance is not None:
                cls._instance.user_languages = other_instance.user_languages
                cls._initialized = True


# pylint: disable=no-member
class SnapshotStore:
    """Store the snapshot data."""

    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.snapshot_save_data = {}
        self.vacuum_json_data = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SnapshotStore, cls).__new__(cls)
            cls._instance.snapshot_save_data = {}
            cls._instance.vacuum_json_data = {}
        return cls._instance

    async def async_set_snapshot_save_data(
        self, vacuum_id: str, snapshot_data: bool = False
    ) -> None:
        """Set the snapshot save data for the vacuum."""
        async with self._lock:
            self.snapshot_save_data[vacuum_id] = snapshot_data

    async def async_get_snapshot_save_data(self, vacuum_id: str) -> bool:
        """Get the snapshot save data for a vacuum."""
        async with self._lock:
            return self.snapshot_save_data.get(vacuum_id, False)

    async def async_get_vacuum_json(self, vacuum_id: str) -> Any:
        """Get the JSON data for a vacuum."""
        async with self._lock:
            return self.vacuum_json_data.get(vacuum_id, {})

    async def async_set_vacuum_json(self, vacuum_id: str, json_data: Any) -> None:
        """Set the JSON data for the vacuum."""
        async with self._lock:
            self.vacuum_json_data[vacuum_id] = json_data


class CameraModes:
    """Constants for the camera modes."""

    MAP_VIEW = "map_view"
    OBSTACLE_VIEW = "obstacle_view"
    OBSTACLE_DOWNLOAD = "load_view"
    OBSTACLE_SEARCH = "search_view"
    CAMERA_STANDBY = "camera_standby"
    CAMERA_OFF = False
    CAMERA_ON = True


# noinspection PyTypeChecker
@dataclass
class TrimsData:
    """Dataclass to store and manage trims data."""

    floor: str = ""
    trim_up: int = 0
    trim_left: int = 0
    trim_down: int = 0
    trim_right: int = 0

    @classmethod
    def from_json(cls, json_data: str):
        """Create a TrimsConfig instance from a JSON string."""
        data = json.loads(json_data)
        return cls(
            floor=data.get("floor", ""),
            trim_up=data.get("trim_up", 0),
            trim_left=data.get("trim_left", 0),
            trim_down=data.get("trim_down", 0),
            trim_right=data.get("trim_right", 0),
        )

    def to_json(self) -> str:
        """Convert TrimsConfig instance to a JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, data: dict):
        """Initialize TrimData from a dictionary."""
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert TrimData to a dictionary."""
        return asdict(self)

    @classmethod
    def from_list(cls, crop_area: List[int], floor: Optional[str] = None):
        """
        Initialize TrimsData from a list [trim_up, trim_left, trim_down, trim_right]
        """
        return cls(
            trim_up=crop_area[0],
            trim_left=crop_area[1],
            trim_down=crop_area[2],
            trim_right=crop_area[3],
            floor=floor,
        )

    def clear(self) -> dict:
        """Clear all the trims."""
        self.floor = ""
        self.trim_up = 0
        self.trim_left = 0
        self.trim_down = 0
        self.trim_right = 0
        return asdict(self)


@dataclass
class FloorData:
    """Dataclass to store floor configuration."""

    trims: TrimsData
    map_name: str = ""

    @classmethod
    def from_dict(cls, data: dict):
        """Initialize FloorData from a dictionary."""
        return cls(
            trims=TrimsData.from_dict(data.get("trims", {})),
            map_name=data.get("map_name", ""),
        )

    def to_dict(self) -> dict:
        """Convert FloorData to a dictionary."""
        return {"trims": self.trims.to_dict(), "map_name": self.map_name}


Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]
Colors = Dict[str, Color]
CalibrationPoints = list[dict[str, Any]]
RobotPosition: type[tuple[Any, Any, dict[str, int | float] | None]] = tuple[
    Any, Any, dict[str, int | float] | None
]
ChargerPosition = dict[str, Any]
RoomsProperties = dict[str, RoomProperty]
ImageSize = dict[str, int | list[int]]
Size = dict[str, int]
JsonType = Any  # json.loads() return type is Any
PilPNG = Image.Image  # Keep for backward compatibility
NumpyArray = np.ndarray
Point = Tuple[int, int]
