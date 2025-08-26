"""
This module contains type aliases for the project.
Version 0.0.1
"""

import asyncio
import json
import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, TypedDict, Union

import numpy as np
from PIL import Image


DEFAULT_ROOMS = 1

LOGGER = logging.getLogger(__package__)


class RoomProperty(TypedDict):
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
    _instances: Dict[str, "RoomStore"] = {}
    _lock = threading.Lock()

    def __new__(cls, vacuum_id: str, rooms_data: Optional[dict] = None) -> "RoomStore":
        with cls._lock:
            if vacuum_id not in cls._instances:
                instance = super(RoomStore, cls).__new__(cls)
                instance.vacuum_id = vacuum_id
                instance.vacuums_data = rooms_data or {}
                cls._instances[vacuum_id] = instance
            else:
                if rooms_data is not None:
                    cls._instances[vacuum_id].vacuums_data = rooms_data
        return cls._instances[vacuum_id]

    def get_rooms(self) -> dict:
        return self.vacuums_data

    def set_rooms(self, rooms_data: dict) -> None:
        self.vacuums_data = rooms_data

    def get_rooms_count(self) -> int:
        if isinstance(self.vacuums_data, dict):
            count = len(self.vacuums_data)
            return count if count > 0 else DEFAULT_ROOMS
        return DEFAULT_ROOMS

    @classmethod
    def get_all_instances(cls) -> Dict[str, "RoomStore"]:
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

    async def get_user_language(self, user_id: str) -> str or None:
        """Get the user language."""
        async with self._lock:
            return self.user_languages.get(user_id, None)

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


Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]
Colors = Dict[str, Color]
CalibrationPoints = list[dict[str, Any]]
RobotPosition = dict[str, int | float]
ChargerPosition = dict[str, Any]
RoomsProperties = dict[str, RoomProperty]
ImageSize = dict[str, int | list[int]]
JsonType = Any  # json.loads() return type is Any
PilPNG = Image.Image  # Keep for backward compatibility
WebPBytes = bytes  # WebP image as bytes
NumpyArray = np.ndarray
Point = Tuple[int, int]

CAMERA_STORAGE = "valetudo_camera"
ATTR_ROTATE = "rotate_image"
ATTR_CROP = "crop_image"
ATTR_MARGINS = "margins"
CONF_OFFSET_TOP = "offset_top"
CONF_OFFSET_BOTTOM = "offset_bottom"
CONF_OFFSET_LEFT = "offset_left"
CONF_OFFSET_RIGHT = "offset_right"
CONF_ASPECT_RATIO = "aspect_ratio"
CONF_VAC_STAT = "show_vac_status"
CONF_VAC_STAT_SIZE = "vac_status_size"
CONF_VAC_STAT_POS = "vac_status_position"
CONF_VAC_STAT_FONT = "vac_status_font"
CONF_VACUUM_CONNECTION_STRING = "vacuum_map"
CONF_VACUUM_ENTITY_ID = "vacuum_entity"
CONF_VACUUM_CONFIG_ENTRY_ID = "vacuum_config_entry"
CONF_VACUUM_IDENTIFIERS = "vacuum_identifiers"
CONF_SNAPSHOTS_ENABLE = "enable_www_snapshots"
CONF_EXPORT_SVG = "get_svg_file"
CONF_AUTO_ZOOM = "auto_zoom"
CONF_ZOOM_LOCK_RATIO = "zoom_lock_ratio"
CONF_TRIMS_SAVE = "save_trims"
ICON = "mdi:camera"
NAME = "MQTT Vacuum Camera"

DEFAULT_IMAGE_SIZE = {
    "x": 5120,
    "y": 5120,
    "centre": [(5120 // 2), (5120 // 2)],
}

COLORS = [
    "wall",
    "zone_clean",
    "robot",
    "background",
    "move",
    "charger",
    "no_go",
    "go_to",
]

SENSOR_NO_DATA = {
    "mainBrush": 0,
    "sideBrush": 0,
    "filter": 0,
    "currentCleanTime": 0,
    "currentCleanArea": 0,
    "cleanTime": 0,
    "cleanArea": 0,
    "cleanCount": 0,
    "battery": 0,
    "state": 0,
    "last_run_start": 0,
    "last_run_end": 0,
    "last_run_duration": 0,
    "last_run_area": 0,
    "last_bin_out": 0,
    "last_bin_full": 0,
    "last_loaded_map": "NoMap",
    "robot_in_room": "Unsupported",
}

DEFAULT_PIXEL_SIZE = 5

DEFAULT_VALUES = {
    "rotate_image": "0",
    "margins": "100",
    "aspect_ratio": "None",
    "offset_top": 0,
    "offset_bottom": 0,
    "offset_left": 0,
    "offset_right": 0,
    "auto_zoom": False,
    "zoom_lock_ratio": True,
    "show_vac_status": False,
    "vac_status_font": "custom_components/mqtt_vacuum_camera/utils/fonts/FiraSans.ttf",
    "vac_status_size": 50,
    "vac_status_position": True,
    "get_svg_file": False,
    "save_trims": True,
    "trims_data": {"trim_left": 0, "trim_up": 0, "trim_right": 0, "trim_down": 0},
    "enable_www_snapshots": False,
    "color_charger": [255, 128, 0],
    "color_move": [238, 247, 255],
    "color_wall": [255, 255, 0],
    "color_robot": [255, 255, 204],
    "color_go_to": [0, 255, 0],
    "color_no_go": [255, 0, 0],
    "color_zone_clean": [255, 255, 255],
    "color_background": [0, 125, 255],
    "color_text": [255, 255, 255],
    "alpha_charger": 255.0,
    "alpha_move": 255.0,
    "alpha_wall": 255.0,
    "alpha_robot": 255.0,
    "alpha_go_to": 255.0,
    "alpha_no_go": 125.0,
    "alpha_zone_clean": 125.0,
    "alpha_background": 255.0,
    "alpha_text": 255.0,
    "color_room_0": [135, 206, 250],
    "color_room_1": [176, 226, 255],
    "color_room_2": [165, 105, 18],
    "color_room_3": [164, 211, 238],
    "color_room_4": [141, 182, 205],
    "color_room_5": [96, 123, 139],
    "color_room_6": [224, 255, 255],
    "color_room_7": [209, 238, 238],
    "color_room_8": [180, 205, 205],
    "color_room_9": [122, 139, 139],
    "color_room_10": [175, 238, 238],
    "color_room_11": [84, 153, 199],
    "color_room_12": [133, 193, 233],
    "color_room_13": [245, 176, 65],
    "color_room_14": [82, 190, 128],
    "color_room_15": [72, 201, 176],
    "alpha_room_0": 255.0,
    "alpha_room_1": 255.0,
    "alpha_room_2": 255.0,
    "alpha_room_3": 255.0,
    "alpha_room_4": 255.0,
    "alpha_room_5": 255.0,
    "alpha_room_6": 255.0,
    "alpha_room_7": 255.0,
    "alpha_room_8": 255.0,
    "alpha_room_9": 255.0,
    "alpha_room_10": 255.0,
    "alpha_room_11": 255.0,
    "alpha_room_12": 255.0,
    "alpha_room_13": 255.0,
    "alpha_room_14": 255.0,
    "alpha_room_15": 255.0,
}

KEYS_TO_UPDATE = [
    "rotate_image",
    "margins",
    "aspect_ratio",
    "offset_top",
    "offset_bottom",
    "offset_left",
    "offset_right",
    "trims_data",
    "auto_zoom",
    "zoom_lock_ratio",
    "show_vac_status",
    "vac_status_size",
    "vac_status_position",
    "vac_status_font",
    "get_svg_file",
    "enable_www_snapshots",
    "color_charger",
    "color_move",
    "color_wall",
    "color_robot",
    "color_go_to",
    "color_no_go",
    "color_zone_clean",
    "color_background",
    "color_text",
    "alpha_charger",
    "alpha_move",
    "alpha_wall",
    "alpha_robot",
    "alpha_go_to",
    "alpha_no_go",
    "alpha_zone_clean",
    "alpha_background",
    "alpha_text",
    "color_room_0",
    "color_room_1",
    "color_room_2",
    "color_room_3",
    "color_room_4",
    "color_room_5",
    "color_room_6",
    "color_room_7",
    "color_room_8",
    "color_room_9",
    "color_room_10",
    "color_room_11",
    "color_room_12",
    "color_room_13",
    "color_room_14",
    "color_room_15",
    "alpha_room_0",
    "alpha_room_1",
    "alpha_room_2",
    "alpha_room_3",
    "alpha_room_4",
    "alpha_room_5",
    "alpha_room_6",
    "alpha_room_7",
    "alpha_room_8",
    "alpha_room_9",
    "alpha_room_10",
    "alpha_room_11",
    "alpha_room_12",
    "alpha_room_13",
    "alpha_room_14",
    "alpha_room_15",
]

ALPHA_VALUES = {
    "min": 0.0,  # Minimum value
    "max": 255.0,  # Maximum value
    "step": 1.0,  # Step value
}

TEXT_SIZE_VALUES = {
    "min": 5,  # Minimum value
    "max": 51,  # Maximum value
    "step": 1,  # Step value
}

ROTATION_VALUES = [
    {"label": "0", "value": "0"},
    {"label": "90", "value": "90"},
    {"label": "180", "value": "180"},
    {"label": "270", "value": "270"},
]

RATIO_VALUES = [
    {"label": "Original Ratio.", "value": "None"},
    {"label": "1:1", "value": "1, 1"},
    {"label": "2:1", "value": "2, 1"},
    {"label": "3:2", "value": "3, 2"},
    {"label": "5:4", "value": "5, 4"},
    {"label": "9:16", "value": "9, 16"},
    {"label": "16:9", "value": "16, 9"},
]

FONTS_AVAILABLE = [
    {
        "label": "Fira Sans",
        "value": "custom_components/mqtt_vacuum_camera/utils/fonts/FiraSans.ttf",
    },
    {
        "label": "Inter",
        "value": "custom_components/mqtt_vacuum_camera/utils/fonts/Inter-VF.ttf",
    },
    {
        "label": "M Plus Regular",
        "value": "custom_components/mqtt_vacuum_camera/utils/fonts/MPLUSRegular.ttf",
    },
    {
        "label": "Noto Sans CJKhk",
        "value": "custom_components/mqtt_vacuum_camera/utils/fonts/NotoSansCJKhk-VF.ttf",
    },
    {
        "label": "Noto Kufi Arabic",
        "value": "custom_components/mqtt_vacuum_camera/utils/fonts/NotoKufiArabic-VF.ttf",
    },
    {
        "label": "Noto Sans Khojki",
        "value": "custom_components/mqtt_vacuum_camera/utils/fonts/NotoSansKhojki.ttf",
    },
    {
        "label": "Lato Regular",
        "value": "custom_components/mqtt_vacuum_camera/utils/fonts/Lato-Regular.ttf",
    },
]

NOT_STREAMING_STATES = {
    "idle",
    "paused",
    "charging",
    "error",
    "docked",
}

DECODED_TOPICS = {
    "/MapData/segments",
    "/maploader/map",
    "/maploader/status",
    "/StatusStateAttribute/status",
    "/StatusStateAttribute/error_description",
    "/$state",
    "/BatteryStateAttribute/level",
    "/WifiConfigurationCapability/ips",
    "/state",  # Rand256
    "/destinations",  # Rand256
    "/command",  # Rand256
    "/custom_command",  # Rand256
    "/attributes",  # Rand256
}


# self.command_topic need to be added to this dictionary after init.
NON_DECODED_TOPICS = {
    "/MapData/map-data",
    "/map_data",
}

"""App Constants. Not in use, and dummy values"""
IDLE_SCAN_INTERVAL = 120
CLEANING_SCAN_INTERVAL = 5
IS_ALPHA = "add_base_alpha"
IS_ALPHA_R1 = "add_room_1_alpha"
IS_ALPHA_R2 = "add_room_2_alpha"
IS_OFFSET = "add_offset"

"""Base Colours RGB"""
COLOR_CHARGER = "color_charger"
COLOR_MOVE = "color_move"
COLOR_ROBOT = "color_robot"
COLOR_NO_GO = "color_no_go"
COLOR_GO_TO = "color_go_to"
COLOR_BACKGROUND = "color_background"
COLOR_ZONE_CLEAN = "color_zone_clean"
COLOR_WALL = "color_wall"
COLOR_TEXT = "color_text"

"Rooms Colours RGB"
COLOR_ROOM_0 = "color_room_0"
COLOR_ROOM_1 = "color_room_1"
COLOR_ROOM_2 = "color_room_2"
COLOR_ROOM_3 = "color_room_3"
COLOR_ROOM_4 = "color_room_4"
COLOR_ROOM_5 = "color_room_5"
COLOR_ROOM_6 = "color_room_6"
COLOR_ROOM_7 = "color_room_7"
COLOR_ROOM_8 = "color_room_8"
COLOR_ROOM_9 = "color_room_9"
COLOR_ROOM_10 = "color_room_10"
COLOR_ROOM_11 = "color_room_11"
COLOR_ROOM_12 = "color_room_12"
COLOR_ROOM_13 = "color_room_13"
COLOR_ROOM_14 = "color_room_14"
COLOR_ROOM_15 = "color_room_15"

"""Alpha for RGBA Colours"""
ALPHA_CHARGER = "alpha_charger"
ALPHA_MOVE = "alpha_move"
ALPHA_ROBOT = "alpha_robot"
ALPHA_NO_GO = "alpha_no_go"
ALPHA_GO_TO = "alpha_go_to"
ALPHA_BACKGROUND = "alpha_background"
ALPHA_ZONE_CLEAN = "alpha_zone_clean"
ALPHA_WALL = "alpha_wall"
ALPHA_TEXT = "alpha_text"
ALPHA_ROOM_0 = "alpha_room_0"
ALPHA_ROOM_1 = "alpha_room_1"
ALPHA_ROOM_2 = "alpha_room_2"
ALPHA_ROOM_3 = "alpha_room_3"
ALPHA_ROOM_4 = "alpha_room_4"
ALPHA_ROOM_5 = "alpha_room_5"
ALPHA_ROOM_6 = "alpha_room_6"
ALPHA_ROOM_7 = "alpha_room_7"
ALPHA_ROOM_8 = "alpha_room_8"
ALPHA_ROOM_9 = "alpha_room_9"
ALPHA_ROOM_10 = "alpha_room_10"
ALPHA_ROOM_11 = "alpha_room_11"
ALPHA_ROOM_12 = "alpha_room_12"
ALPHA_ROOM_13 = "alpha_room_13"
ALPHA_ROOM_14 = "alpha_room_14"
ALPHA_ROOM_15 = "alpha_room_15"

""" Constants for the attribute keys """
ATTR_FRIENDLY_NAME = "friendly_name"
ATTR_VACUUM_BATTERY = "battery"
ATTR_VACUUM_CHARGING = "charging"
ATTR_VACUUM_POSITION = "vacuum_position"
ATTR_VACUUM_TOPIC = "vacuum_topic"
ATTR_VACUUM_STATUS = "vacuum_status"
ATTR_JSON_DATA = "json_data"
ATTR_VACUUM_JSON_ID = "vacuum_json_id"
ATTR_CALIBRATION_POINTS = "calibration_points"
ATTR_SNAPSHOT = "snapshot"
ATTR_SNAPSHOT_PATH = "snapshot_path"
ATTR_ROOMS = "rooms"
ATTR_ZONES = "zones"
ATTR_POINTS = "points"
ATTR_OBSTACLES = "obstacles"
ATTR_CAMERA_MODE = "camera_mode"


class CameraModes:
    """Constants for the camera modes"""

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

    def clear(self) -> dict:
        """Clear all the trims."""
        self.floor = ""
        self.trim_up = 0
        self.trim_left = 0
        self.trim_down = 0
        self.trim_right = 0
        return asdict(self)
