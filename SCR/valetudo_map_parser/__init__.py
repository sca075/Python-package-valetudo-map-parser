"""Valetudo map parser.
Version: 0.2.5b0"""

from pathlib import Path

from .config.colors import ColorsManagement
from .config.drawable import Drawable
from .config.drawable_elements import DrawableElement, DrawingConfig
from .config.rand256_parser import RRMapParser
from .config.shared import CameraShared, CameraSharedManager
from .config.status_text.status_text import StatusText
from .config.status_text.translations import translations as STATUS_TEXT_TRANSLATIONS
from .config.types import (
    CameraModes,
    FloorData,
    ImageSize,
    JsonType,
    NumpyArray,
    PilPNG,
    RoomsProperties,
    RoomStore,
    SnapshotStore,
    TrimCropData,
    TrimsData,
    UserLanguageStore,
)
from .config.utils import ResizeParams, async_resize_image
from .const import (
    ATTR_CALIBRATION_POINTS,
    ATTR_CAMERA_MODE,
    ATTR_CONTENT_TYPE,
    ATTR_DOCK_STATE,
    ATTR_FRIENDLY_NAME,
    ATTR_IMAGE_LAST_UPDATED,
    ATTR_JSON_DATA,
    ATTR_OBSTACLES,
    ATTR_POINTS,
    ATTR_ROOMS,
    ATTR_ROTATE,
    ATTR_SNAPSHOT,
    ATTR_SNAPSHOT_PATH,
    ATTR_VACUUM_BATTERY,
    ATTR_VACUUM_CHARGING,
    ATTR_VACUUM_JSON_ID,
    ATTR_VACUUM_POSITION,
    ATTR_VACUUM_STATUS,
    ATTR_VACUUM_TOPIC,
    ATTR_ZONES,
    CAMERA_STORAGE,
    COLORS,
    CONF_ASPECT_RATIO,
    CONF_AUTO_ZOOM,
    CONF_EXPORT_SVG,
    CONF_OBSTACLE_LINK_IP,
    CONF_OBSTACLE_LINK_PORT,
    CONF_OBSTACLE_LINK_PROTOCOL,
    CONF_OFFSET_BOTTOM,
    CONF_OFFSET_LEFT,
    CONF_OFFSET_RIGHT,
    CONF_OFFSET_TOP,
    CONF_SNAPSHOTS_ENABLE,
    CONF_TRIMS_SAVE,
    CONF_VAC_STAT,
    CONF_VAC_STAT_FONT,
    CONF_VAC_STAT_POS,
    CONF_VAC_STAT_SIZE,
    CONF_VACUUM_CONFIG_ENTRY_ID,
    CONF_VACUUM_CONNECTION_STRING,
    CONF_VACUUM_ENTITY_ID,
    CONF_VACUUM_IDENTIFIERS,
    CONF_ZOOM_LOCK_RATIO,
    DECODED_TOPICS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PIXEL_SIZE,
    DEFAULT_ROOMS,
    DEFAULT_ROOMS_NAMES,
    DEFAULT_VALUES,
    FONTS_AVAILABLE,
    ICON,
    NAME,
    NON_DECODED_TOPICS,
    NOT_STREAMING_STATES,
    SENSOR_NO_DATA,
)
from .hypfer_handler import HypferMapImageHandler
from .map_data import HyperMapData
from .rand256_handler import ReImageHandler
from .rooms_handler import RandRoomsHandler, RoomsHandler


def get_default_font_path() -> str:
    """Return the absolute path to the bundled default font directory.

    This returns the path to the fonts folder; the caller can join a specific font file
    to avoid hard-coding a particular font here.
    """
    return str((Path(__file__).resolve().parent / "config" / "fonts").resolve())


__all__ = [
    # Attribute Constants
    "ATTR_CALIBRATION_POINTS",
    "ATTR_CAMERA_MODE",
    "ATTR_CONTENT_TYPE",
    "ATTR_DOCK_STATE",
    "ATTR_FRIENDLY_NAME",
    "ATTR_IMAGE_LAST_UPDATED",
    "ATTR_JSON_DATA",
    "ATTR_OBSTACLES",
    "ATTR_POINTS",
    "ATTR_ROOMS",
    "ATTR_ROTATE",
    "ATTR_SNAPSHOT",
    "ATTR_SNAPSHOT_PATH",
    "ATTR_VACUUM_BATTERY",
    "ATTR_VACUUM_CHARGING",
    "ATTR_VACUUM_JSON_ID",
    "ATTR_VACUUM_POSITION",
    "ATTR_VACUUM_STATUS",
    "ATTR_VACUUM_TOPIC",
    "ATTR_ZONES",
    # Configuration Constants
    "CAMERA_STORAGE",
    "COLORS",
    "CONF_ASPECT_RATIO",
    "CONF_AUTO_ZOOM",
    "CONF_EXPORT_SVG",
    "CONF_OBSTACLE_LINK_IP",
    "CONF_OBSTACLE_LINK_PORT",
    "CONF_OBSTACLE_LINK_PROTOCOL",
    "CONF_OFFSET_BOTTOM",
    "CONF_OFFSET_LEFT",
    "CONF_OFFSET_RIGHT",
    "CONF_OFFSET_TOP",
    "CONF_SNAPSHOTS_ENABLE",
    "CONF_TRIMS_SAVE",
    "CONF_VACUUM_CONFIG_ENTRY_ID",
    "CONF_VACUUM_CONNECTION_STRING",
    "CONF_VACUUM_ENTITY_ID",
    "CONF_VACUUM_IDENTIFIERS",
    "CONF_VAC_STAT",
    "CONF_VAC_STAT_FONT",
    "CONF_VAC_STAT_POS",
    "CONF_VAC_STAT_SIZE",
    "CONF_ZOOM_LOCK_RATIO",
    # Default Values
    "DECODED_TOPICS",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_PIXEL_SIZE",
    "DEFAULT_ROOMS",
    "DEFAULT_ROOMS_NAMES",
    "DEFAULT_VALUES",
    "FONTS_AVAILABLE",
    "ICON",
    "NAME",
    "NON_DECODED_TOPICS",
    "NOT_STREAMING_STATES",
    "SENSOR_NO_DATA",
    # Classes and Handlers
    "CameraShared",
    "CameraSharedManager",
    "ColorsManagement",
    "Drawable",
    "DrawableElement",
    "DrawingConfig",
    "HyperMapData",
    "HypferMapImageHandler",
    "RRMapParser",
    "RandRoomsHandler",
    "ReImageHandler",
    "RoomsHandler",
    "StatusText",
    # Types
    "CameraModes",
    "FloorData",
    "ImageSize",
    "JsonType",
    "NumpyArray",
    "PilPNG",
    "RoomsProperties",
    "RoomStore",
    "SnapshotStore",
    "TrimCropData",
    "TrimsData",
    "UserLanguageStore",
    # Utilities
    "ResizeParams",
    "STATUS_TEXT_TRANSLATIONS",
    "async_resize_image",
    "get_default_font_path",
]
