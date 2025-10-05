"""Valetudo map parser.
Version: 0.1.10"""

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
    ImageSize,
    JsonType,
    NumpyArray,
    PilPNG,
    RoomsProperties,
    RoomStore,
    SnapshotStore,
    TrimCropData,
    UserLanguageStore,
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
    "RoomsHandler",
    "RandRoomsHandler",
    "HyperMapData",
    "HypferMapImageHandler",
    "ReImageHandler",
    "RRMapParser",
    "CameraShared",
    "CameraSharedManager",
    "ColorsManagement",
    "Drawable",
    "DrawableElement",
    "DrawingConfig",
    "SnapshotStore",
    "UserLanguageStore",
    "RoomStore",
    "RoomsProperties",
    "TrimCropData",
    "CameraModes",
    "JsonType",
    "PilPNG",
    "NumpyArray",
    "ImageSize",
    "StatusText",
    "STATUS_TEXT_TRANSLATIONS",
    "get_default_font_path",
]
