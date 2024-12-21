"""Colors for the maps Elements."""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar
from .types import Color

T = TypeVar("T")

class SupportedColor(StrEnum):
    """Color of a supported map element."""

    DEFAULT_ROOM_COLORS: dict[str, Color] = {
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
        "color_room_15": [72, 201, 176]
    }

    CHARGER = "color_charger"
    PATH = "color_move"
    PREDICTED_PATH = "color_predicted_move"
    WALLS ="color_wall"
    ROBOT = "color_robot"
    GO_TO = "color_go_to"
    NO_GO = "color_no_go"
    ZONE_CLEAN = "color_zone_clean"
    MAP_BACKGROUND = "color_background"
    TEXT= "color_text"
    ROOMS_LIST = list(dict(DEFAULT_ROOM_COLORS).keys())


DEFAULT_ALPHA: dict[str, float] = {
    "alpha_charger": 255.0,
    "alpha_move": 255.0,
    "alpha_wall": 255.0,
    "alpha_robot": 255.0,
    "alpha_go_to": 255.0,
    "alpha_no_go": 125.0,
    "alpha_zone_clean": 125.0,
    "alpha_background": 255.0,
    "alpha_text": 255.0,
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

class DefaultColors:
    """Container that simplifies retrieving desired color."""

    COLORS: dict[SupportedColor, Color] = {
        SupportedColor.CHARGER: [255, 128, 0, DEFAULT_ALPHA["alpha_charger"]],
        SupportedColor.PATH: [238, 247, 255, DEFAULT_ALPHA["alpha_move"]],
        SupportedColor.PREDICTED_PATH: [93, 109, 126, DEFAULT_ALPHA["alpha_move"]],
        SupportedColor.WALLS: [255, 255, 0, DEFAULT_ALPHA["alpha_wall"]],
        SupportedColor.ROBOT: [255, 255, 204, DEFAULT_ALPHA["alpha_robot"]],
        SupportedColor.GO_TO: [0, 255, 0, DEFAULT_ALPHA["alpha_go_to"]],
        SupportedColor.NO_GO: [255, 0, 0, DEFAULT_ALPHA["alpha_no_go"]],
        SupportedColor.ZONE_CLEAN: [255, 255, 255, DEFAULT_ALPHA["alpha_zone_clean"]],
        SupportedColor.MAP_BACKGROUND: [0, 125, 255, DEFAULT_ALPHA["alpha_background"]],
        SupportedColor.TEXT: [0, 0, 0, DEFAULT_ALPHA["alpha_text"]],
    }