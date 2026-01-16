"""Constants for the Valetudo Map Parser library."""

CAMERA_STORAGE = "valetudo_camera"
ATTR_IMAGE_LAST_UPDATED = "image_last_updated"
ATTR_ROTATE = "rotate_image"
ATTR_CROP = "crop_image"
ATTR_MARGINS = "margins"
ATTR_CONTENT_TYPE = "content_type"
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
    "carpet",
    "no_go",
    "go_to",
    "text",
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
    "vac_status_font": "SCR/valetudo_map_parser/config/fonts/FiraSans.ttf",
    "vac_status_size": 50,
    "vac_status_position": True,
    "get_svg_file": False,
    "save_trims": True,
    "trims_data": {
        "floor": "floor_0",
        "trim_left": 0,
        "trim_up": 0,
        "trim_right": 0,
        "trim_down": 0,
    },
    "enable_www_snapshots": False,
    "color_charger": [255, 128, 0],
    "color_move": [238, 247, 255],
    "color_wall": [255, 255, 0],
    "color_robot": [255, 255, 204],
    "color_go_to": [0, 255, 0],
    "color_no_go": [255, 0, 0],
    "color_zone_clean": [255, 255, 255],
    "color_carpet": [67, 103, 125],
    "color_background": [0, 125, 255],
    "color_text": [255, 255, 255],
    "color_material_wood": [40, 40, 40],
    "color_material_tile": [40, 40, 40],
    "alpha_charger": 255.0,
    "alpha_move": 255.0,
    "alpha_wall": 255.0,
    "alpha_robot": 255.0,
    "alpha_go_to": 255.0,
    "alpha_no_go": 125.0,
    "alpha_zone_clean": 125.0,
    "alpha_carpet": 255.0,
    "alpha_background": 255.0,
    "alpha_text": 255.0,
    "alpha_material_wood": 38.0,
    "alpha_material_tile": 45.0,
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

DEFAULT_ROOMS = 1

DEFAULT_ROOMS_NAMES = {
    "room_0_name": "Room 1",
    "room_1_name": "Room 2",
    "room_2_name": "Room 3",
    "room_3_name": "Room 4",
    "room_4_name": "Room 5",
    "room_5_name": "Room 6",
    "room_6_name": "Room 7",
    "room_7_name": "Room 8",
    "room_8_name": "Room 9",
    "room_9_name": "Room 10",
    "room_10_name": "Room 11",
    "room_11_name": "Room 12",
    "room_12_name": "Room 13",
    "room_13_name": "Room 14",
    "room_14_name": "Room 15",
}

FONTS_AVAILABLE = [
    {
        "label": "Fira Sans",
        "value": "config/fonts/FiraSans.ttf",
    },
    {
        "label": "Inter",
        "value": "config/fonts/Inter-VF.ttf",
    },
    {
        "label": "M Plus Regular",
        "value": "config/fonts/MPLUSRegular.ttf",
    },
    {
        "label": "Noto Sans CJKhk",
        "value": "config/fonts/NotoSansCJKhk-VF.ttf",
    },
    {
        "label": "Noto Kufi Arabic",
        "value": "config/fonts/NotoKufiArabic-VF.ttf",
    },
    {
        "label": "Noto Sans Khojki",
        "value": "config/fonts/NotoSansKhojki.ttf",
    },
    {
        "label": "Lato Regular",
        "value": "config/fonts/Lato-Regular.ttf",
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
COLOR_CARPET = "color_carpet"
COLOR_MOVE = "color_move"
COLOR_MATERIAL_WOOD = "color_material_wood"
COLOR_MATERIAL_TILE = "color_material_tile"
COLOR_ROBOT = "color_robot"
COLOR_NO_GO = "color_no_go"
COLOR_GO_TO = "color_go_to"
COLOR_BACKGROUND = "color_background"
COLOR_ZONE_CLEAN = "color_zone_clean"
COLOR_WALL = "color_wall"
COLOR_TEXT = "color_text"

"""Rooms Colours RGB"""
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
ALPHA_CARPET = "alpha_carpet"
ALPHA_MOVE = "alpha_move"
ALPHA_ROBOT = "alpha_robot"
ALPHA_NO_GO = "alpha_no_go"
ALPHA_GO_TO = "alpha_go_to"
ALPHA_BACKGROUND = "alpha_background"
ALPHA_MATERIAL_WOOD = "alpha_material_wood"
ALPHA_MATERIAL_TILE = "alpha_material_tile"
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

# Status text constants
charge_level = "\u03de"  # unicode Koppa symbol
charging = "\u2211"  # unicode Charging symbol
dot = " \u00b7 "  # unicode middle dot
text_size_coverage = 1.5  # resize factor for the text
