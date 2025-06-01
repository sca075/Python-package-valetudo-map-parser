from __future__ import annotations

import asyncio
import json
import logging
import os
import cProfile
import pstats

import sys
import os

# Add the parent directory to the path so we can import the SCR module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SCR.valetudo_map_parser.config.colors import ColorsManagement
from SCR.valetudo_map_parser.config.drawable_elements import DrawableElement
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.config.types import RoomStore
from SCR.valetudo_map_parser.rand25_handler import ReImageHandler

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Ensures DEBUG logs are displayed
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s (line %(lineno)d) - %(message)s",
)

_LOGGER = logging.getLogger(__name__)


class TestRandImageHandler:
    def __init__(self):
        self.test_data = None
        self.image = None

    def setUp(self):
        # Load test data from the rand.json file
        test_file_path = os.path.join(os.path.dirname(__file__), "rand.json")
        logging.getLogger(__name__).info(f"Loading test data from {test_file_path}")

        with open(test_file_path, "r") as file:
            self.test_data = json.load(file)

        _LOGGER.debug("Test data loaded.")

    async def test_image_handler(self):
        _LOGGER.info("Starting test_rand_image_handler...")

        device_info = {
            'platform': 'mqtt_vacuum_camera',
            'unique_id': 'rockrobo_camera',
            'vacuum_config_entry': '9abe4e81972b00d1682c2363d3584041',
            'vacuum_map': 'valetudo/rockrobo',
            'vacuum_identifiers': {('mqtt', 'rockrobo')},
            'coordinator': "<custom_components.mqtt_vacuum_camera.coordinator.MQTTVacuumCoordinator object>",
            'is_rand256': True,
            'unsub_options_update_listener': "<function ConfigEntry.add_update_listener.<locals>.<lambda>>",
            'alpha_background': 255.0,
            'alpha_charger': 255.0,
            'alpha_go_to': 255.0,
            'alpha_move': 100.0,
            'alpha_no_go': 125.0,
            'alpha_robot': 255.0,
            'alpha_text': 255.0,
            'alpha_wall': 115.0,  # Testing with a lower alpha value
            'alpha_zone_clean': 125.0,
            'aspect_ratio': '16, 9',
            'auto_zoom': True,
            'image_auto_zoom': True,  # Explicitly set image_auto_zoom to True
            'zoom_lock_ratio': True,
            'color_background': [0, 125, 255],
            'color_charger': [255, 128, 0],
            'color_go_to': [0, 255, 0],
            'color_move': [238, 247, 255],
            'color_no_go': [255, 0, 0],
            'color_robot': [255, 255, 204],
            'color_text': [164, 25, 25],
            'color_wall': [255, 255, 0],
            'color_zone_clean': [255, 255, 255],
            'color_room_0': [135, 206, 250],
            'color_room_1': [176, 226, 255],
            'color_room_2': [165, 141, 18],
            'color_room_3': [164, 211, 238],
            'color_room_4': [141, 182, 205],
            'color_room_5': [96, 123, 139],
            'color_room_6': [224, 255, 255],
            'color_room_7': [209, 238, 238],
            'color_room_8': [180, 205, 205],
            'color_room_9': [122, 139, 139],
            'color_room_10': [175, 238, 238],
            'color_room_11': [84, 153, 199],
            'color_room_12': [133, 193, 233],
            'color_room_13': [245, 176, 65],
            'color_room_14': [82, 190, 128],
            'color_room_15': [72, 201, 176],
            'alpha_room_0': 255.0,
            'alpha_room_1': 255.0,
            'alpha_room_2': 255.0,
            'alpha_room_3': 255.0,
            'alpha_room_4': 255.0,
            'alpha_room_5': 255.0,
            'alpha_room_6': 255.0,
            'alpha_room_7': 255.0,
            'alpha_room_8': 255.0,
            'alpha_room_9': 255.0,
            'alpha_room_10': 255.0,
            'alpha_room_11': 255.0,
            'alpha_room_12': 255.0,
            'alpha_room_13': 255.0,
            'alpha_room_14': 255.0,
            'alpha_room_15': 255.0,
            'offset_top': 0,
            'offset_bottom': 0,
            'offset_left': 10,
            'offset_right': 0,
            'rotate_image': '90',
            'margins': '100',
            'show_vac_status': False,
            'vac_status_font': 'custom_components/mqtt_vacuum_camera/utils/fonts/FiraSans.ttf',
            'vac_status_position': True,
            'vac_status_size': 50.0,
            'enable_www_snapshots': False,
            'get_svg_file': False,
            'trims_data': {
                'trim_left': 0,
                'trim_up': 0,
                'trim_right': 0,
                'trim_down': 0
            },
            'disable_floor': False,
            'disable_wall': False,
            'disable_robot': False,
            'disable_charger': False,
            'disable_virtual_walls': False,
            'disable_restricted_areas': False,
            'disable_no_mop_areas': False,
            'disable_obstacles': False,
            'disable_path': False,
            'disable_predicted_path': False,
            'disable_go_to_target': False,
            'disable_room_1': False,
            'disable_room_2': False,
            'disable_room_3': False,
            'disable_room_4': False,
            'disable_room_5': False,
            'disable_room_6': False,
            'disable_room_7': False,
            'disable_room_8': False,
            'disable_room_9': False,
            'disable_room_10': False,
            'disable_room_11': False,
            'disable_room_12': False,
            'disable_room_13': False,
            'disable_room_14': False,
            'disable_room_15': False
        }

        shared_data = CameraSharedManager("test_vacuum", device_info)
        shared = shared_data.get_instance()
        shared.vacuum_state = "cleaning"

        # The room IDs in the test data are 16-20, but the handler uses an internal ID (0-4)
        # We need to set up the active zones array to match the internal IDs
        # Test with Living Room as active zone - zoom should be enabled
        shared.rand256_active_zone = [0] * 5  # Disable all rooms first
        shared.rand256_active_zone[4] = 1  # Enable Living Room (array index 4, segment ID 16)

        _LOGGER.debug(f"Shared instance trims: {shared.trims}")

        colors = ColorsManagement(shared)
        colors.set_initial_colours(device_info)

        _LOGGER.debug(f"Colors initialized: {shared.user_colors}")

        # Create a handler instance
        handler = ReImageHandler(shared)

        # Try to generate an image from the JSON data
        try:
            _LOGGER.info("Attempting to generate image from JSON data...")
            self.image = await handler.get_image_from_rrm(self.test_data)
            _LOGGER.info("Successfully generated image from JSON data")
            if self.image is None:
                _LOGGER.error("Failed to generate image from JSON data")
                return
        except Exception as e:
            _LOGGER.warning(f"Error generating image from JSON: {e}")

        # Display image size and other properties
        _LOGGER.info(f"Image size: {self.image.size}")
        _LOGGER.info(f"Trims update: {shared.trims.to_dict()}")
        _LOGGER.info(f"Calibration_data: {handler.get_calibration_data()}")
        _LOGGER.info(await handler.get_rooms_attributes({
            "spots":[{"name":"test_point","coordinates":[25566,27289]}],
            "zones":[{"name":"test_zone","coordinates":[[20809,25919,22557,26582,1]]}],
            "rooms":[{"name":"Bathroom","id":19},
                     {"name":"Bedroom","id":20},
                     {"name":"Entrance","id":18},
                     {"name":"Kitchen","id":17},
                     {"name":"Living Room","id":16}],
                     "updated":1746298038728})
                     )
        # Get the robot room detection result
        robot_room_result = await handler.async_get_robot_in_room(25400, 25580, 0)
        _LOGGER.info(f"Robot in room: {robot_room_result}")

        # Check if the robot is in a room
        if robot_room_result and "in_room" in robot_room_result:
            room_name = robot_room_result["in_room"]
            _LOGGER.info(f"Robot is in room: {room_name}")

            # Check if the room has an ID in the robot_in_room property
            if handler.robot_in_room and "id" in handler.robot_in_room:
                room_id = handler.robot_in_room["id"]
                _LOGGER.info(f"Room ID: {room_id}")

                # Make sure the active zones array has enough elements
                if len(handler.shared.rand256_active_zone) > room_id:
                    _LOGGER.info(f"Active zone for room {room_id}: {handler.shared.rand256_active_zone[room_id]}")
                else:
                    _LOGGER.info(f"Room ID {room_id} is out of range for active zones array (length: {len(handler.shared.rand256_active_zone)})")
            else:
                _LOGGER.info("Robot is in a room but room_id is not set")
        else:
            _LOGGER.info("Robot is not in any room")

        # Check if image_auto_zoom is set correctly
        _LOGGER.info(f"image_auto_zoom: {handler.shared.image_auto_zoom}")

        # Check if all zooming conditions are met
        _LOGGER.info(f"Zoom conditions: zoom={handler.zooming}, vacuum_state={handler.shared.vacuum_state}, image_auto_zoom={handler.shared.image_auto_zoom}")

        _LOGGER.info(f"Zooming enabled: {handler.zooming}")
        # Show the image
        self.image.show()


def __main__():
    test = TestRandImageHandler()
    test.setUp()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        loop.run_until_complete(test.test_image_handler())
    finally:
        profiler.disable()
        loop.close()

        # Save profiling data
        profile_output = "profile_output_rand.prof"
        profiler.dump_stats(profile_output)

        # Print profiling summary
        stats = pstats.Stats(profile_output)
        stats.strip_dirs().sort_stats("cumulative").print_stats(50)  # Show top 50 functions


if __name__ == "__main__":
    __main__()
