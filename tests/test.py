from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import cProfile
import pstats
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from SCR package
from SCR.valetudo_map_parser.config.colors_man import ColorsManagment
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.config.types import RoomStore
from SCR.valetudo_map_parser.hypfer_handler import HypferMapImageHandler
from tests.element_map_visualizer import ElementMapVisualizer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Ensures DEBUG logs are displayed
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s (line %(lineno)d) - %(message)s",
)

_LOGGER = logging.getLogger(__name__)


class TestImageHandler:
    def __init__(self):
        self.test_data = None
        self.image = None

    def setUp(self):
        # Load the test.json file
        test_file_path = os.path.join(os.path.dirname(__file__), "test.json")
        _LOGGER.info(f"Loading test data from {test_file_path}")

        with open(test_file_path, "r") as file:
            self.test_data = json.load(file)
            _LOGGER.debug("Test data loaded.")

    async def test_image_handler(self):
        _LOGGER.info("Starting test_image_handler...")

        device_info = {
            'platform': 'mqtt_vacuum_camera',
            'unique_id': 'rockrobo_camera',
            'vacuum_config_entry': '9abe4e81972b00d1682c2363d3584041',
            'vacuum_map': 'valetudo/rockrobo',
            'vacuum_identifiers': {('mqtt', 'rockrobo')},
            'coordinator': "<custom_components.mqtt_vacuum_camera.coordinator.MQTTVacuumCoordinator object>",
            'is_rand256': False,
            'unsub_options_update_listener': "<function ConfigEntry.add_update_listener.<locals>.<lambda>>",
            'alpha_background': 255.0,
            'alpha_charger': 255.0,
            'alpha_go_to': 255.0,
            'alpha_move': 200.0,  # Higher alpha for better visibility
            'alpha_no_go': 125.0,
            'alpha_robot': 255.0,
            'alpha_text': 255.0,
            'alpha_wall': 150.0,  # Testing with a lower alpha value
            'alpha_zone_clean': 125.0,
            'aspect_ratio': '16, 9',
            'auto_zoom': True,
            'zoom_lock_ratio': True,
            'color_background': [0, 125, 255],
            'color_charger': [255, 128, 0],
            'color_go_to': [0, 255, 0],
            'color_move': [50, 150, 255],  # More vibrant blue for better visibility
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
            'rotate_image': '180',
            'margins': '100',
            'show_vac_status': False,
            'vac_status_font': 'custom_components/mqtt_vacuum_camera/utils/fonts/FiraSans.ttf',
            'vac_status_position': True,
            'vac_status_size': 50.0,
            'enable_www_snapshots': False,
            'get_svg_file': False,
            'trims_data': {
                'trim_left': 1980,
                'trim_up': 1650,
                'trim_right': 3974,
                'trim_down': 3474
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
        shared.vacuum_state = "docked"
        _LOGGER.debug(f"Shared instance trims: {shared.trims}")

        colors = ColorsManagment(shared)
        colors.set_initial_colours(device_info)

        _LOGGER.debug(f"Colors initialized: {shared.user_colors}")

        handler = HypferMapImageHandler(shared)

        # Get the image with elements disabled from device_info
        self.image = await handler.async_get_image_from_json(self.test_data)
        if self.image is None:
            _LOGGER.error("Failed to generate image from JSON data")
            return

        _LOGGER.info(f"Calibration_data: {handler.get_calibration_data()}")
        _LOGGER.info(f"Image size: {self.image.size}")
        store = RoomStore("test_vacuum")
        rooms_data = await handler.async_get_rooms_attributes()
        _LOGGER.info(f"Room Properties: {rooms_data}")
        count = store.get_rooms_count()
        _LOGGER.info(f"Room Store Properties: {count}")
        _LOGGER.info(f"Trims update: {shared.trims.to_dict()}")

        # Check if element_map exists and what it contains
        if hasattr(handler.shared, 'element_map'):
            _LOGGER.info(f"Element map exists: {type(handler.shared.element_map)}")
            if handler.shared.element_map is not None:
                _LOGGER.info(f"Element map shape: {handler.shared.element_map.shape}")
                _LOGGER.info(f"Element map dtype: {handler.shared.element_map.dtype}")
                _LOGGER.info(f"Element map min: {handler.shared.element_map.min()}, max: {handler.shared.element_map.max()}")
                # Analyze the element map in more detail
                element_map = handler.shared.element_map
                _LOGGER.info(f"Element map shape: {element_map.shape}")
                _LOGGER.info(f"Element map non-zero count: {np.count_nonzero(element_map)}")
                _LOGGER.info(f"Element map zero count: {np.size(element_map) - np.count_nonzero(element_map)}")
                _LOGGER.info(f"Element map percentage non-zero: {np.count_nonzero(element_map) / np.size(element_map) * 100:.2f}%")

                # Visualize the element map
                # ElementMapVisualizer.visualize_element_map(element_map, "element_map_hypfer.png")
            else:
                _LOGGER.warning("Element map is None")
        else:
            _LOGGER.warning("No element_map attribute in handler.shared")
        _LOGGER.info("Element map visualization complete")


        self.image.show()


def __main__():
    test = TestImageHandler()
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
        profile_output = "hypfer_profile_output.prof"
        profiler.dump_stats(profile_output)

        # Print profiling summary
        stats = pstats.Stats(profile_output)
        stats.strip_dirs().sort_stats("cumulative").print_stats(50)  # Show top 50 functions


if __name__ == "__main__":
    __main__()
