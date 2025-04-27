from __future__ import annotations

import asyncio
import json
import logging
import os
import cProfile
import pstats

from SCR.valetudo_map_parser.config.colors_man import ColorsManagment
from SCR.valetudo_map_parser.config.drawable_elements import DrawableElement
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.config.types import RoomStore
from SCR.valetudo_map_parser.hypfer_handler import HypferMapImageHandler

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
            'is_rand256': True,
            'unsub_options_update_listener': "<function ConfigEntry.add_update_listener.<locals>.<lambda>>",
            'alpha_background': 255.0,
            'alpha_charger': 255.0,
            'alpha_go_to': 255.0,
            'alpha_move': 150.0,
            'alpha_no_go': 125.0,
            'alpha_robot': 255.0,
            'alpha_text': 255.0,
            'alpha_wall': 115.0,  # Testing with a lower alpha value
            'alpha_zone_clean': 125.0,
            'aspect_ratio': '16, 9',
            'auto_zoom': True,
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
            # Element visibility controls
            # Base elements
            'disable_floor': False,           # Show floor
            'disable_wall': False,            # Show walls
            'disable_robot': False,           # Show robot
            'disable_charger': False,         # Show charger
            'disable_virtual_walls': False,   # Show virtual walls
            'disable_restricted_areas': False, # Show restricted areas
            'disable_no_mop_areas': False,    # Show no-mop areas
            'disable_obstacles': True,        # Hide obstacles
            'disable_path': False,             # Hide path
            'disable_predicted_path': False,  # Show predicted path
            'disable_go_to_target': False,    # Show go-to target

            # Room visibility (all rooms visible by default)
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

        # Demonstrate the drawable element selection system
        _LOGGER.info("Testing drawable element selection system...")

        # First, check which elements are disabled based on device_info
        _LOGGER.info("Checking elements disabled from device_info:")
        _LOGGER.info(f"PATH element enabled: {handler.drawing_config.is_enabled(DrawableElement.PATH)}")
        _LOGGER.info(f"OBSTACLE element enabled: {handler.drawing_config.is_enabled(DrawableElement.OBSTACLE)}")

        # Get the image with elements disabled from device_info
        self.image = await handler.async_get_image_from_json(self.test_data)
        _LOGGER.info("Created image with elements disabled from device_info")

        # Show what elements can be disabled
        _LOGGER.info("\nAvailable elements that can be disabled:")
        for element in DrawableElement:
            _LOGGER.info(f"  - {element.name}: {element.value} (enabled: {handler.drawing_config.is_enabled(element)})")

        # Example 1: Re-enable path to see the difference
        _LOGGER.info("\nExample 1: Re-enabling path to see the difference")

        # Enable the path
        handler.enable_element(DrawableElement.PATH)
        _LOGGER.info(f"PATH element enabled: {handler.drawing_config.is_enabled(DrawableElement.PATH)}")

        # Verify that the PATH element is enabled
        if not handler.drawing_config.is_enabled(DrawableElement.PATH):
            _LOGGER.warning("PATH element is still disabled after calling enable_element!")
            # Force enable it
            handler.drawing_config._enabled_elements[DrawableElement.PATH] = True
            _LOGGER.info(f"Forced PATH element to be enabled: {handler.drawing_config.is_enabled(DrawableElement.PATH)}")

        # Get a new image with path enabled
        self.image = await handler.async_get_image_from_json(self.test_data)
        _LOGGER.info("Created image with path re-enabled")

        # Disable the path again to match device_info
        handler.disable_element(DrawableElement.PATH)
        _LOGGER.info(f"PATH element disabled again: {not handler.drawing_config.is_enabled(DrawableElement.PATH)}")

        # Example 2: Disable all room segments except room 1
        _LOGGER.info("\nExample 2: Disabling multiple room segments")
        # First re-enable everything
        for element in DrawableElement:
            handler.enable_element(element)

        # Then disable rooms 2-15
        for room_id in range(2, 16):
            room_element = getattr(DrawableElement, f"ROOM_{room_id}")
            handler.disable_element(room_element)
        _LOGGER.info("Disabled all rooms except Room 1")

        # Get a new image with only room 1 visible
        self.image = await handler.async_get_image_from_json(self.test_data)
        _LOGGER.info("Created image with only Room 1 visible")

        # Example 3: Change the color of a specific element
        _LOGGER.info("\nExample 3: Changing element properties")
        # First re-enable everything
        for element in DrawableElement:
            handler.enable_element(element)

        # Change the color of the robot
        handler.set_element_property(DrawableElement.ROBOT, "color", (255, 0, 0, 255))  # Bright red
        _LOGGER.info("Changed robot color to bright red")

        # Get a new image with the red robot
        self.image = await handler.async_get_image_from_json(self.test_data)
        _LOGGER.info("Created image with red robot")

        # Restore all elements and default properties for final image
        _LOGGER.info("\nRestoring all elements to default")
        # Re-initialize the handler to reset all properties
        handler = HypferMapImageHandler(shared)
        self.image = await handler.async_get_image_from_json(self.test_data)

        _LOGGER.info(f"Calibration_data: {handler.get_calibration_data()}")
        _LOGGER.info(f"Image size: {self.image.size}")
        store = RoomStore("test_vacuum")
        rooms_data = await handler.async_get_rooms_attributes()
        _LOGGER.info(f"Room Properties: {rooms_data}")
        count = store.get_rooms_count()
        _LOGGER.info(f"Room Store Properties: {count}")
        _LOGGER.info(f"Trims update: {shared.trims.to_dict()}")
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
        profile_output = "profile_output.prof"
        profiler.dump_stats(profile_output)

        # Print profiling summary
        stats = pstats.Stats(profile_output)
        stats.strip_dirs().sort_stats("cumulative").print_stats(50)  # Show top 50 functions


if __name__ == "__main__":
    __main__()
