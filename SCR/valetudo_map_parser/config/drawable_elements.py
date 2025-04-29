"""
Drawable Elements Configuration.
Defines the elements that can be drawn on the map and their properties.
Version: 0.1.9
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Tuple, Union
import numpy as np
from .types import LOGGER

from .colors import DefaultColors, SupportedColor

# Type aliases
Color = Tuple[int, int, int, int]  # RGBA color
PropertyDict = Dict[str, Union[Color, float, int]]


class DrawableElement(IntEnum):
    """Enumeration of drawable map elements with unique integer codes."""

    # Base elements
    FLOOR = 1
    WALL = 2
    ROBOT = 3
    CHARGER = 4
    VIRTUAL_WALL = 5
    RESTRICTED_AREA = 6
    NO_MOP_AREA = 7
    OBSTACLE = 8
    PATH = 9
    PREDICTED_PATH = 10
    GO_TO_TARGET = 11

    # Rooms (101-115 for up to 15 rooms)
    ROOM_1 = 101
    ROOM_2 = 102
    ROOM_3 = 103
    ROOM_4 = 104
    ROOM_5 = 105
    ROOM_6 = 106
    ROOM_7 = 107
    ROOM_8 = 108
    ROOM_9 = 109
    ROOM_10 = 110
    ROOM_11 = 111
    ROOM_12 = 112
    ROOM_13 = 113
    ROOM_14 = 114
    ROOM_15 = 115


class DrawingConfig:
    """Configuration for which elements to draw and their properties."""

    def __init__(self):
        """Initialize with all elements enabled by default."""
        # Dictionary of element_code -> enabled status
        self._enabled_elements = {element: True for element in DrawableElement}

        # Dictionary of element_code -> drawing properties (color, opacity, etc.)
        self._element_properties: Dict[DrawableElement, PropertyDict] = {}

        # Initialize default properties
        self._set_default_properties()

    def _set_default_properties(self):
        """Set default drawing properties for each element."""
        # Set properties for rooms using DefaultColors
        for room_id in range(1, 16):
            room_element = getattr(DrawableElement, f"ROOM_{room_id}")
            room_key = SupportedColor.room_key(room_id - 1)
            rgb = DefaultColors.DEFAULT_ROOM_COLORS.get(room_key, (135, 206, 250))
            alpha = DefaultColors.DEFAULT_ALPHA.get(f"alpha_room_{room_id - 1}", 255.0)

            self._element_properties[room_element] = {
                "color": (*rgb, int(alpha)),
                "opacity": alpha / 255.0,
                "z_index": 10,  # Drawing order
            }

        # Map DrawableElement to SupportedColor
        element_color_mapping = {
            DrawableElement.FLOOR: SupportedColor.MAP_BACKGROUND,
            DrawableElement.WALL: SupportedColor.WALLS,
            DrawableElement.ROBOT: SupportedColor.ROBOT,
            DrawableElement.CHARGER: SupportedColor.CHARGER,
            DrawableElement.VIRTUAL_WALL: SupportedColor.NO_GO,
            DrawableElement.RESTRICTED_AREA: SupportedColor.NO_GO,
            DrawableElement.PATH: SupportedColor.PATH,
            DrawableElement.PREDICTED_PATH: SupportedColor.PREDICTED_PATH,
            DrawableElement.GO_TO_TARGET: SupportedColor.GO_TO,
            DrawableElement.NO_MOP_AREA: SupportedColor.NO_GO,  # Using NO_GO for no-mop areas
            DrawableElement.OBSTACLE: SupportedColor.NO_GO,  # Using NO_GO for obstacles
        }

        # Set z-index for each element type
        z_indices = {
            DrawableElement.FLOOR: 0,
            DrawableElement.WALL: 20,
            DrawableElement.ROBOT: 50,
            DrawableElement.CHARGER: 40,
            DrawableElement.VIRTUAL_WALL: 30,
            DrawableElement.RESTRICTED_AREA: 25,
            DrawableElement.NO_MOP_AREA: 25,
            DrawableElement.OBSTACLE: 15,
            DrawableElement.PATH: 35,
            DrawableElement.PREDICTED_PATH: 34,
            DrawableElement.GO_TO_TARGET: 45,
        }

        # Set properties for other elements using DefaultColors
        for element, color_key in element_color_mapping.items():
            rgb = DefaultColors.COLORS_RGB.get(color_key, (0, 0, 0))
            alpha_key = f"alpha_{color_key}"
            alpha = DefaultColors.DEFAULT_ALPHA.get(alpha_key, 255.0)

            # Special case for semi-transparent elements
            if element in [
                DrawableElement.RESTRICTED_AREA,
                DrawableElement.NO_MOP_AREA,
                DrawableElement.PREDICTED_PATH,
            ]:
                alpha = 125.0  # Semi-transparent by default

            self._element_properties[element] = {
                "color": (*rgb, int(alpha)),
                "opacity": alpha / 255.0,
                "z_index": z_indices.get(element, 0),
            }

    def enable_element(self, element_code: DrawableElement) -> None:
        """Enable drawing of a specific element."""
        if element_code in self._enabled_elements:
            self._enabled_elements[element_code] = True
            LOGGER.info(
                "Enabled element %s (%s)", element_code.name, element_code.value
            )
            LOGGER.info(
                "Element %s is now enabled: %s",
                element_code.name,
                self._enabled_elements[element_code],
            )

    def disable_element(self, element_code: DrawableElement) -> None:
        """Disable drawing of a specific element."""
        if element_code in self._enabled_elements:
            self._enabled_elements[element_code] = False
            LOGGER.info(
                "Disabled element %s (%s)", element_code.name, element_code.value
            )
            LOGGER.info(
                "Element %s is now enabled: %s",
                element_code.name,
                self._enabled_elements[element_code],
            )

    def set_elements(self, element_codes: List[DrawableElement]) -> None:
        """Enable only the specified elements, disable all others."""
        # First disable all
        for element in self._enabled_elements:
            self._enabled_elements[element] = False

        # Then enable specified ones
        for element in element_codes:
            if element in self._enabled_elements:
                self._enabled_elements[element] = True

    def is_enabled(self, element_code: DrawableElement) -> bool:
        """Check if an element is enabled for drawing."""
        enabled = self._enabled_elements.get(element_code, False)
        LOGGER.debug(
            "Checking if element %s is enabled: %s",
            element_code.name if hasattr(element_code, "name") else element_code,
            enabled,
        )
        return enabled

    def set_property(
        self, element_code: DrawableElement, property_name: str, value
    ) -> None:
        """Set a drawing property for an element."""
        if element_code in self._element_properties:
            self._element_properties[element_code][property_name] = value

    def get_property(
        self, element_code: DrawableElement, property_name: str, default=None
    ):
        """Get a drawing property for an element."""
        if element_code in self._element_properties:
            return self._element_properties[element_code].get(property_name, default)
        return default

    def get_enabled_elements(self) -> List[DrawableElement]:
        """Get list of enabled element codes."""
        return [
            element for element, enabled in self._enabled_elements.items() if enabled
        ]

    def get_drawing_order(self) -> List[DrawableElement]:
        """Get list of enabled elements in drawing order (by z_index)."""
        enabled = self.get_enabled_elements()
        return sorted(enabled, key=lambda e: self.get_property(e, "z_index", 0))

    def update_from_device_info(self, device_info: dict) -> None:
        """Update configuration based on device info dictionary."""
        # Map DrawableElement to SupportedColor
        element_color_mapping = {
            DrawableElement.FLOOR: SupportedColor.MAP_BACKGROUND,
            DrawableElement.WALL: SupportedColor.WALLS,
            DrawableElement.ROBOT: SupportedColor.ROBOT,
            DrawableElement.CHARGER: SupportedColor.CHARGER,
            DrawableElement.VIRTUAL_WALL: SupportedColor.NO_GO,
            DrawableElement.RESTRICTED_AREA: SupportedColor.NO_GO,
            DrawableElement.PATH: SupportedColor.PATH,
            DrawableElement.PREDICTED_PATH: SupportedColor.PREDICTED_PATH,
            DrawableElement.GO_TO_TARGET: SupportedColor.GO_TO,
            DrawableElement.NO_MOP_AREA: SupportedColor.NO_GO,
            DrawableElement.OBSTACLE: SupportedColor.NO_GO,
        }

        # Update room colors from device info
        for room_id in range(1, 16):
            room_element = getattr(DrawableElement, f"ROOM_{room_id}")
            color_key = SupportedColor.room_key(room_id - 1)
            alpha_key = f"alpha_room_{room_id - 1}"

            if color_key in device_info:
                rgb = device_info[color_key]
                alpha = device_info.get(alpha_key, 255.0)

                # Create RGBA color
                rgba = (*rgb, int(alpha))

                # Update color and opacity
                self.set_property(room_element, "color", rgba)
                self.set_property(room_element, "opacity", alpha / 255.0)

                LOGGER.debug(
                    "Updated room %d color to %s with alpha %s", room_id, rgb, alpha
                )

        # Update other element colors
        for element, color_key in element_color_mapping.items():
            if color_key in device_info:
                rgb = device_info[color_key]
                alpha_key = f"alpha_{color_key}"
                alpha = device_info.get(alpha_key, 255.0)

                # Special case for semi-transparent elements
                if element in [
                    DrawableElement.RESTRICTED_AREA,
                    DrawableElement.NO_MOP_AREA,
                    DrawableElement.PREDICTED_PATH,
                ]:
                    if alpha > 200:  # If alpha is too high for these elements
                        alpha = 125.0  # Use a more appropriate default

                # Create RGBA color
                rgba = (*rgb, int(alpha))

                # Update color and opacity
                self.set_property(element, "color", rgba)
                self.set_property(element, "opacity", alpha / 255.0)

                LOGGER.debug(
                    "Updated element %s color to %s with alpha %s",
                    element.name,
                    rgb,
                    alpha,
                )

        # Check for disabled elements using specific boolean flags
        # Map element disable flags to DrawableElement enum values
        element_disable_mapping = {
            "disable_floor": DrawableElement.FLOOR,
            "disable_wall": DrawableElement.WALL,
            "disable_robot": DrawableElement.ROBOT,
            "disable_charger": DrawableElement.CHARGER,
            "disable_virtual_walls": DrawableElement.VIRTUAL_WALL,
            "disable_restricted_areas": DrawableElement.RESTRICTED_AREA,
            "disable_no_mop_areas": DrawableElement.NO_MOP_AREA,
            "disable_obstacles": DrawableElement.OBSTACLE,
            "disable_path": DrawableElement.PATH,
            "disable_predicted_path": DrawableElement.PREDICTED_PATH,
            "disable_go_to_target": DrawableElement.GO_TO_TARGET,
        }

        # Process base element disable flags
        for disable_key, element in element_disable_mapping.items():
            if device_info.get(disable_key, False):
                self.disable_element(element)
                LOGGER.info(
                    "Disabled %s element from device_info setting", element.name
                )

        # Process room disable flags (1-15)
        for room_id in range(1, 16):
            disable_key = f"disable_room_{room_id}"
            if device_info.get(disable_key, False):
                room_element = getattr(DrawableElement, f"ROOM_{room_id}")
                self.disable_element(room_element)
                LOGGER.info(
                    "Disabled ROOM_%d element from device_info setting", room_id
                )


class ElementMapGenerator:
    """Class for generating 2D element maps from JSON data.

    This class creates a 2D array where each cell contains an integer code
    representing the element at that position (floor, wall, room, etc.).
    It focuses only on the static structure (rooms and walls).
    """

    def __init__(self, drawing_config: DrawingConfig = None, shared_data=None):
        """Initialize the element map generator.

        Args:
            drawing_config: Optional drawing configuration for element properties
        """
        self.drawing_config = drawing_config or DrawingConfig()
        self.shared = shared_data
        self.element_map = None

    async def async_generate_from_json(self, json_data, existing_element_map=None):
        """Generate a 2D element map from JSON data without visual rendering.

        Args:
            json_data: The JSON data from the vacuum
            existing_element_map: Optional pre-created element map to populate

        Returns:
            numpy.ndarray: The 2D element map array
        """
        if not self.shared:
            LOGGER.warning("Shared data not provided, some features may not work.")
            return None

        # Use existing element map if provided
        if existing_element_map is not None:
            self.element_map = existing_element_map

        # Check if this is a Valetudo map or a Rand256 map
        is_valetudo = "size" in json_data and "pixelSize" in json_data and "layers" in json_data
        is_rand256 = "image" in json_data and "map_data" in json_data

        # Debug logging
        LOGGER.debug(f"JSON data keys: {list(json_data.keys())}")
        LOGGER.debug(f"Is Valetudo: {is_valetudo}, Is Rand256: {is_rand256}")

        # Create element map if not provided
        if self.element_map is None:
            if is_valetudo:
                # Get map dimensions from Valetudo map
                map_size = json_data.get("size", 0)
                if isinstance(map_size, dict):
                    # If map_size is a dictionary, extract the values
                    size_x = map_size.get("width", 0)
                    size_y = map_size.get("height", 0)
                else:
                    # If map_size is a number, use it for both dimensions
                    pixel_size = json_data.get("pixelSize", 5)  # Default to 5mm per pixel
                    size_x = int(map_size // pixel_size)
                    size_y = int(map_size // pixel_size)
                self.element_map = np.zeros((size_y, size_x), dtype=np.int32)
                self.element_map[:] = DrawableElement.FLOOR
            elif is_rand256:
                # Get map dimensions from Rand256 map
                map_data = json_data.get("map_data", {})
                size_x = map_data.get("width", 0)
                size_y = map_data.get("height", 0)
                self.element_map = np.zeros((size_y, size_x), dtype=np.int32)

        if not (is_valetudo or is_rand256):
            LOGGER.error("Unknown JSON format, cannot generate element map")
            return None

        if is_valetudo:
            # Get map dimensions from the Valetudo JSON data
            size_x = json_data["size"]["x"]
            size_y = json_data["size"]["y"]
            pixel_size = json_data["pixelSize"]

            # Calculate scale factor based on pixel size (normalize to 5mm standard)
            # This helps handle maps with different scales
            scale_factor = pixel_size if pixel_size != 0 else 1.0
            LOGGER.info(f"Map dimensions: {size_x}x{size_y}, pixel size: {pixel_size}mm, scale factor: {scale_factor:.2f}")

            # Ensure element_map is properly initialized with the correct dimensions
            if self.element_map is None or self.element_map.shape[0] == 0 or self.element_map.shape[1] == 0:
                # For now, create a full-sized element map to ensure coordinates match
                # We'll resize it at the end for efficiency
                map_width = int(size_x // pixel_size) if pixel_size != 0 else size_x
                map_height = int(size_y // pixel_size) if pixel_size != 0 else size_y

                LOGGER.info(f"Creating element map with dimensions: {map_width}x{map_height}")
                self.element_map = np.zeros((map_height, map_width), dtype=np.int32)
                self.element_map[:] = DrawableElement.FLOOR

            # Process layers (rooms, walls, etc.)
            for layer in json_data["layers"]:
                layer_type = layer["type"]

                # Process rooms (segments)
                if layer_type == "segment":
                    # Handle different segment formats
                    if "segments" in layer:
                        segments = layer["segments"]
                    else:
                        segments = [layer]  # Some formats have segment data directly in the layer

                    for segment in segments:
                        # Get room ID and check if it's enabled
                        if "id" in segment:
                            room_id = segment["id"]
                            room_id_int = int(room_id)
                        elif "metaData" in segment and "segmentId" in segment["metaData"]:
                            # Handle Hypfer format
                            room_id = segment["metaData"]["segmentId"]
                            room_id_int = int(room_id)
                        else:
                            # Skip segments without ID
                            continue

                        # Skip if room is disabled
                        room_element = getattr(DrawableElement, f"ROOM_{room_id_int}", None)
                        if room_element is None or not self.drawing_config.is_enabled(room_element):
                            continue

                        # Room element code was already retrieved above
                        if room_element is not None:
                            # Process pixels for this room
                            if "pixels" in segment and segment["pixels"]:
                                # Regular pixel format
                                for x, y, z in segment["pixels"]:
                                    # Calculate pixel coordinates
                                    col = x * pixel_size
                                    row = y * pixel_size

                                    # Fill the element map with room code
                                    for i in range(z):
                                        # Get the region to update
                                        region_col_start = col + i * pixel_size
                                        region_col_end = col + (i + 1) * pixel_size
                                        region_row_start = row
                                        region_row_end = row + pixel_size

                                        # Update element map for this region
                                        if region_row_start < size_y and region_col_start < size_x:
                                            # Ensure we stay within bounds
                                            end_row = min(region_row_end, size_y)
                                            end_col = min(region_col_end, size_x)

                                            # Set element code for this region
                                            # Only set pixels that are not already set (floor is 1)
                                            region = self.element_map[region_row_start:end_row, region_col_start:end_col]
                                            mask = region == DrawableElement.FLOOR
                                            region[mask] = room_element

                            elif "compressedPixels" in segment and segment["compressedPixels"]:
                                # Compressed pixel format (used in Valetudo)
                                compressed_pixels = segment["compressedPixels"]
                                i = 0
                                pixel_count = 0

                                while i < len(compressed_pixels):
                                    x = compressed_pixels[i]
                                    y = compressed_pixels[i+1]
                                    count = compressed_pixels[i+2]
                                    pixel_count += count

                                    # Set element code for this run of pixels
                                    for j in range(count):
                                        px = x + j
                                        if 0 <= px < size_x and 0 <= y < size_y:
                                            self.element_map[y, px] = room_element

                                    i += 3

                                # Debug: Log that we're adding room pixels
                                LOGGER.info(f"Adding room {room_id_int} pixels to element map with code {room_element}")
                                LOGGER.info(f"Room {room_id_int} has {len(compressed_pixels)//3} compressed runs with {pixel_count} total pixels")

                # Process walls
                elif layer_type == "wall":
                    # Skip if walls are disabled
                    if not self.drawing_config.is_enabled(DrawableElement.WALL):
                        continue

                    # Process wall pixels
                    if "pixels" in layer and layer["pixels"]:
                        # Regular pixel format
                        for x, y, z in layer["pixels"]:
                            # Calculate pixel coordinates
                            col = x * pixel_size
                            row = y * pixel_size

                            # Fill the element map with wall code
                            for i in range(z):
                                # Get the region to update
                                region_col_start = col + i * pixel_size
                                region_col_end = col + (i + 1) * pixel_size
                                region_row_start = row
                                region_row_end = row + pixel_size

                                # Update element map for this region
                                if region_row_start < size_y and region_col_start < size_x:
                                    # Ensure we stay within bounds
                                    end_row = min(region_row_end, size_y)
                                    end_col = min(region_col_end, size_x)

                                    # Set element code for this region
                                    # Only set pixels that are not already set (floor is 1)
                                    region = self.element_map[region_row_start:end_row, region_col_start:end_col]
                                    mask = region == DrawableElement.FLOOR
                                    region[mask] = DrawableElement.WALL

                    elif "compressedPixels" in layer and layer["compressedPixels"]:
                        # Compressed pixel format (used in Valetudo)
                        compressed_pixels = layer["compressedPixels"]
                        i = 0
                        pixel_count = 0

                        while i < len(compressed_pixels):
                            x = compressed_pixels[i]
                            y = compressed_pixels[i+1]
                            count = compressed_pixels[i+2]
                            pixel_count += count

                            # Set element code for this run of pixels
                            for j in range(count):
                                px = x + j
                                if 0 <= px < size_x and 0 <= y < size_y:
                                    self.element_map[y, px] = DrawableElement.WALL

                            i += 3

                        # Debug: Log that we're adding wall pixels
                        LOGGER.info(f"Adding wall pixels to element map with code {DrawableElement.WALL}")
                        LOGGER.info(f"Wall layer has {len(compressed_pixels)//3} compressed runs with {pixel_count} total pixels")

        elif is_rand256:
            # Get map dimensions from the Rand256 JSON data
            map_data = json_data["map_data"]
            size_x = map_data["dimensions"]["width"]
            size_y = map_data["dimensions"]["height"]

            # Create empty element map initialized with floor
            self.element_map = np.zeros((size_y, size_x), dtype=np.int32)
            self.element_map[:] = DrawableElement.FLOOR

            # Process rooms
            if "rooms" in map_data and map_data["rooms"]:
                for room in map_data["rooms"]:
                    # Get room ID and check if it's enabled
                    room_id_int = room["id"]

                    # Skip if room is disabled
                    if not self.drawing_config.is_enabled(f"ROOM_{room_id_int}"):
                        continue

                    # Get room element code (ROOM_1, ROOM_2, etc.)
                    room_element = None
                    if 0 < room_id_int <= 15:
                        room_element = getattr(DrawableElement, f"ROOM_{room_id_int}", None)

                    if room_element is not None and "coordinates" in room:
                        # Process coordinates for this room
                        for coord in room["coordinates"]:
                            x, y = coord
                            # Update element map for this pixel
                            if 0 <= y < size_y and 0 <= x < size_x:
                                self.element_map[y, x] = room_element

            # Process segments (alternative format for rooms)
            if "segments" in map_data and map_data["segments"]:
                for segment_id, coordinates in map_data["segments"].items():
                    # Get room ID and check if it's enabled
                    room_id_int = int(segment_id)

                    # Skip if room is disabled
                    if not self.drawing_config.is_enabled(f"ROOM_{room_id_int}"):
                        continue

                    # Get room element code (ROOM_1, ROOM_2, etc.)
                    room_element = None
                    if 0 < room_id_int <= 15:
                        room_element = getattr(DrawableElement, f"ROOM_{room_id_int}", None)

                    if room_element is not None and coordinates:
                        # Process individual coordinates
                        for coord in coordinates:
                            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                                x, y = coord
                                # Update element map for this pixel
                                if 0 <= y < size_y and 0 <= x < size_x:
                                    self.element_map[y, x] = room_element

            # Process walls
            if "walls" in map_data and map_data["walls"]:
                # Skip if walls are disabled
                if self.drawing_config.is_element_enabled("WALL"):
                    # Process wall coordinates
                    for coord in map_data["walls"]:
                        x, y = coord
                        # Update element map for this pixel
                        if 0 <= y < size_y and 0 <= x < size_x:
                            self.element_map[y, x] = DrawableElement.WALL

        # Find the bounding box of non-zero elements to crop the element map
        non_zero_indices = np.nonzero(self.element_map)
        if len(non_zero_indices[0]) > 0:  # If there are any non-zero elements
            # Get the bounding box coordinates
            min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
            min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])

            # Add a margin around the bounding box
            margin = 20  # Pixels of margin around the non-zero elements
            crop_min_y = max(0, min_y - margin)
            crop_max_y = min(self.element_map.shape[0] - 1, max_y + margin)
            crop_min_x = max(0, min_x - margin)
            crop_max_x = min(self.element_map.shape[1] - 1, max_x + margin)

            # Log the cropping information
            LOGGER.info(f"Cropping element map from {self.element_map.shape} to bounding box: "
                       f"({crop_min_x}, {crop_min_y}) to ({crop_max_x}, {crop_max_y})")
            LOGGER.info(f"Cropped dimensions: {crop_max_x - crop_min_x + 1}x{crop_max_y - crop_min_y + 1}")

            # Create a new, smaller array with just the non-zero region
            cropped_map = self.element_map[crop_min_y:crop_max_y+1, crop_min_x:crop_max_x+1].copy()

            # Store the cropping coordinates in the shared data for later reference
            if self.shared:
                self.shared.element_map_crop = {
                    'min_x': crop_min_x,
                    'min_y': crop_min_y,
                    'max_x': crop_max_x,
                    'max_y': crop_max_y,
                    'original_shape': self.element_map.shape
                }

            # Replace the element map with the cropped version
            self.element_map = cropped_map

            # Now resize the element map to reduce its dimensions
            # Calculate the resize factor based on the current size
            resize_factor = 5  # Reduce to 1/5 of the current size
            new_height = max(self.element_map.shape[0] // resize_factor, 50)  # Ensure minimum size
            new_width = max(self.element_map.shape[1] // resize_factor, 50)   # Ensure minimum size

            # Create a resized element map
            resized_map = np.zeros((new_height, new_width), dtype=np.int32)
            resized_map[:] = DrawableElement.FLOOR  # Initialize with floor

            # Calculate scaling factors
            y_scale = self.element_map.shape[0] / new_height
            x_scale = self.element_map.shape[1] / new_width

            # Populate the resized map by sampling from the original map
            for y in range(new_height):
                for x in range(new_width):
                    # Get the corresponding position in the original map
                    orig_y = min(int(y * y_scale), self.element_map.shape[0] - 1)
                    orig_x = min(int(x * x_scale), self.element_map.shape[1] - 1)
                    # Copy the element code
                    resized_map[y, x] = self.element_map[orig_y, orig_x]

            # Store the resize information in shared data
            if self.shared:
                if hasattr(self.shared, 'element_map_crop'):
                    self.shared.element_map_crop['resize_factor'] = resize_factor
                    self.shared.element_map_crop['resized_shape'] = resized_map.shape
                    self.shared.element_map_crop['original_cropped_shape'] = self.element_map.shape

            # Log the resizing information
            LOGGER.info(f"Resized element map from {self.element_map.shape} to {resized_map.shape} (1/{resize_factor} of cropped size)")

            # Replace the element map with the resized version
            self.element_map = resized_map

        return self.element_map

    def get_element_map(self):
        """Return the element map.

        Returns:
            numpy.ndarray: The 2D element map array or None if not initialized
        """
        return self.element_map

    def get_element_at_image_position(self, x: int, y: int):
        """Get the element code at the specified position in the image.

        This method uses calibration points to accurately map between image coordinates
        and element map coordinates.

        Args:
            x: X coordinate in the image (e.g., 0-1984)
            y: Y coordinate in the image (e.g., 0-1824)

        Returns:
            Element code at the specified position, or None if out of bounds
        """
        if self.shared is None or self.element_map is None:
            return None

        # Get calibration points if available
        calibration_points = None
        if hasattr(self.shared, 'attr_calibration_points'):
            calibration_points = self.shared.attr_calibration_points

        if calibration_points and len(calibration_points) >= 4:
            # Extract image and vacuum coordinates from calibration points
            image_points = []
            vacuum_points = []
            for point in calibration_points:
                if 'map' in point and 'vacuum' in point:
                    image_points.append((point['map']['x'], point['map']['y']))
                    vacuum_points.append((point['vacuum']['x'], point['vacuum']['y']))

            if len(image_points) >= 2:
                # Calculate scaling factors
                img_x_min = min(p[0] for p in image_points)
                img_x_max = max(p[0] for p in image_points)
                img_y_min = min(p[1] for p in image_points)
                img_y_max = max(p[1] for p in image_points)

                vac_x_min = min(p[0] for p in vacuum_points)
                vac_x_max = max(p[0] for p in vacuum_points)
                vac_y_min = min(p[1] for p in vacuum_points)
                vac_y_max = max(p[1] for p in vacuum_points)

                # Normalize the input coordinates to 0-1 range in image space
                norm_x = (x - img_x_min) / (img_x_max - img_x_min) if img_x_max > img_x_min else 0
                norm_y = (y - img_y_min) / (img_y_max - img_y_min) if img_y_max > img_y_min else 0

                # Map to vacuum coordinates
                vac_x = vac_x_min + norm_x * (vac_x_max - vac_x_min)
                vac_y = vac_y_min + norm_y * (vac_y_max - vac_y_min)

                LOGGER.debug(f"Mapped image ({x}, {y}) to vacuum ({vac_x:.1f}, {vac_y:.1f})")

                # Now map from vacuum coordinates to element map coordinates
                # This depends on how the element map was created
                if hasattr(self.shared, 'element_map_crop') and self.shared.element_map_crop:
                    crop_info = self.shared.element_map_crop

                    # Adjust for cropping
                    if 'min_x' in crop_info and 'min_y' in crop_info:
                        elem_x = int(vac_x - crop_info['min_x'])
                        elem_y = int(vac_y - crop_info['min_y'])

                        # Adjust for resizing
                        if 'resize_factor' in crop_info and 'original_cropped_shape' in crop_info and 'resized_shape' in crop_info:
                            orig_h, orig_w = crop_info['original_cropped_shape']
                            resized_h, resized_w = crop_info['resized_shape']

                            # Scale to resized coordinates
                            elem_x = int(elem_x * resized_w / orig_w)
                            elem_y = int(elem_y * resized_h / orig_h)

                        LOGGER.debug(f"Mapped vacuum ({vac_x:.1f}, {vac_y:.1f}) to element map ({elem_x}, {elem_y})")

                        # Check bounds and return element
                        height, width = self.element_map.shape
                        if 0 <= elem_y < height and 0 <= elem_x < width:
                            return self.element_map[elem_y, elem_x]

        # Fallback to the simpler method if calibration points aren't available
        return self.get_element_at_position(x, y, is_image_coords=True)

    def get_element_name(self, element_code):
        """Get the name of the element from its code.

        Args:
            element_code: The element code (e.g., 1, 2, 101, etc.)

        Returns:
            The name of the element (e.g., 'FLOOR', 'WALL', 'ROOM_1', etc.)
        """
        if element_code is None:
            return 'NONE'

        # Check if it's a room
        if element_code >= 100:
            room_number = element_code - 100
            return f'ROOM_{room_number}'

        # Check standard elements
        for name, code in vars(DrawableElement).items():
            if not name.startswith('_') and isinstance(code, int) and code == element_code:
                return name

        return f'UNKNOWN_{element_code}'

    def get_element_at_position(self, x: int, y: int, is_image_coords: bool = False):
        """Get the element code at the specified position in the element map.

        Args:
            x: X coordinate in the original (uncropped) element map or image
            y: Y coordinate in the original (uncropped) element map or image
            is_image_coords: If True, x and y are image coordinates (e.g., 1984x1824)
                           If False, x and y are element map coordinates

        Returns:
            Element code at the specified position, or None if out of bounds
        """
        if self.element_map is None:
            return None

        # If coordinates are from the image, convert them to element map coordinates first
        if is_image_coords and self.shared:
            # Get image dimensions
            if hasattr(self.shared, 'image_size') and self.shared.image_size is not None and len(self.shared.image_size) >= 2:
                image_width = self.shared.image_size[0]
                image_height = self.shared.image_size[1]
            else:
                # Default image dimensions if not available
                image_width = 1984
                image_height = 1824

            # Get original element map dimensions (before resizing)
            if hasattr(self.shared, 'element_map_crop') and self.shared.element_map_crop is not None and 'original_cropped_shape' in self.shared.element_map_crop:
                original_map_height, original_map_width = self.shared.element_map_crop['original_cropped_shape']
            else:
                # Estimate based on typical values
                original_map_width = 1310
                original_map_height = 1310

            # Calculate scaling factors between image and original element map
            x_scale_to_map = original_map_width / image_width
            y_scale_to_map = original_map_height / image_height

            # Convert image coordinates to element map coordinates
            # Apply a small offset to better align with the actual elements
            # This is based on empirical testing with the sample coordinates
            x_offset = 50  # Adjust as needed based on testing
            y_offset = 20  # Adjust as needed based on testing
            x = int((x + x_offset) * x_scale_to_map)
            y = int((y + y_offset) * y_scale_to_map)

            LOGGER.debug(f"Converted image coordinates ({x}, {y}) to element map coordinates")

        # Adjust coordinates if the element map has been cropped and resized
        if self.shared and hasattr(self.shared, 'element_map_crop'):
            # Get the crop information
            crop_info = self.shared.element_map_crop

            # Adjust coordinates to the cropped map
            x_cropped = x - crop_info['min_x']
            y_cropped = y - crop_info['min_y']

            # If the map has been resized, adjust coordinates further
            if 'resize_factor' in crop_info:
                resize_factor = crop_info['resize_factor']
                original_cropped_shape = crop_info.get('original_cropped_shape')
                resized_shape = crop_info.get('resized_shape')

                if original_cropped_shape and resized_shape:
                    # Calculate scaling factors
                    y_scale = original_cropped_shape[0] / resized_shape[0]
                    x_scale = original_cropped_shape[1] / resized_shape[1]

                    # Scale the coordinates to the resized map
                    x_resized = int(x_cropped / x_scale)
                    y_resized = int(y_cropped / y_scale)

                    # Check if the coordinates are within the resized map
                    height, width = self.element_map.shape
                    if 0 <= y_resized < height and 0 <= x_resized < width:
                        return self.element_map[y_resized, x_resized]
                    return None

            # If no resizing or missing resize info, use cropped coordinates
            height, width = self.element_map.shape
            if 0 <= y_cropped < height and 0 <= x_cropped < width:
                return self.element_map[y_cropped, x_cropped]
            return None
        else:
            # No cropping, use coordinates as is
            height, width = self.element_map.shape
            if 0 <= y < height and 0 <= x < width:
                return self.element_map[y, x]
            return None
