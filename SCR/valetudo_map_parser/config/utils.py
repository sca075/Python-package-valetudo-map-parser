"""Utility code for the valetudo map parser."""

import datetime
from time import time
import hashlib
import json
from dataclasses import dataclass
from typing import Callable, List, Optional
import io

import numpy as np
from PIL import Image, ImageOps

from .drawable import Drawable
from .drawable_elements import DrawingConfig
from .enhanced_drawable import EnhancedDrawable
from .status_text.status_text import StatusText
from .types import (
    LOGGER,
    ChargerPosition,
    ImageSize,
    NumpyArray,
    PilPNG,
    RobotPosition,
    WebPBytes,
)
from ..map_data import HyperMapData


@dataclass
class ResizeParams:
    """Resize the image to the given dimensions and aspect ratio."""

    pil_img: PilPNG
    width: int
    height: int
    aspect_ratio: str
    crop_size: List[int]
    is_rand: Optional[bool] = False
    offset_func: Optional[Callable] = None


@dataclass
class OffsetParams:
    """Map parameters."""

    wsf: int
    hsf: int
    width: int
    height: int
    rand256: Optional[bool] = False


class BaseHandler:
    """Avoid Code duplication"""

    def __init__(self):
        self.file_name = None
        self.shared = None
        self.img_size = None
        self.json_data = None
        self.json_id = None
        self.path_pixels = None
        self.robot_in_room = None
        self.robot_pos = None
        self.room_propriety = None
        self.rooms_pos = None
        self.charger_pos = None
        self.frame_number = 0
        self.max_frames = 1024
        self.crop_img_size = [0, 0]
        self.offset_x = 0
        self.offset_y = 0
        self.crop_area = None
        self.zooming = False
        self.async_resize_images = async_resize_image

    def get_frame_number(self) -> int:
        """Return the frame number of the image."""
        return self.frame_number

    def get_robot_position(self) -> RobotPosition | None:
        """Return the robot position."""
        return self.robot_pos

    async def async_get_image(
        self,
        m_json: dict | None,
        destinations: list | None = None,
        bytes_format: bool = False,
    ) -> PilPNG | None:
        """
        Unified async function to get PIL image from JSON data for both Hypfer and Rand256 handlers.

        This function:
        1. Calls the appropriate async_get_image_from_json method
        2. Stores the processed data in shared.new_image
        3. Backs up previous data to shared.last_image
        4. Updates shared.image_last_updated with current datetime

        @param m_json: The JSON data to use to draw the image
        @param destinations: MQTT destinations for labels (used by Rand256)
        @param bytes_format: If True, also convert to PNG bytes and store in shared.binary_image
        @param text_enabled: If True, draw text on the image
        @param vacuum_status: Vacuum status to display on the image
        @return: PIL Image or None
        """
        try:
            # Backup current image to last_image before processing new one
            if hasattr(self.shared, "new_image") and self.shared.new_image is not None:
                self.shared.last_image = self.shared.new_image

            # Call the appropriate handler method based on handler type
            if hasattr(self, "get_image_from_rrm"):
                # This is a Rand256 handler
                new_image = await self.get_image_from_rrm(
                    m_json=m_json,
                    destinations=destinations,
                    return_webp=False,  # Always return PIL Image
                )
            elif hasattr(self, "async_get_image_from_json"):
                # This is a Hypfer handler
                self.json_data = await HyperMapData.async_from_valetudo_json(m_json)
                new_image = await self.async_get_image_from_json(
                    m_json=m_json,
                    return_webp=False,  # Always return PIL Image
                )
            else:
                LOGGER.warning(
                    "%s: Handler type not recognized for async_get_image",
                    self.file_name,
                )
                return (
                    self.shared.last_image
                    if hasattr(self.shared, "last_image")
                    else None
                )

            # Store the new image in shared data
            if new_image is not None:
                self.shared.new_image = new_image
                if self.shared.show_vacuum_state:
                    text_editor = StatusText(self.shared)
                    img_text = await text_editor.get_status_text(new_image)
                    Drawable.status_text(
                        new_image,
                        img_text[1],
                        self.shared.user_colors[8],
                        img_text[0],
                        self.shared.vacuum_status_font,
                        self.shared.vacuum_status_position,
                    )
                # Convert to binary (PNG bytes) if requested
                if bytes_format:
                    with io.BytesIO() as buf:
                        new_image.save(buf, format="PNG", compress_level=1)
                        self.shared.binary_image = buf.getvalue()
                    LOGGER.debug(
                        "%s: Binary image conversion completed", self.file_name
                    )
                else:
                    self.shared.binary_image = None
                # Update the timestamp with current datetime
                self.shared.image_last_updated = datetime.datetime.fromtimestamp(time())
                LOGGER.debug(
                    "%s: Image processed and stored in shared data", self.file_name
                )
                return new_image
            else:
                LOGGER.warning(
                    "%s: Failed to generate image from JSON data", self.file_name
                )
                return (
                    self.shared.last_image
                    if hasattr(self.shared, "last_image")
                    else None
                )

        except Exception as e:
            LOGGER.error(
                "%s: Error in async_get_image: %s",
                self.file_name,
                str(e),
                exc_info=True,
            )
            return (
                self.shared.last_image if hasattr(self.shared, "last_image") else None
            )

    def get_charger_position(self) -> ChargerPosition | None:
        """Return the charger position."""
        return self.charger_pos

    def get_img_size(self) -> ImageSize | None:
        """Return the size of the image."""
        return self.img_size

    def get_json_id(self) -> str | None:
        """Return the JSON ID from the image."""
        return self.json_id

    def check_zoom_and_aspect_ratio(self) -> bool:
        """Check if the image is zoomed and has an aspect ratio."""
        return (
            self.shared.image_auto_zoom
            and self.shared.vacuum_state == "cleaning"
            and self.zooming
            and self.shared.image_zoom_lock_ratio
            or self.shared.image_aspect_ratio != "None"
        )

    def _set_image_offset_ratio_1_1(
        self, width: int, height: int, rand256: Optional[bool] = False
    ) -> None:
        """Set the image offset ratio to 1:1."""

        rotation = self.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.offset_y = self.crop_img_size[0] - width
                self.offset_x = (height - self.crop_img_size[1]) // 2
            elif rotation in [90, 270]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = (self.crop_img_size[1] - height) // 2
        else:
            if rotation in [0, 180]:
                self.offset_x = (width - self.crop_img_size[0]) // 2
                self.offset_y = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_y = (self.crop_img_size[0] - width) // 2
                self.offset_x = self.crop_img_size[1] - height
        LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.offset_x,
            self.offset_y,
        )

    def _set_image_offset_ratio_2_1(
        self, width: int, height: int, rand256: Optional[bool] = False
    ) -> None:
        """Set the image offset ratio to 2:1."""

        rotation = self.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_x = width - self.crop_img_size[0]
                self.offset_y = height - self.crop_img_size[1]
        else:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_x = width - self.crop_img_size[0]
                self.offset_y = height - self.crop_img_size[1]

        LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.offset_x,
            self.offset_y,
        )

    def _set_image_offset_ratio_3_2(
        self, width: int, height: int, rand256: Optional[bool] = False
    ) -> None:
        """Set the image offset ratio to 3:2."""

        rotation = self.shared.image_rotate

        if not rand256:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = ((height - self.crop_img_size[1]) // 2) - (
                    self.crop_img_size[1] // 10
                )
            elif rotation in [90, 270]:
                self.offset_y = (self.crop_img_size[0] - width) // 2
                self.offset_x = (self.crop_img_size[1] - height) + ((height // 10) // 2)
        else:
            if rotation in [0, 180]:
                self.offset_x = (width - self.crop_img_size[0]) // 2
                self.offset_y = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_y = (self.crop_img_size[0] - width) // 2
                self.offset_x = self.crop_img_size[1] - height

        LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.offset_x,
            self.offset_y,
        )

    def _set_image_offset_ratio_5_4(
        self, width: int, height: int, rand256: Optional[bool] = False
    ) -> None:
        """Set the image offset ratio to 5:4."""

        rotation = self.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.offset_x = ((width - self.crop_img_size[0]) // 2) - (
                    self.crop_img_size[0] // 2
                )
                self.offset_y = (self.crop_img_size[1] - height) - (
                    self.crop_img_size[1] // 2
                )
            elif rotation in [90, 270]:
                self.offset_y = ((self.crop_img_size[0] - width) // 2) - 10
                self.offset_x = (self.crop_img_size[1] - height) + (height // 10)
        else:
            if rotation in [0, 180]:
                self.offset_y = (width - self.crop_img_size[0]) // 2
                self.offset_x = self.crop_img_size[1] - height
            elif rotation in [90, 270]:
                self.offset_y = (self.crop_img_size[0] - width) // 2
                self.offset_x = self.crop_img_size[1] - height

        LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.offset_x,
            self.offset_y,
        )

    def _set_image_offset_ratio_9_16(
        self, width: int, height: int, rand256: Optional[bool] = False
    ) -> None:
        """Set the image offset ratio to 9:16."""

        rotation = self.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_x = (width - self.crop_img_size[0]) + (height // 10)
                self.offset_y = height - self.crop_img_size[1]
        else:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_x = width - self.crop_img_size[0]
                self.offset_y = height - self.crop_img_size[1]

        LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.offset_x,
            self.offset_y,
        )

    def _set_image_offset_ratio_16_9(
        self, width: int, height: int, rand256: Optional[bool] = False
    ) -> None:
        """Set the image offset ratio to 16:9."""

        rotation = self.shared.image_rotate
        if not rand256:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_x = width - self.crop_img_size[0]
                self.offset_y = height - self.crop_img_size[1]
        else:
            if rotation in [0, 180]:
                self.offset_y = width - self.crop_img_size[0]
                self.offset_x = height - self.crop_img_size[1]
            elif rotation in [90, 270]:
                self.offset_x = width - self.crop_img_size[0]
                self.offset_y = height - self.crop_img_size[1]

        LOGGER.debug(
            "%s Image Coordinates Offsets (x,y): %s. %s",
            self.file_name,
            self.offset_x,
            self.offset_y,
        )

    async def async_map_coordinates_offset(
        self, params: OffsetParams
    ) -> tuple[int, int]:
        """
        Offset the coordinates to the map.
        """
        if params.wsf == 1 and params.hsf == 1:
            self._set_image_offset_ratio_1_1(
                params.width, params.height, params.rand256
            )
        elif params.wsf == 2 and params.hsf == 1:
            self._set_image_offset_ratio_2_1(
                params.width, params.height, params.rand256
            )
        elif params.wsf == 3 and params.hsf == 2:
            self._set_image_offset_ratio_3_2(
                params.width, params.height, params.rand256
            )
        elif params.wsf == 5 and params.hsf == 4:
            self._set_image_offset_ratio_5_4(
                params.width, params.height, params.rand256
            )
        elif params.wsf == 9 and params.hsf == 16:
            self._set_image_offset_ratio_9_16(
                params.width, params.height, params.rand256
            )
        elif params.wsf == 16 and params.hsf == 9:
            self._set_image_offset_ratio_16_9(
                params.width, params.height, params.rand256
            )
        return params.width, params.height

    @staticmethod
    async def calculate_array_hash(
        layers: dict, active: Optional[List[int]]
    ) -> str | None:
        """Calculate the hash of the image based on layers and active zones."""
        if layers and active:
            data_to_hash = {
                "layers": len(layers["wall"][0]),
                "active_segments": tuple(active),
            }
            data_json = json.dumps(data_to_hash, sort_keys=True)
            return hashlib.sha256(data_json.encode()).hexdigest()
        return None

    @staticmethod
    async def async_copy_array(original_array: NumpyArray) -> NumpyArray:
        """Copy the array."""
        return NumpyArray.copy(original_array)

    def get_map_points(
        self,
    ) -> list[dict[str, int] | dict[str, int] | dict[str, int] | dict[str, int]]:
        """Return the map points."""
        return [
            {"x": 0, "y": 0},  # Top-left corner 0
            {"x": self.crop_img_size[0], "y": 0},  # Top-right corner 1
            {
                "x": self.crop_img_size[0],
                "y": self.crop_img_size[1],
            },  # Bottom-right corner 2
            {"x": 0, "y": self.crop_img_size[1]},  # Bottom-left corner (optional) 3
        ]

    def get_vacuum_points(self, rotation_angle: int) -> list[dict[str, int]]:
        """Calculate the calibration points based on the rotation angle."""

        # get_calibration_data
        vacuum_points = [
            {
                "x": self.crop_area[0] + self.offset_x,
                "y": self.crop_area[1] + self.offset_y,
            },  # Top-left corner 0
            {
                "x": self.crop_area[2] - self.offset_x,
                "y": self.crop_area[1] + self.offset_y,
            },  # Top-right corner 1
            {
                "x": self.crop_area[2] - self.offset_x,
                "y": self.crop_area[3] - self.offset_y,
            },  # Bottom-right corner 2
            {
                "x": self.crop_area[0] + self.offset_x,
                "y": self.crop_area[3] - self.offset_y,
            },  # Bottom-left corner (optional)3
        ]

        # Rotate the vacuum points based on the rotation angle
        if rotation_angle == 90:
            vacuum_points = [
                vacuum_points[1],
                vacuum_points[2],
                vacuum_points[3],
                vacuum_points[0],
            ]
        elif rotation_angle == 180:
            vacuum_points = [
                vacuum_points[2],
                vacuum_points[3],
                vacuum_points[0],
                vacuum_points[1],
            ]
        elif rotation_angle == 270:
            vacuum_points = [
                vacuum_points[3],
                vacuum_points[0],
                vacuum_points[1],
                vacuum_points[2],
            ]

        return vacuum_points

    def re_get_vacuum_points(self, rotation_angle: int) -> list[dict[str, int]]:
        """Recalculate the calibration points based on the rotation angle.
        RAND256 Vacuums Calibration Points are in 10th of a mm."""
        vacuum_points = [
            {
                "x": ((self.crop_area[0] + self.offset_x) * 10),
                "y": ((self.crop_area[1] + self.offset_y) * 10),
            },  # Top-left corner 0
            {
                "x": ((self.crop_area[2] - self.offset_x) * 10),
                "y": ((self.crop_area[1] + self.offset_y) * 10),
            },  # Top-right corner 1
            {
                "x": ((self.crop_area[2] - self.offset_x) * 10),
                "y": ((self.crop_area[3] - self.offset_y) * 10),
            },  # Bottom-right corner 2
            {
                "x": ((self.crop_area[0] + self.offset_x) * 10),
                "y": ((self.crop_area[3] - self.offset_y) * 10),
            },  # Bottom-left corner (optional)3
        ]

        # Rotate the vacuum points based on the rotation angle
        if rotation_angle == 90:
            vacuum_points = [
                vacuum_points[1],
                vacuum_points[2],
                vacuum_points[3],
                vacuum_points[0],
            ]
        elif rotation_angle == 180:
            vacuum_points = [
                vacuum_points[2],
                vacuum_points[3],
                vacuum_points[0],
                vacuum_points[1],
            ]
        elif rotation_angle == 270:
            vacuum_points = [
                vacuum_points[3],
                vacuum_points[0],
                vacuum_points[1],
                vacuum_points[2],
            ]

        return vacuum_points

    async def async_zone_propriety(self, zones_data) -> dict:
        """Get the zone propriety"""
        zone_properties = {}
        id_count = 1
        for zone in zones_data:
            zone_name = zone.get("name")
            coordinates = zone.get("coordinates")
            if coordinates and len(coordinates) > 0:
                coordinates[0].pop()
                x1, y1, x2, y2 = coordinates[0]
                zone_properties[zone_name] = {
                    "zones": coordinates,
                    "name": zone_name,
                    "x": ((x1 + x2) // 2),
                    "y": ((y1 + y2) // 2),
                }
                id_count += 1
            if id_count > 1:
                LOGGER.debug("%s: Zones Properties updated.", self.file_name)
        return zone_properties

    async def async_points_propriety(self, points_data) -> dict:
        """Get the point propriety"""
        point_properties = {}
        id_count = 1
        for point in points_data:
            point_name = point.get("name")
            coordinates = point.get("coordinates")
            if coordinates and len(coordinates) > 0:
                coordinates = point.get("coordinates")
                x1, y1 = coordinates
                point_properties[id_count] = {
                    "position": coordinates,
                    "name": point_name,
                    "x": x1,
                    "y": y1,
                }
                id_count += 1
            if id_count > 1:
                LOGGER.debug("%s: Point Properties updated.", self.file_name)
        return point_properties

    @staticmethod
    def get_corners(
        x_max: int, x_min: int, y_max: int, y_min: int
    ) -> list[tuple[int, int]]:
        """Return the corners of the image."""
        return [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ]


async def async_resize_image(params: ResizeParams):
    """Resize the image to the given dimensions and aspect ratio."""
    if params.aspect_ratio:
        wsf, hsf = [int(x) for x in params.aspect_ratio.split(",")]

        if wsf == 0 or hsf == 0 or params.width <= 0 or params.height <= 0:
            LOGGER.warning(
                "Invalid aspect ratio parameters: width=%s, height=%s, wsf=%s, hsf=%s. Returning original image.",
                params.width,
                params.height,
                wsf,
                hsf,
            )
            return params.pil_img  # Return original image if invalid
        if params.width == 0:
            params.width = params.pil_img.width
        if params.height == 0:
            params.height = params.pil_img.height
        new_aspect_ratio = wsf / hsf
        if params.width / params.height > new_aspect_ratio:
            new_width = int(params.pil_img.height * new_aspect_ratio)
            new_height = params.pil_img.height
        else:
            new_width = params.pil_img.width
            new_height = int(params.pil_img.width / new_aspect_ratio)

        LOGGER.debug("Resizing image to aspect ratio: %s, %s", wsf, hsf)
        LOGGER.debug("New image size: %s x %s", new_width, new_height)

        if (params.crop_size is not None) and (params.offset_func is not None):
            offset = OffsetParams(wsf, hsf, new_width, new_height, params.is_rand)
            params.crop_size[0], params.crop_size[1] = await params.offset_func(offset)

        return ImageOps.pad(params.pil_img, (new_width, new_height))

    return ImageOps.pad(params.pil_img, (params.width, params.height))


def prepare_resize_params(handler, pil_img, rand):
    """Prepare resize parameters for image resizing."""
    return ResizeParams(
        pil_img=pil_img,
        width=handler.shared.image_ref_width,
        height=handler.shared.image_ref_height,
        aspect_ratio=handler.shared.image_aspect_ratio,
        crop_size=handler.crop_img_size,
        offset_func=handler.async_map_coordinates_offset,
        is_rand=rand,
    )


def initialize_drawing_config(handler):
    """
    Initialize drawing configuration from device_info.

    Args:
        handler: The handler instance with shared data and file_name attributes

    Returns:
        Tuple of (DrawingConfig, Drawable, EnhancedDrawable)
    """

    # Initialize drawing configuration
    drawing_config = DrawingConfig()

    if (
        hasattr(handler.shared, "device_info")
        and handler.shared.device_info is not None
    ):
        drawing_config.update_from_device_info(handler.shared.device_info)

    # Initialize both drawable systems for backward compatibility
    draw = Drawable()  # Legacy drawing utilities
    enhanced_draw = EnhancedDrawable(drawing_config)  # New enhanced drawing system

    return drawing_config, draw, enhanced_draw


def blend_colors(base_color, overlay_color):
    """
    Blend two RGBA colors using alpha compositing.

    Args:
        base_color: Base RGBA color tuple (r, g, b, a)
        overlay_color: Overlay RGBA color tuple (r, g, b, a)

    Returns:
        Blended RGBA color tuple (r, g, b, a)
    """
    r1, g1, b1, a1 = base_color
    r2, g2, b2, a2 = overlay_color

    # Convert alpha to 0-1 range
    a1 = a1 / 255.0
    a2 = a2 / 255.0

    # Calculate resulting alpha
    a_out = a1 + a2 * (1 - a1)

    # Avoid division by zero
    if a_out < 0.0001:
        return [0, 0, 0, 0]

    # Calculate blended RGB components
    r_out = (r1 * a1 + r2 * a2 * (1 - a1)) / a_out
    g_out = (g1 * a1 + g2 * a2 * (1 - a1)) / a_out
    b_out = (b1 * a1 + b2 * a2 * (1 - a1)) / a_out

    # Convert back to 0-255 range and return as tuple
    return (
        int(max(0, min(255, r_out))),
        int(max(0, min(255, g_out))),
        int(max(0, min(255, b_out))),
        int(max(0, min(255, a_out * 255))),
    )


def blend_pixel(array, x, y, color, element, element_map=None, drawing_config=None):
    """
    Blend a pixel color with the existing color at the specified position.
    Also updates the element map if the new element has higher z-index.

    Args:
        array: The image array to modify
        x: X coordinate
        y: Y coordinate
        color: RGBA color tuple to blend
        element: Element code for the pixel
        element_map: Optional element map to update
        drawing_config: Optional drawing configuration for z-index lookup

    Returns:
        None
    """
    # Check bounds
    if not (0 <= y < array.shape[0] and 0 <= x < array.shape[1]):
        return

    # Get current element at this position
    current_element = None
    if element_map is not None:
        current_element = element_map[y, x]

    # Get z-index values for comparison
    current_z = 0
    new_z = 0

    if drawing_config is not None:
        current_z = (
            drawing_config.get_property(current_element, "z_index", 0)
            if current_element
            else 0
        )
        new_z = drawing_config.get_property(element, "z_index", 0)

    # Update element map if new element has higher z-index
    if element_map is not None and new_z >= current_z:
        element_map[y, x] = element

    # Blend colors
    base_color = array[y, x]
    blended_color = blend_colors(base_color, color)
    array[y, x] = blended_color


def manage_drawable_elements(
    handler,
    action,
    element_code=None,
    element_codes=None,
    property_name=None,
    value=None,
):
    """
    Manage drawable elements (enable, disable, set elements, set properties).

    Args:
        handler: The handler instance with drawing_config attribute
        action: Action to perform ('enable', 'disable', 'set_elements', 'set_property')
        element_code: Element code for enable/disable/set_property actions
        element_codes: List of element codes for set_elements action
        property_name: Property name for set_property action
        value: Property value for set_property action

    Returns:
        None
    """
    if not hasattr(handler, "drawing_config") or handler.drawing_config is None:
        return

    if action == "enable" and element_code is not None:
        handler.drawing_config.enable_element(element_code)
    elif action == "disable" and element_code is not None:
        handler.drawing_config.disable_element(element_code)
    elif action == "set_elements" and element_codes is not None:
        handler.drawing_config.set_elements(element_codes)
    elif (
        action == "set_property"
        and element_code is not None
        and property_name is not None
    ):
        handler.drawing_config.set_property(element_code, property_name, value)


def handle_room_outline_error(file_name, room_id, error):
    """
    Handle errors during room outline extraction.

    Args:
        file_name: Name of the file for logging
        room_id: Room ID for logging
        error: The error that occurred

    Returns:
        None
    """

    LOGGER.warning(
        "%s: Failed to trace outline for room %s: %s",
        file_name,
        str(room_id),
        str(error),
    )


async def async_extract_room_outline(
    room_mask, min_x, min_y, max_x, max_y, file_name, room_id_int
):
    """
    Extract the outline of a room from a binary mask.

    Args:
        room_mask: Binary mask where room pixels are 1 and non-room pixels are 0
        min_x: Minimum x coordinate of the room
        min_y: Minimum y coordinate of the room
        max_x: Maximum x coordinate of the room
        max_y: Maximum y coordinate of the room
        file_name: Name of the file for logging
        room_id_int: Room ID for logging
    Returns:
        List of (x, y) points forming the room outline
    """

    # Get the dimensions of the mask
    height, width = room_mask.shape

    # Find the coordinates of all room pixels
    room_y, room_x = np.where(room_mask > 0)
    if len(room_y) == 0:
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

    # Get the bounding box of the room
    min_y, max_y = np.min(room_y), np.max(room_y)
    min_x, max_x = np.min(room_x), np.max(room_x)

    # For simple rooms, just use the rectangular outline
    rect_outline = [
        (min_x, min_y),  # Top-left
        (max_x, min_y),  # Top-right
        (max_x, max_y),  # Bottom-right
        (min_x, max_y),  # Bottom-left
    ]

    # For more complex room shapes, trace the boundary
    # This is a custom boundary tracing algorithm that works without OpenCV
    try:
        # Create a padded mask to handle edge cases
        padded_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = room_mask

        # Find boundary pixels (pixels that have at least one non-room neighbor)
        boundary_points = []

        # More efficient boundary detection - only check pixels that are part of the room
        for y, x in zip(room_y, room_x):
            # Check if this is a boundary pixel (at least one neighbor is 0)
            is_boundary = False
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (
                    ny < 0
                    or ny >= height
                    or nx < 0
                    or nx >= width
                    or room_mask[ny, nx] == 0
                ):
                    is_boundary = True
                    break
            if is_boundary:
                boundary_points.append((x, y))

        # Log the number of boundary points found
        LOGGER.debug(
            "%s: Room %s has %d boundary points",
            file_name,
            str(room_id_int),
            len(boundary_points),
        )

        # If we found too few boundary points, use the rectangular outline
        if len(boundary_points) < 8:  # Need at least 8 points for a meaningful shape
            LOGGER.debug(
                "%s: Room %s has too few boundary points (%d), using rectangular outline",
                file_name,
                str(room_id_int),
                len(boundary_points),
            )
            return rect_outline

        # Use a more sophisticated algorithm to create a coherent outline
        # We'll use a convex hull approach to get the main shape
        # Sort points by angle from centroid
        centroid_x = np.mean([p[0] for p in boundary_points])
        centroid_y = np.mean([p[1] for p in boundary_points])

        # Calculate angles from centroid
        def calculate_angle(point):
            return np.arctan2(point[1] - int(centroid_y), point[0] - int(centroid_x))

        # Sort boundary points by angle
        boundary_points.sort(key=calculate_angle)

        # Simplify the outline if it has too many points
        if len(boundary_points) > 20:
            # Take every Nth point to simplify
            step = len(boundary_points) // 20
            simplified_outline = [
                boundary_points[i] for i in range(0, len(boundary_points), step)
            ]
            # Make sure we have at least 8 points
            if len(simplified_outline) < 8:
                simplified_outline = boundary_points[:: len(boundary_points) // 8]
        else:
            simplified_outline = boundary_points

        # Make sure to close the loop
        if simplified_outline[0] != simplified_outline[-1]:
            simplified_outline.append(simplified_outline[0])

        # Convert NumPy int64 values to regular Python integers
        simplified_outline = [(int(x), int(y)) for x, y in simplified_outline]

        LOGGER.debug(
            "%s: Room %s outline has %d points",
            file_name,
            str(room_id_int),
            len(simplified_outline),
        )

        return simplified_outline

    except (ValueError, IndexError, TypeError, ArithmeticError) as e:
        LOGGER.warning(
            "%s: Error tracing room outline: %s. Using rectangular outline instead.",
            file_name,
            str(e),
        )
        return rect_outline


async def numpy_to_webp_bytes(
    img_np_array: np.ndarray, quality: int = 85, lossless: bool = False
) -> WebPBytes:
    """
    Convert NumPy array directly to WebP bytes.

    Args:
        img_np_array: RGBA NumPy array
        quality: WebP quality (0-100, ignored if lossless=True)
        lossless: Use lossless WebP compression

    Returns:
        WebP image as bytes
    """
    # Convert NumPy array to PIL Image
    pil_img = Image.fromarray(img_np_array, mode="RGBA")

    # Create bytes buffer
    webp_buffer = io.BytesIO()

    # Save as WebP - PIL images should use lossless=True for best results
    pil_img.save(
        webp_buffer,
        format="WEBP",
        lossless=True,  # Always lossless for PIL images
        method=1,  # Fastest method for lossless
    )

    # Get bytes and cleanup
    webp_bytes = webp_buffer.getvalue()
    webp_buffer.close()

    return webp_bytes


async def pil_to_webp_bytes(
    pil_img: Image.Image, quality: int = 85, lossless: bool = False
) -> bytes:
    """
    Convert PIL Image to WebP bytes.

    Args:
        pil_img: PIL Image object
        quality: WebP quality (0-100, ignored if lossless=True)
        lossless: Use lossless WebP compression

    Returns:
        WebP image as bytes
    """
    # Create bytes buffer
    webp_buffer = io.BytesIO()

    # Save as WebP - PIL images should use lossless=True for best results
    pil_img.save(
        webp_buffer,
        format="WEBP",
        lossless=True,  # Always lossless for PIL images
        method=1,  # Fastest method for lossless
    )

    # Get bytes and cleanup
    webp_bytes = webp_buffer.getvalue()
    webp_buffer.close()

    return webp_bytes


def webp_bytes_to_pil(webp_bytes: bytes) -> Image.Image:
    """
    Convert WebP bytes back to PIL Image for display or further processing.

    Args:
        webp_bytes: WebP image as bytes

    Returns:
        PIL Image object
    """
    webp_buffer = io.BytesIO(webp_bytes)
    pil_img = Image.open(webp_buffer)
    return pil_img
