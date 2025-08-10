"""Auto Crop Class for trimming and zooming images.
Version: 2024.10.0"""

from __future__ import annotations

import logging

import numpy as np
from numpy import rot90
from scipy import ndimage

from .async_utils import AsyncNumPy, make_async
from .types import Color, NumpyArray, TrimCropData, TrimsData
from .utils import BaseHandler


_LOGGER = logging.getLogger(__name__)


class TrimError(Exception):
    """Exception raised for errors in the trim process."""

    def __init__(self, message, image):
        super().__init__(message)
        self.image = image


class AutoCrop:
    """Auto Crop Class for trimming and zooming images."""

    def __init__(self, handler: BaseHandler):
        self.auto_crop = None  # auto crop data to be calculate once.
        self.crop_area = None
        self.handler = handler
        trim_data = self.handler.shared.trims.to_dict()  # trims data
        self.trim_up = trim_data.get("trim_up", 0)  # trim up
        self.trim_down = trim_data.get("trim_down", 0)  # trim down
        self.trim_left = trim_data.get("trim_left", 0)  # trim left
        self.trim_right = trim_data.get("trim_right", 0)  # trim right
        self.offset_top = self.handler.shared.offset_top  # offset top
        self.offset_bottom = self.handler.shared.offset_down  # offset bottom
        self.offset_left = self.handler.shared.offset_left  # offset left
        self.offset_right = self.handler.shared.offset_right  # offset right

    @staticmethod
    def validate_crop_dimensions(shared):
        """Ensure width and height are valid before processing cropping."""
        if shared.image_ref_width <= 0 or shared.image_ref_height <= 0:
            _LOGGER.warning(
                "Auto-crop failed: Invalid dimensions (width=%s, height=%s). Using original image.",
                shared.image_ref_width,
                shared.image_ref_height,
            )
            return False
        return True

    def check_trim(
        self, trimmed_height, trimmed_width, margin_size, image_array, file_name, rotate
    ):
        """Check if the trim is okay."""
        if trimmed_height <= margin_size or trimmed_width <= margin_size:
            self.crop_area = [0, 0, image_array.shape[1], image_array.shape[0]]
            self.handler.img_size = (image_array.shape[1], image_array.shape[0])
            raise TrimError(
                f"{file_name}: Trimming failed at rotation {rotate}.", image_array
            )

    def _calculate_trimmed_dimensions(self):
        """Calculate and update the dimensions after trimming."""
        trimmed_width = max(
            1,  # Ensure at least 1px
            (self.trim_right - self.offset_right) - (self.trim_left + self.offset_left),
        )
        trimmed_height = max(
            1,  # Ensure at least 1px
            (self.trim_down - self.offset_bottom) - (self.trim_up + self.offset_top),
        )

        # Ensure shared reference dimensions are updated
        if hasattr(self.handler.shared, "image_ref_height") and hasattr(
            self.handler.shared, "image_ref_width"
        ):
            self.handler.shared.image_ref_height = trimmed_height
            self.handler.shared.image_ref_width = trimmed_width
        else:
            _LOGGER.warning(
                "Shared attributes for image dimensions are not initialized."
            )

        return trimmed_width, trimmed_height

    async def _async_auto_crop_data(self, tdata: TrimsData):  # , tdata=None
        """Load the auto crop data from the Camera config."""
        _LOGGER.debug("Auto Crop init data: %s, %s", str(tdata), str(self.auto_crop))
        if not self.auto_crop:
            trims_data = TrimCropData.from_dict(dict(tdata.to_dict())).to_list()
            (
                self.trim_left,
                self.trim_up,
                self.trim_right,
                self.trim_down,
            ) = trims_data
            _LOGGER.debug("Auto Crop trims data: %s", trims_data)
            if trims_data != [0, 0, 0, 0]:
                self._calculate_trimmed_dimensions()
            else:
                trims_data = None
            return trims_data
        return None

    def auto_crop_offset(self):
        """Calculate the offset for the auto crop."""
        if self.auto_crop:
            self.auto_crop[0] += self.offset_left
            self.auto_crop[1] += self.offset_top
            self.auto_crop[2] -= self.offset_right
            self.auto_crop[3] -= self.offset_bottom

    async def _init_auto_crop(self):
        """Initialize the auto crop data."""
        _LOGGER.debug("Auto Crop Init data: %s", str(self.auto_crop))
        _LOGGER.debug(
            "Auto Crop Init trims data: %r", self.handler.shared.trims.to_dict()
        )
        if not self.auto_crop:  # and self.handler.shared.vacuum_state == "docked":
            self.auto_crop = await self._async_auto_crop_data(self.handler.shared.trims)
            if self.auto_crop:
                self.auto_crop_offset()
        else:
            self.handler.max_frames = 1205

        # Fallback: Ensure auto_crop is valid
        if not self.auto_crop or any(v < 0 for v in self.auto_crop):
            _LOGGER.debug("Auto-crop data unavailable. Scanning full image.")
            self.auto_crop = None

        return self.auto_crop

    async def async_image_margins(
        self, image_array: NumpyArray, detect_colour: Color
    ) -> tuple[int, int, int, int]:
        """Crop the image based on the auto crop area using scipy.ndimage for better performance."""
        # Import scipy.ndimage here to avoid import at module level

        # Create a binary mask where True = non-background pixels
        # This is much more memory efficient than storing coordinates
        mask = ~np.all(image_array == list(detect_colour), axis=2)

        # Use scipy.ndimage.find_objects to efficiently find the bounding box
        # This returns a list of slice objects that define the bounding box
        # Label the mask with a single label (1) and find its bounding box
        labeled_mask = mask.astype(np.int8)  # Convert to int8 (smallest integer type)
        objects = ndimage.find_objects(labeled_mask)

        if not objects:  # No objects found
            _LOGGER.warning(
                "%s: No non-background pixels found in image", self.handler.file_name
            )
            # Return full image dimensions as fallback
            return 0, 0, image_array.shape[1], image_array.shape[0]

        # Extract the bounding box coordinates from the slice objects
        y_slice, x_slice = objects[0]
        min_y, max_y = y_slice.start, y_slice.stop - 1
        min_x, max_x = x_slice.start, x_slice.stop - 1

        _LOGGER.debug(
            "%s: Found trims max and min values (y,x) (%s, %s) (%s, %s)...",
            self.handler.file_name,
            int(max_y),
            int(max_x),
            int(min_y),
            int(min_x),
        )
        return min_y, min_x, max_x, max_y

    async def async_get_room_bounding_box(
        self, room_name: str, rand256: bool = False
    ) -> tuple[int, int, int, int] | None:
        """Calculate bounding box coordinates from room outline for zoom functionality.

        Args:
            room_name: Name of the room to get bounding box for
            rand256: Whether this is for a rand256 vacuum (applies /10 scaling)

        Returns:
            Tuple of (left, right, up, down) coordinates or None if room not found
        """
        try:
            # For Hypfer vacuums, check room_propriety first, then rooms_pos
            if hasattr(self.handler, "room_propriety") and self.handler.room_propriety:
                # Handle different room_propriety formats
                room_data_dict = None

                if isinstance(self.handler.room_propriety, dict):
                    # Hypfer handler: room_propriety is a dictionary
                    room_data_dict = self.handler.room_propriety
                elif (
                    isinstance(self.handler.room_propriety, tuple)
                    and len(self.handler.room_propriety) >= 1
                ):
                    # Rand256 handler: room_propriety is a tuple (room_properties, zone_properties, point_properties)
                    room_data_dict = self.handler.room_propriety[0]

                if room_data_dict and isinstance(room_data_dict, dict):
                    for room_id, room_data in room_data_dict.items():
                        if room_data.get("name") == room_name:
                            outline = room_data.get("outline", [])
                            if outline:
                                xs, ys = zip(*outline)
                                left, right = min(xs), max(xs)
                                up, down = min(ys), max(ys)

                                if rand256:
                                    # Apply scaling for rand256 vacuums
                                    left = round(left / 10)
                                    right = round(right / 10)
                                    up = round(up / 10)
                                    down = round(down / 10)

                                return left, right, up, down

            # Fallback: check rooms_pos (used by both Hypfer and Rand256)
            if hasattr(self.handler, "rooms_pos") and self.handler.rooms_pos:
                for room in self.handler.rooms_pos:
                    if room.get("name") == room_name:
                        outline = room.get("outline", [])
                        if outline:
                            xs, ys = zip(*outline)
                            left, right = min(xs), max(xs)
                            up, down = min(ys), max(ys)

                            if rand256:
                                # Apply scaling for rand256 vacuums
                                left = round(left / 10)
                                right = round(right / 10)
                                up = round(up / 10)
                                down = round(down / 10)

                            return left, right, up, down

            _LOGGER.warning(
                "%s: Room '%s' not found for zoom bounding box calculation",
                self.handler.file_name,
                room_name,
            )
            return None

        except Exception as e:
            _LOGGER.error(
                "%s: Error calculating room bounding box for '%s': %s",
                self.handler.file_name,
                room_name,
                e,
            )
            return None

    async def async_check_if_zoom_is_on(
        self,
        image_array: NumpyArray,
        margin_size: int = 100,
        zoom: bool = False,
        rand256: bool = False,
    ) -> NumpyArray:
        """Check if the image needs to be zoomed."""

        if (
            zoom
            and self.handler.shared.vacuum_state == "cleaning"
            and self.handler.shared.image_auto_zoom
        ):
            # Get the current room name from robot_pos (not robot_in_room)
            current_room = (
                self.handler.robot_pos.get("in_room")
                if self.handler.robot_pos
                else None
            )
            _LOGGER.info(f"Current room: {current_room}")

            if not current_room:
                # For Rand256 handler, try to zoom based on robot position even without room data
                if (
                    rand256
                    and hasattr(self.handler, "robot_position")
                    and self.handler.robot_position
                ):
                    robot_x, robot_y = (
                        self.handler.robot_position[0],
                        self.handler.robot_position[1],
                    )

                    # Create a zoom area around the robot position (e.g., 800x800 pixels for better view)
                    zoom_size = 800
                    trim_left = max(0, int(robot_x - zoom_size // 2))
                    trim_right = min(
                        image_array.shape[1], int(robot_x + zoom_size // 2)
                    )
                    trim_up = max(0, int(robot_y - zoom_size // 2))
                    trim_down = min(image_array.shape[0], int(robot_y + zoom_size // 2))

                    _LOGGER.info(
                        "%s: Zooming to robot position area (%d, %d) with size %dx%d",
                        self.handler.file_name,
                        robot_x,
                        robot_y,
                        trim_right - trim_left,
                        trim_down - trim_up,
                    )

                    return image_array[trim_up:trim_down, trim_left:trim_right]
                else:
                    _LOGGER.warning(
                        "%s: No room information available for zoom. Using full image.",
                        self.handler.file_name,
                    )
                    return image_array[
                        self.auto_crop[1] : self.auto_crop[3],
                        self.auto_crop[0] : self.auto_crop[2],
                    ]

            # Calculate bounding box from room outline
            bounding_box = await self.async_get_room_bounding_box(current_room, rand256)

            if not bounding_box:
                _LOGGER.warning(
                    "%s: Could not calculate bounding box for room '%s'. Using full image.",
                    self.handler.file_name,
                    current_room,
                )
                return image_array[
                    self.auto_crop[1] : self.auto_crop[3],
                    self.auto_crop[0] : self.auto_crop[2],
                ]

            left, right, up, down = bounding_box

            # Apply margins
            trim_left = left - margin_size
            trim_right = right + margin_size
            trim_up = up - margin_size
            trim_down = down + margin_size
            # Ensure valid trim values
            trim_left, trim_right = sorted([trim_left, trim_right])
            trim_up, trim_down = sorted([trim_up, trim_down])

            # Prevent zero-sized images
            if trim_right - trim_left < 1 or trim_down - trim_up < 1:
                _LOGGER.warning(
                    "Zooming resulted in an invalid crop area. Using full image."
                )
                return image_array  # Return original image

            trimmed = image_array[trim_up:trim_down, trim_left:trim_right]

        else:
            trimmed = image_array[
                self.auto_crop[1] : self.auto_crop[3],
                self.auto_crop[0] : self.auto_crop[2],
            ]

        return trimmed

    async def async_rotate_the_image(
        self, trimmed: NumpyArray, rotate: int
    ) -> NumpyArray:
        """Rotate the image and return the new array."""
        if rotate == 90:
            rotated = await AsyncNumPy.async_rot90(trimmed)
            self.crop_area = [
                self.trim_left,
                self.trim_up,
                self.trim_right,
                self.trim_down,
            ]
        elif rotate == 180:
            rotated = await AsyncNumPy.async_rot90(trimmed, 2)
            self.crop_area = self.auto_crop
        elif rotate == 270:
            rotated = await AsyncNumPy.async_rot90(trimmed, 3)
            self.crop_area = [
                self.trim_left,
                self.trim_up,
                self.trim_right,
                self.trim_down,
            ]
        else:
            rotated = trimmed
            self.crop_area = self.auto_crop
        return rotated

    async def async_auto_trim_and_zoom_image(
        self,
        image_array: NumpyArray,
        detect_colour: Color = (93, 109, 126, 255),
        margin_size: int = 0,
        rotate: int = 0,
        zoom: bool = False,
        rand256: bool = False,
    ):
        """
        Automatically crops and trims a numpy array and returns the processed image.
        """
        try:
            self.auto_crop = await self._init_auto_crop()
            if (self.auto_crop is None) or (self.auto_crop == [0, 0, 0, 0]):
                _LOGGER.debug("%s: Calculating auto trim box", self.handler.file_name)
                # Find the coordinates of the first occurrence of a non-background color
                min_y, min_x, max_x, max_y = await self.async_image_margins(
                    image_array, detect_colour
                )
                # Calculate and store the trims coordinates with margins
                self.trim_left = int(min_x) - margin_size
                self.trim_up = int(min_y) - margin_size
                self.trim_right = int(max_x) + margin_size
                self.trim_down = int(max_y) + margin_size
                del min_y, min_x, max_x, max_y

                # Calculate the dimensions after trimming using min/max values
                trimmed_width, trimmed_height = self._calculate_trimmed_dimensions()

                # Test if the trims are okay or not
                try:
                    self.check_trim(
                        trimmed_height,
                        trimmed_width,
                        margin_size,
                        image_array,
                        self.handler.file_name,
                        rotate,
                    )
                except TrimError as e:
                    return e.image

                # Store Crop area of the original image_array we will use from the next frame.
                self.auto_crop = TrimCropData(
                    self.trim_left,
                    self.trim_up,
                    self.trim_right,
                    self.trim_down,
                ).to_list()
                # Update the trims data in the shared instance
                self.handler.shared.trims = TrimsData.from_dict(
                    {
                        "trim_left": self.trim_left,
                        "trim_up": self.trim_up,
                        "trim_right": self.trim_right,
                        "trim_down": self.trim_down,
                    }
                )
                self.auto_crop_offset()
            # If it is needed to zoom the image.
            trimmed = await self.async_check_if_zoom_is_on(
                image_array, margin_size, zoom, rand256
            )
            del image_array  # Free memory.
            # Rotate the cropped image based on the given angle
            rotated = await self.async_rotate_the_image(trimmed, rotate)
            del trimmed  # Free memory.
            _LOGGER.debug(
                "%s: Auto Trim Box data: %s", self.handler.file_name, self.crop_area
            )
            self.handler.crop_img_size = [rotated.shape[1], rotated.shape[0]]
            _LOGGER.debug(
                "%s: Auto Trimmed image size: %s",
                self.handler.file_name,
                self.handler.crop_img_size,
            )

        except RuntimeError as e:
            _LOGGER.warning(
                "%s: Error %s during auto trim and zoom.",
                self.handler.file_name,
                e,
                exc_info=True,
            )
            return None
        return rotated
