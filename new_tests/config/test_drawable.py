"""Tests for config/drawable.py and drawable_elements.py modules."""

import numpy as np
import pytest

from valetudo_map_parser.config.drawable import Drawable
from valetudo_map_parser.config.drawable_elements import DrawableElement, DrawingConfig


class TestDrawableElement:
    """Tests for DrawableElement enum."""

    def test_base_elements(self):
        """Test that base elements have correct values."""
        assert DrawableElement.FLOOR == 1
        assert DrawableElement.WALL == 2
        assert DrawableElement.ROBOT == 3
        assert DrawableElement.CHARGER == 4
        assert DrawableElement.VIRTUAL_WALL == 5
        assert DrawableElement.RESTRICTED_AREA == 6
        assert DrawableElement.NO_MOP_AREA == 7
        assert DrawableElement.OBSTACLE == 8
        assert DrawableElement.PATH == 9
        assert DrawableElement.PREDICTED_PATH == 10
        assert DrawableElement.GO_TO_TARGET == 11

    def test_room_elements(self):
        """Test that room elements have correct values."""
        assert DrawableElement.ROOM_1 == 101
        assert DrawableElement.ROOM_2 == 102
        assert DrawableElement.ROOM_15 == 115

    def test_all_elements_unique(self):
        """Test that all element codes are unique."""
        values = [element.value for element in DrawableElement]
        assert len(values) == len(set(values))


class TestDrawingConfig:
    """Tests for DrawingConfig class."""

    def test_initialization(self):
        """Test DrawingConfig initialization."""
        config = DrawingConfig()
        assert config._enabled_elements is not None
        assert config._element_properties is not None

    def test_all_elements_enabled_by_default(self):
        """Test that all elements are enabled by default."""
        config = DrawingConfig()
        for element in DrawableElement:
            assert config.is_enabled(element) is True

    def test_enable_element(self):
        """Test enabling an element."""
        config = DrawingConfig()
        config.disable_element(DrawableElement.WALL)
        assert config.is_enabled(DrawableElement.WALL) is False
        config.enable_element(DrawableElement.WALL)
        assert config.is_enabled(DrawableElement.WALL) is True

    def test_disable_element(self):
        """Test disabling an element."""
        config = DrawingConfig()
        config.disable_element(DrawableElement.ROBOT)
        assert config.is_enabled(DrawableElement.ROBOT) is False

    def test_toggle_element(self):
        """Test toggling an element (manual toggle by checking state)."""
        config = DrawingConfig()
        initial_state = config.is_enabled(DrawableElement.PATH)
        # Manually toggle by disabling if enabled, enabling if disabled
        if initial_state:
            config.disable_element(DrawableElement.PATH)
        else:
            config.enable_element(DrawableElement.PATH)
        assert config.is_enabled(DrawableElement.PATH) is not initial_state
        # Toggle back
        if not initial_state:
            config.disable_element(DrawableElement.PATH)
        else:
            config.enable_element(DrawableElement.PATH)
        assert config.is_enabled(DrawableElement.PATH) is initial_state

    def test_get_property(self):
        """Test getting element property."""
        config = DrawingConfig()
        color = config.get_property(DrawableElement.ROBOT, "color")
        assert isinstance(color, tuple)
        assert len(color) == 4  # RGBA

    def test_set_property(self):
        """Test setting element property."""
        config = DrawingConfig()
        new_color = (255, 0, 0, 255)
        config.set_property(DrawableElement.ROBOT, "color", new_color)
        assert config.get_property(DrawableElement.ROBOT, "color") == new_color

    def test_get_nonexistent_property(self):
        """Test getting non-existent property returns None."""
        config = DrawingConfig()
        result = config.get_property(DrawableElement.ROBOT, "nonexistent_property")
        assert result is None

    def test_room_properties_initialized(self):
        """Test that room properties are initialized."""
        config = DrawingConfig()
        for room_id in range(1, 16):
            room_element = getattr(DrawableElement, f"ROOM_{room_id}")
            color = config.get_property(room_element, "color")
            assert color is not None
            assert len(color) == 4

    def test_disable_multiple_rooms(self):
        """Test disabling multiple rooms."""
        config = DrawingConfig()
        config.disable_element(DrawableElement.ROOM_1)
        config.disable_element(DrawableElement.ROOM_5)
        config.disable_element(DrawableElement.ROOM_10)
        assert config.is_enabled(DrawableElement.ROOM_1) is False
        assert config.is_enabled(DrawableElement.ROOM_5) is False
        assert config.is_enabled(DrawableElement.ROOM_10) is False
        assert config.is_enabled(DrawableElement.ROOM_2) is True


class TestDrawable:
    """Tests for Drawable class."""

    @pytest.mark.asyncio
    async def test_create_empty_image(self):
        """Test creating an empty image."""
        width, height = 800, 600
        bg_color = (255, 255, 255, 255)
        image = await Drawable.create_empty_image(width, height, bg_color)
        assert isinstance(image, np.ndarray)
        assert image.shape == (height, width, 4)
        assert image.dtype == np.uint8
        assert np.all(image == bg_color)

    @pytest.mark.asyncio
    async def test_create_empty_image_different_colors(self):
        """Test creating empty images with different colors."""
        colors = [(0, 0, 0, 255), (128, 128, 128, 255), (255, 0, 0, 128)]
        for color in colors:
            image = await Drawable.create_empty_image(100, 100, color)
            assert np.all(image == color)

    @pytest.mark.asyncio
    async def test_from_json_to_image(self):
        """Test drawing pixels from JSON data."""
        layer = np.zeros((100, 100, 4), dtype=np.uint8)
        pixels = [[0, 0, 5], [10, 10, 3]]  # [x, y, count]
        pixel_size = 5
        color = (255, 0, 0, 255)
        result = await Drawable.from_json_to_image(layer, pixels, pixel_size, color)
        assert isinstance(result, np.ndarray)
        # Check that some pixels were drawn
        assert not np.all(result == 0)

    @pytest.mark.asyncio
    async def test_from_json_to_image_with_alpha(self):
        """Test drawing pixels with alpha blending."""
        layer = np.full((100, 100, 4), (128, 128, 128, 255), dtype=np.uint8)
        pixels = [[5, 5, 2]]
        pixel_size = 5
        color = (255, 0, 0, 128)  # Semi-transparent red
        result = await Drawable.from_json_to_image(layer, pixels, pixel_size, color)
        assert isinstance(result, np.ndarray)

