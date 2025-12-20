"""Tests for config/colors.py module."""

import pytest

from valetudo_map_parser.config.colors import ColorsManagement, DefaultColors, SupportedColor


class TestSupportedColor:
    """Tests for SupportedColor enum."""

    def test_color_values(self):
        """Test that color enum values are correct."""
        assert SupportedColor.CHARGER == "color_charger"
        assert SupportedColor.PATH == "color_move"
        assert SupportedColor.WALLS == "color_wall"
        assert SupportedColor.ROBOT == "color_robot"
        assert SupportedColor.GO_TO == "color_go_to"
        assert SupportedColor.NO_GO == "color_no_go"
        assert SupportedColor.ZONE_CLEAN == "color_zone_clean"
        assert SupportedColor.MAP_BACKGROUND == "color_background"
        assert SupportedColor.TEXT == "color_text"
        assert SupportedColor.TRANSPARENT == "color_transparent"

    def test_room_key(self):
        """Test room_key static method."""
        assert SupportedColor.room_key(0) == "color_room_0"
        assert SupportedColor.room_key(5) == "color_room_5"
        assert SupportedColor.room_key(15) == "color_room_15"


class TestDefaultColors:
    """Tests for DefaultColors class."""

    def test_colors_rgb_defined(self):
        """Test that default RGB colors are defined."""
        assert SupportedColor.CHARGER in DefaultColors.COLORS_RGB
        assert SupportedColor.PATH in DefaultColors.COLORS_RGB
        assert SupportedColor.WALLS in DefaultColors.COLORS_RGB
        assert SupportedColor.ROBOT in DefaultColors.COLORS_RGB

    def test_colors_rgb_format(self):
        """Test that RGB colors are in correct format (3-tuple)."""
        for color_key, color_value in DefaultColors.COLORS_RGB.items():
            assert isinstance(color_value, tuple)
            assert len(color_value) == 3
            assert all(isinstance(c, int) for c in color_value)
            assert all(0 <= c <= 255 for c in color_value)

    def test_default_room_colors(self):
        """Test that default room colors are defined for 16 rooms."""
        assert len(DefaultColors.DEFAULT_ROOM_COLORS) == 16
        for i in range(16):
            room_key = SupportedColor.room_key(i)
            assert room_key in DefaultColors.DEFAULT_ROOM_COLORS
            color = DefaultColors.DEFAULT_ROOM_COLORS[room_key]
            assert isinstance(color, tuple)
            assert len(color) == 3

    def test_default_alpha_values(self):
        """Test that default alpha values are defined."""
        assert isinstance(DefaultColors.DEFAULT_ALPHA, dict)
        assert len(DefaultColors.DEFAULT_ALPHA) > 0
        # Check specific alpha overrides
        assert "alpha_color_path" in DefaultColors.DEFAULT_ALPHA
        assert "alpha_color_wall" in DefaultColors.DEFAULT_ALPHA

    def test_get_rgba(self):
        """Test get_rgba method converts RGB to RGBA."""
        rgba = DefaultColors.get_rgba(SupportedColor.CHARGER, 255.0)
        assert isinstance(rgba, tuple)
        assert len(rgba) == 4
        assert rgba[3] == 255  # Alpha channel

    def test_get_rgba_with_custom_alpha(self):
        """Test get_rgba with custom alpha value."""
        rgba = DefaultColors.get_rgba(SupportedColor.ROBOT, 128.0)
        assert rgba[3] == 128

    def test_get_rgba_unknown_key(self):
        """Test get_rgba with unknown key returns black."""
        rgba = DefaultColors.get_rgba("unknown_color", 255.0)
        assert rgba == (0, 0, 0, 255)


class TestColorsManagement:
    """Tests for ColorsManagement class."""

    def test_initialization(self, camera_shared):
        """Test ColorsManagement initialization."""
        colors_mgmt = ColorsManagement(camera_shared)
        assert colors_mgmt.shared_var is camera_shared
        assert isinstance(colors_mgmt.color_cache, dict)
        assert colors_mgmt.user_colors is not None
        assert colors_mgmt.rooms_colors is not None

    def test_add_alpha_to_rgb_matching_lengths(self):
        """Test adding alpha to RGB colors with matching lengths."""
        alpha_channels = [255.0, 128.0, 64.0]
        rgb_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        result = ColorsManagement.add_alpha_to_rgb(alpha_channels, rgb_colors)
        assert len(result) == 3
        assert result[0] == (255, 0, 0, 255)
        assert result[1] == (0, 255, 0, 128)
        assert result[2] == (0, 0, 255, 64)

    def test_add_alpha_to_rgb_mismatched_lengths(self):
        """Test adding alpha to RGB colors with mismatched lengths."""
        alpha_channels = [255.0, 128.0]
        rgb_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        result = ColorsManagement.add_alpha_to_rgb(alpha_channels, rgb_colors)
        # Should handle mismatch gracefully
        assert isinstance(result, list)

    def test_add_alpha_to_rgb_none_alpha(self):
        """Test adding alpha with None values."""
        alpha_channels = [255.0, None, 128.0]
        rgb_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        result = ColorsManagement.add_alpha_to_rgb(alpha_channels, rgb_colors)
        assert len(result) == 3
        # None alpha should be handled (likely default to 255)
        assert isinstance(result[1], tuple)
        assert len(result[1]) == 4

    def test_add_alpha_to_rgb_empty_lists(self):
        """Test adding alpha with empty lists."""
        result = ColorsManagement.add_alpha_to_rgb([], [])
        assert result == []

    def test_initialize_user_colors(self, camera_shared):
        """Test initializing user colors from device info."""
        colors_mgmt = ColorsManagement(camera_shared)
        user_colors = colors_mgmt.initialize_user_colors(camera_shared.device_info)
        # Returns a list of RGBA tuples, not a dict
        assert isinstance(user_colors, list)
        # Should contain color tuples
        assert len(user_colors) > 0
        # Each color should be an RGBA tuple
        for color in user_colors:
            assert isinstance(color, tuple)
            assert len(color) == 4

    def test_initialize_rooms_colors(self, camera_shared):
        """Test initializing rooms colors from device info."""
        colors_mgmt = ColorsManagement(camera_shared)
        rooms_colors = colors_mgmt.initialize_rooms_colors(camera_shared.device_info)
        # Returns a list of RGBA tuples, not a dict
        assert isinstance(rooms_colors, list)
        # Should contain room color tuples
        assert len(rooms_colors) > 0
        # Each color should be an RGBA tuple
        for color in rooms_colors:
            assert isinstance(color, tuple)
            assert len(color) == 4

    def test_color_cache_usage(self, camera_shared):
        """Test that color cache is initialized and can be used."""
        colors_mgmt = ColorsManagement(camera_shared)
        assert isinstance(colors_mgmt.color_cache, dict)
        # Cache should be empty initially
        assert len(colors_mgmt.color_cache) == 0
        # Can add to cache
        colors_mgmt.color_cache["test_key"] = (255, 0, 0, 255)
        assert colors_mgmt.color_cache["test_key"] == (255, 0, 0, 255)

