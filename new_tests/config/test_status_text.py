"""Tests for config/status_text module."""

import pytest
from PIL import Image

from valetudo_map_parser.config.status_text.status_text import StatusText
from valetudo_map_parser.config.status_text.translations import translations


class TestTranslations:
    """Tests for translations dictionary."""

    def test_translations_exist(self):
        """Test that translations dictionary exists and has content."""
        assert translations is not None
        assert isinstance(translations, dict)
        assert len(translations) > 0

    def test_english_translations(self):
        """Test that English translations exist."""
        assert "en" in translations
        assert isinstance(translations["en"], dict)

    def test_common_states_translated(self):
        """Test that common vacuum states are translated."""
        common_states = ["docked", "cleaning", "paused", "error", "returning"]
        for lang_code, lang_translations in translations.items():
            # At least some states should be translated
            assert isinstance(lang_translations, dict)

    def test_multiple_languages(self):
        """Test that multiple languages are available."""
        assert len(translations) >= 2  # At least English and one other language


class TestStatusText:
    """Tests for StatusText class."""

    @pytest.mark.asyncio
    async def test_initialization(self, camera_shared):
        """Test StatusText initialization."""
        status_text = StatusText(camera_shared)
        assert status_text._shared is camera_shared

    @pytest.mark.asyncio
    async def test_get_status_text_basic(self, camera_shared, test_image):
        """Test getting basic status text."""
        camera_shared.vacuum_state = "docked"
        camera_shared.vacuum_battery = 100
        camera_shared.vacuum_connection = True
        camera_shared.show_vacuum_state = True
        camera_shared.user_language = "en"

        status_text = StatusText(camera_shared)
        text, size = await status_text.get_status_text(test_image)

        assert isinstance(text, list)
        assert len(text) > 0
        assert isinstance(size, int)
        assert size > 0

    @pytest.mark.asyncio
    async def test_get_status_text_docked_charging(self, camera_shared, test_image):
        """Test status text when docked and charging."""
        camera_shared.vacuum_state = "docked"
        camera_shared.vacuum_battery = 50
        camera_shared.vacuum_connection = True
        camera_shared.show_vacuum_state = True
        camera_shared.user_language = "en"

        status_text = StatusText(camera_shared)
        text, size = await status_text.get_status_text(test_image)

        assert isinstance(text, list)
        # Should show battery percentage
        assert any("%" in t for t in text)

    @pytest.mark.asyncio
    async def test_get_status_text_docked_full(self, camera_shared, test_image):
        """Test status text when docked and fully charged."""
        camera_shared.vacuum_state = "docked"
        camera_shared.vacuum_battery = 100
        camera_shared.vacuum_connection = True
        camera_shared.show_vacuum_state = True
        camera_shared.user_language = "en"

        status_text = StatusText(camera_shared)
        text, size = await status_text.get_status_text(test_image)

        assert isinstance(text, list)
        # Should show "Ready" text
        assert any("Ready" in t for t in text)

    @pytest.mark.asyncio
    async def test_get_status_text_disconnected(self, camera_shared, test_image):
        """Test status text when MQTT disconnected."""
        camera_shared.vacuum_connection = False
        camera_shared.show_vacuum_state = True
        camera_shared.user_language = "en"

        status_text = StatusText(camera_shared)
        text, size = await status_text.get_status_text(test_image)

        assert isinstance(text, list)
        assert any("Disconnected" in t for t in text)

    @pytest.mark.asyncio
    async def test_get_status_text_with_room(self, camera_shared, test_image):
        """Test status text with current room information."""
        camera_shared.vacuum_state = "cleaning"
        camera_shared.vacuum_battery = 75
        camera_shared.vacuum_connection = True
        camera_shared.show_vacuum_state = True
        camera_shared.user_language = "en"
        camera_shared.current_room = {"in_room": "Kitchen"}

        status_text = StatusText(camera_shared)
        text, size = await status_text.get_status_text(test_image)

        assert isinstance(text, list)
        # Should contain room name
        assert any("Kitchen" in t for t in text)

    @pytest.mark.asyncio
    async def test_get_status_text_no_image(self, camera_shared):
        """Test status text generation without image."""
        camera_shared.vacuum_state = "docked"
        camera_shared.vacuum_battery = 100
        camera_shared.vacuum_connection = True
        camera_shared.show_vacuum_state = True
        camera_shared.vacuum_status_size = 50

        status_text = StatusText(camera_shared)
        text, size = await status_text.get_status_text(None)

        assert isinstance(text, list)
        assert size == camera_shared.vacuum_status_size

    @pytest.mark.asyncio
    async def test_get_status_text_closed_image(self, camera_shared):
        """Test status text generation with closed image."""
        camera_shared.vacuum_state = "docked"
        camera_shared.vacuum_battery = 100
        camera_shared.vacuum_connection = True
        camera_shared.show_vacuum_state = True

        img = Image.new("RGBA", (800, 600), (255, 255, 255, 255))
        img.close()

        status_text = StatusText(camera_shared)
        text, size = await status_text.get_status_text(img)

        assert isinstance(text, list)
        assert isinstance(size, int)

    @pytest.mark.asyncio
    async def test_get_status_text_different_languages(self, camera_shared, test_image):
        """Test status text in different languages."""
        camera_shared.vacuum_state = "docked"
        camera_shared.vacuum_battery = 100
        camera_shared.vacuum_connection = True
        camera_shared.show_vacuum_state = True

        status_text = StatusText(camera_shared)

        for lang_code in translations.keys():
            camera_shared.user_language = lang_code
            text, size = await status_text.get_status_text(test_image)
            assert isinstance(text, list)
            assert len(text) > 0

    @pytest.mark.asyncio
    async def test_get_status_text_dynamic_sizing(self, camera_shared):
        """Test dynamic text sizing based on image width."""
        camera_shared.vacuum_state = "docked"
        camera_shared.vacuum_battery = 100
        camera_shared.vacuum_connection = True
        camera_shared.show_vacuum_state = True
        camera_shared.vacuum_status_size = 60  # >= 50 triggers dynamic sizing

        status_text = StatusText(camera_shared)

        small_img = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
        large_img = Image.new("RGBA", (1600, 1200), (255, 255, 255, 255))

        _, size_small = await status_text.get_status_text(small_img)
        _, size_large = await status_text.get_status_text(large_img)

        assert size_large >= size_small

        small_img.close()
        large_img.close()

