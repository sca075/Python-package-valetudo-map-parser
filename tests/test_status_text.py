"""Tests for status text generation and translations."""

from __future__ import annotations

import asyncio
import os
import sys

from PIL import Image
from valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.config.status_text.status_text import StatusText

# Ensure package root is on sys.path for test execution contexts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


async def _build_shared(file_name: str = "test_device"):
    manager = CameraSharedManager(file_name, device_info={})
    shared = manager.get_instance()
    # Defaults
    shared.show_vacuum_state = True
    shared.vacuum_status_size = 60  # >=50 triggers dynamic sizing
    shared.user_language = None
    shared.vacuum_connection = True
    shared.vacuum_state = "docked"
    shared.vacuum_battery = 90
    shared.current_room = {"in_room": "Kitchen"}
    return shared


def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_language_fallback_and_ready_text():
    async def inner():
        shared = await _build_shared()
        shared.user_language = None  # fallback to EN
        st = StatusText(hass=None, camera_shared=shared)
        img = Image.new("RGBA", (300, 50), (0, 0, 0, 0))

        status_text, size = await st.get_status_text(img)

        # Expect device name + translated status (docked) and ready text or charging indicator
        assert status_text[0].startswith(shared.file_name + ": ")
        assert any(t in status_text for t in ["Ready.", "∑Ϟ ", "∑Ϟ"]) or any(
            "Docked" in t for t in status_text
        )
        assert isinstance(size, int) and size > 0

    run_async(inner())


def test_docked_battery_branches():
    async def inner():
        shared = await _build_shared()
        st = StatusText(hass=None, camera_shared=shared)
        img = Image.new("RGBA", (600, 60), (0, 0, 0, 0))

        # Battery < 100 -> should show charging symbol and percentage
        shared.vacuum_battery = 50
        status_text, _ = await st.get_status_text(img)
        assert any("%" in t for t in status_text)

        # Battery 100 -> should show 'Ready.' text branch
        shared.vacuum_battery = 100
        status_text, _ = await st.get_status_text(img)
        assert any("Ready." in t for t in status_text)

    run_async(inner())


def test_mqtt_disconnected_uses_translated_key():
    async def inner():
        shared = await _build_shared()
        shared.vacuum_connection = False
        shared.user_language = "en"
        st = StatusText(hass=None, camera_shared=shared)
        img = Image.new("RGBA", (400, 40), (0, 0, 0, 0))

        status_text, _ = await st.get_status_text(img)
        assert status_text[0] == f"{shared.file_name}: Disconnected from MQTT?"

    run_async(inner())


def test_current_room_suffix_appended():
    async def inner():
        shared = await _build_shared()
        st = StatusText(hass=None, camera_shared=shared)
        img = Image.new("RGBA", (400, 40), (0, 0, 0, 0))

        status_text, _ = await st.get_status_text(img)
        # Should contain " (Kitchen)"
        print(status_text)
        assert any(" (Kitchen)" in t for t in status_text)


def test_graceful_when_no_image_passed():
    async def inner():
        shared = await _build_shared()
        st = StatusText(hass=None, camera_shared=shared)
        status_text, size = await st.get_status_text(None)  # type: ignore[arg-type]
        assert size == shared.vacuum_status_size
        assert status_text[0].startswith(shared.file_name + ": ")

    run_async(inner())


def test_graceful_when_image_closed():
    async def inner():
        shared = await _build_shared()
        st = StatusText(hass=None, camera_shared=shared)
        img = Image.new("RGBA", (300, 50), (0, 0, 0, 0))
        img.close()
        status_text, size = await st.get_status_text(img)
        assert isinstance(size, int) and size >= 0
        assert status_text[0].startswith(shared.file_name + ": ")

    run_async(inner())


def test_dynamic_size_uses_image_width_when_large_size():
    async def inner():
        shared = await _build_shared()
        st = StatusText(hass=None, camera_shared=shared)

        # Two images with different widths should yield different computed sizes when size >= 50
        img_small = Image.new("RGBA", (200, 40), (0, 0, 0, 0))
        img_large = Image.new("RGBA", (800, 40), (0, 0, 0, 0))

        _, size_small = await st.get_status_text(img_small)
        _, size_large = await st.get_status_text(img_large)

        assert size_large >= size_small

    run_async(inner())
