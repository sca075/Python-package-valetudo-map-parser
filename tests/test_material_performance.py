"""
Performance comparison test between Python and C material implementations.
Tests both material.py (production) and material_mvcrender.py (C-based) implementations.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import sys
import time
from pathlib import Path

import psutil


# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from SCR.valetudo_map_parser.config.colors import ColorsManagement

# Import both material implementations
from SCR.valetudo_map_parser.config.material import (
    MaterialTileRenderer as MaterialPython,
)
from SCR.valetudo_map_parser.config.material_mvcrender import (
    MaterialTileRendererMVC as MaterialC,
)
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.hypfer_handler import HypferMapImageHandler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

_LOGGER = logging.getLogger(__name__)

# ----- Test Configuration -----
TEST_FILE = "X40_carpet.json"
FRAME_COUNT = 100  # Number of frames to generate for performance testing
# ------------------------------


class MaterialPerformanceTester:
    """Compare performance between Python and C material implementations."""

    def __init__(self, json_file: str):
        self.json_file = json_file
        self.json_data = None
        self.device_info = None
        self.shared = None
        self.process = psutil.Process()

    def load_test_data(self):
        """Load test JSON data."""
        json_path = Path(__file__).parent / self.json_file
        _LOGGER.info(f"Loading test data from {json_path}")
        with open(json_path) as f:
            self.json_data = json.load(f)

    def setup_device_info(self, use_material: bool = True):
        """Setup device_info configuration."""
        self.device_info = {
            "rand256": False,
            "image_rotate": 0,
            "image_auto_zoom": True,
            "image_ref_height": 1080,
            "image_ref_width": 1920,
            "user_language": "en",
            "disable_carpets": False,
            "disable_material_overlay": not use_material,
            "color_material_wood": [255, 0, 0],  # Bright red for visibility
            "alpha_material_wood": 200,
            "color_material_tile": [0, 0, 255],  # Bright blue for visibility
            "alpha_material_tile": 200,
        }

        # Initialize shared data
        shared_data = CameraSharedManager("test_hypfer", self.device_info)
        self.shared = shared_data.get_instance()
        self.shared.vacuum_state = "docked"
        self.shared.vacuum_connection = True
        self.shared.vacuum_battery = 100

        # Initialize colors
        colors = ColorsManagement(self.shared)
        colors.set_initial_colours(self.device_info)

    async def test_implementation(
        self, implementation_name: str, use_c_impl: bool = False
    ) -> dict:
        """
        Test a specific material implementation.

        Args:
            implementation_name: Name for logging ("Python" or "C")
            use_c_impl: If True, monkey-patch to use C implementation

        Returns:
            Dictionary with performance metrics
        """
        _LOGGER.info(f"\n{'=' * 60}")
        _LOGGER.info(f"Testing {implementation_name} Implementation")
        _LOGGER.info(f"{'=' * 60}")

        # Monkey-patch the material renderer if using C implementation
        original_renderer = None
        if use_c_impl:
            import SCR.valetudo_map_parser.hypfer_draw as hypfer_draw_module

            original_renderer = hypfer_draw_module.MaterialTileRenderer
            hypfer_draw_module.MaterialTileRenderer = MaterialC
            _LOGGER.info("Patched to use C implementation (MaterialTileRendererMVC)")

        times = []
        mem_start = self.process.memory_info().rss / 1024 / 1024  # MB

        try:
            # Create handler ONCE and reuse it across frames (critical for memory!)
            handler = HypferMapImageHandler(self.shared)

            for frame_num in range(FRAME_COUNT):
                gc.collect()

                start_time = time.perf_counter()

                # Generate image using the same handler instance
                image = await handler.async_get_image(self.json_data)

                end_time = time.perf_counter()
                frame_time = (end_time - start_time) * 1000  # Convert to ms
                times.append(frame_time)

                if frame_num == 0:
                    _LOGGER.info(f"Frame 1: {frame_time:.1f}ms")
                    # Image is returned as tuple (pil_image, attributes)
                    if image and isinstance(image, tuple) and len(image) > 0:
                        pil_image = image[0]
                        if pil_image and hasattr(pil_image, "size"):
                            _LOGGER.info(f"Image size: {pil_image.size}")
                            # Save first frame for visual inspection - make a copy to avoid closed image error
                            try:
                                output_path = (
                                    Path(__file__).parent
                                    / f"output_material_{implementation_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
                                )
                                # Make a copy before saving to avoid issues with closed images
                                img_copy = pil_image.copy()
                                img_copy.save(output_path)
                                img_copy.close()
                                _LOGGER.info(f"Saved first frame to: {output_path}")
                            except Exception as e:
                                _LOGGER.warning(f"Failed to save image: {e}")

                # Clean up
                if image and isinstance(image, tuple) and len(image) > 0:
                    pil_image = image[0]
                    if pil_image and hasattr(pil_image, "close"):
                        pil_image.close()
                del image

        finally:
            # Restore original renderer if we patched it
            if use_c_impl and original_renderer:
                import SCR.valetudo_map_parser.hypfer_draw as hypfer_draw_module

                hypfer_draw_module.MaterialTileRenderer = original_renderer
                _LOGGER.info("Restored original Python implementation")

        mem_end = self.process.memory_info().rss / 1024 / 1024  # MB
        mem_growth = mem_end - mem_start

        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        _LOGGER.info(f"\n{implementation_name} Results:")
        _LOGGER.info(f"  Average: {avg_time:.1f}ms")
        _LOGGER.info(f"  Min: {min_time:.1f}ms")
        _LOGGER.info(f"  Max: {max_time:.1f}ms")
        _LOGGER.info(f"  Memory growth: {mem_growth:+.1f}MB")

        return {
            "name": implementation_name,
            "avg_ms": avg_time,
            "min_ms": min_time,
            "max_ms": max_time,
            "mem_growth_mb": mem_growth,
            "times": times,
        }


async def run_comparison():
    """Run performance comparison tests."""
    tester = MaterialPerformanceTester(TEST_FILE)
    tester.load_test_data()

    # Test with materials enabled
    tester.setup_device_info(use_material=True)

    _LOGGER.info(f"\nRunning {FRAME_COUNT} frames for each implementation...")

    # Test Python implementation (production)
    results_python = await tester.test_implementation(
        "Python (Production)", use_c_impl=False
    )

    # Small delay between tests
    await asyncio.sleep(1)
    gc.collect()

    # Test C implementation (mvcrender)
    results_c = await tester.test_implementation("C (mvcrender)", use_c_impl=True)

    # Compare results
    _LOGGER.info(f"\n{'=' * 60}")
    _LOGGER.info("PERFORMANCE COMPARISON")
    _LOGGER.info(f"{'=' * 60}")
    _LOGGER.info(f"Python avg: {results_python['avg_ms']:.1f}ms")
    _LOGGER.info(f"C avg:      {results_c['avg_ms']:.1f}ms")

    speedup = results_python["avg_ms"] / results_c["avg_ms"]
    improvement = (
        (results_python["avg_ms"] - results_c["avg_ms"]) / results_python["avg_ms"]
    ) * 100

    _LOGGER.info(f"\nSpeedup: {speedup:.2f}x")
    _LOGGER.info(f"Improvement: {improvement:.1f}%")
    _LOGGER.info(f"\nPython memory growth: {results_python['mem_growth_mb']:+.1f}MB")
    _LOGGER.info(f"C memory growth:      {results_c['mem_growth_mb']:+.1f}MB")

    return results_python, results_c


def test_material_performance_comparison():
    """Pytest test for material performance comparison."""
    results_python, results_c = asyncio.run(run_comparison())

    # Assert that C implementation is at least as fast as Python
    assert results_c["avg_ms"] <= results_python["avg_ms"] * 1.1, (
        f"C implementation slower than expected: {results_c['avg_ms']:.1f}ms vs {results_python['avg_ms']:.1f}ms"
    )

    # Assert that C implementation has better memory behavior
    assert results_c["mem_growth_mb"] < results_python["mem_growth_mb"] * 0.5, (
        f"C implementation memory not better: {results_c['mem_growth_mb']:.1f}MB vs {results_python['mem_growth_mb']:.1f}MB"
    )

    _LOGGER.info("\n✅ All performance assertions passed!")


if __name__ == "__main__":
    asyncio.run(run_comparison())
