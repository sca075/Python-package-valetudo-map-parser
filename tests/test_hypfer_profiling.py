#!/usr/bin/env python3
"""
Profiling test for Hypfer vacuum image generation.
This test includes comprehensive memory and CPU profiling capabilities.
"""

import asyncio
import cProfile
import gc
import logging
import os
import pstats
import sys
import time
import tracemalloc
from typing import Dict, List, Tuple

import psutil


# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.hypfer_handler import HypferMapImageHandler


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

_LOGGER = logging.getLogger(__name__)


class PerformanceProfiler:
    """Comprehensive profiling for memory and CPU usage analysis."""

    def __init__(
        self, enable_memory_profiling: bool = True, enable_cpu_profiling: bool = True
    ):
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_cpu_profiling = enable_cpu_profiling
        self.memory_snapshots: List[Tuple[str, tracemalloc.Snapshot]] = []
        self.cpu_profiles: List[Tuple[str, cProfile.Profile]] = []
        self.memory_stats: List[Dict] = []
        self.timing_stats: List[Dict] = []

        if self.enable_memory_profiling:
            tracemalloc.start()
            _LOGGER.info("🔍 Memory profiling enabled")

        if self.enable_cpu_profiling:
            _LOGGER.info("⚡ CPU profiling enabled")

    def take_memory_snapshot(self, label: str) -> None:
        """Take a memory snapshot with a descriptive label."""
        if not self.enable_memory_profiling:
            return

        snapshot = tracemalloc.take_snapshot()
        self.memory_snapshots.append((label, snapshot))

        # Get current memory usage
        process = psutil.Process()
        memory_info = process.memory_info()

        self.memory_stats.append(
            {
                "label": label,
                "timestamp": time.time(),
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
                "percent": process.memory_percent(),
            }
        )

        _LOGGER.debug(
            f"📊 Memory snapshot '{label}': RSS={memory_info.rss / 1024 / 1024:.1f}MB"
        )

    def start_cpu_profile(self, label: str) -> cProfile.Profile:
        """Start CPU profiling for a specific operation."""
        if not self.enable_cpu_profiling:
            return None

        profiler = cProfile.Profile()
        profiler.enable()
        self.cpu_profiles.append((label, profiler))
        return profiler

    def stop_cpu_profile(self, profiler: cProfile.Profile) -> None:
        """Stop CPU profiling."""
        if profiler:
            profiler.disable()

    def time_operation(self, label: str, start_time: float, end_time: float) -> None:
        """Record timing information for an operation."""
        duration = end_time - start_time
        self.timing_stats.append(
            {"label": label, "duration_ms": duration * 1000, "timestamp": start_time}
        )
        _LOGGER.info(f"⏱️  {label}: {duration * 1000:.1f}ms")

    def generate_report(self) -> None:
        """Generate comprehensive performance report."""
        print("\n" + "=" * 80)
        print("🎯 HYPFER COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 80)

        # Memory usage analysis
        if self.memory_stats:
            print("\n🔍 Memory Usage Timeline:")
            for i, stats in enumerate(self.memory_stats):
                print(
                    f"  {i + 1:2d}. {stats['label']:30s} | RSS: {stats['rss_mb']:6.1f}MB | VMS: {stats['vms_mb']:6.1f}MB | {stats['percent']:4.1f}%"
                )

        # Timing analysis
        if self.timing_stats:
            print("\n⏱️  Timing Summary:")
            for stat in self.timing_stats:
                print(f"   {stat['label']:40s} | {stat['duration_ms']:6.1f}ms")

        # Garbage collection stats
        print("\n🗑️  Garbage Collection Stats:")
        gc_stats = gc.get_stats()
        for i, stats in enumerate(gc_stats):
            print(
                f"   Generation {i}: Collections={stats['collections']}, Collected={stats['collected']}, Uncollectable={stats['uncollectable']}"
            )

        print("\n" + "=" * 80)


class TestHypferImageHandler:
    def __init__(self, enable_profiling: bool = True):
        self.test_data = None
        self.image = None

        # Initialize profiler
        self.profiler = (
            PerformanceProfiler(
                enable_memory_profiling=enable_profiling,
                enable_cpu_profiling=enable_profiling,
            )
            if enable_profiling
            else None
        )

    def setUp(self):
        """Set up test data for Hypfer vacuum."""
        _LOGGER.debug("Setting up test data for Hypfer vacuum...")

        if self.profiler:
            self.profiler.take_memory_snapshot("Test Setup Start")

        # Sample Hypfer JSON data (you would replace this with real data)
        self.test_data = {
            "metaData": {"version": "1.0.0", "nonce": 123456789},
            "size": {"x": 1600, "y": 900},
            "pixelSize": 5,
            "layers": [
                {
                    "type": "floor",
                    "pixels": [],  # Add real floor data here
                },
                {
                    "type": "wall",
                    "pixels": [],  # Add real wall data here
                },
            ],
            "entities": [
                {
                    "type": "robot_position",
                    "points": [800, 450],
                    "metaData": {"angle": 90},
                }
            ],
        }

        if self.profiler:
            self.profiler.take_memory_snapshot("Test Setup Complete")

    async def test_image_handler(self):
        """Test image generation with profiling."""
        _LOGGER.info("Testing Hypfer image generation with profiling...")

        # Start profiling for image generation
        start_time = time.time()
        if self.profiler:
            self.profiler.take_memory_snapshot("Before Image Generation")
            cpu_profiler = self.profiler.start_cpu_profile("Hypfer Image Generation")

        try:
            # Create device info (similar to real Home Assistant setup)
            device_info = {
                "platform": "mqtt_vacuum_camera",
                "unique_id": "hypfer_camera",
                "vacuum_config_entry": "test_entry_id",
                "vacuum_map": "valetudo/hypfer",
                "vacuum_identifiers": {("mqtt", "hypfer")},
                "is_rand256": False,
                "alpha_background": 255.0,
                "color_background": [0, 125, 255],
                "aspect_ratio": "1, 1",
                "auto_zoom": False,
                "margins": "100",
                "rotate_image": "0",
                "show_vac_status": False,
                "enable_www_snapshots": False,
                "get_svg_file": False,
            }

            # Create shared data manager
            shared_data = CameraSharedManager("test_hypfer", device_info)
            shared = shared_data.get_instance()

            # Create handler
            handler = HypferMapImageHandler(shared)

            # Generate image
            self.image = await handler.get_image_from_json(
                self.test_data, return_webp=False
            )

            # Display results
            if self.image is not None:
                print("\n🖼️  HYPFER IMAGE GENERATED SUCCESSFULLY")
                if hasattr(self.image, "size"):
                    print(f"   📐 Image size: {self.image.size}")
                    # Optionally display the image
                    # self.image.show()
                else:
                    print(f"   ❌ Unexpected image type: {type(self.image)}")
            else:
                print("\n❌ HYPFER IMAGE GENERATION FAILED")

        except Exception as e:
            _LOGGER.error(f"❌ Hypfer test failed: {e}")
            raise

        finally:
            # End profiling
            end_time = time.time()
            if self.profiler:
                self.profiler.stop_cpu_profile(cpu_profiler)
                self.profiler.take_memory_snapshot("After Image Generation")
                self.profiler.time_operation(
                    "Hypfer Image Generation", start_time, end_time
                )


def __main__():
    # Enable comprehensive profiling
    test = TestHypferImageHandler(enable_profiling=True)
    test.setUp()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Legacy cProfile for compatibility
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        if test.profiler:
            test.profiler.take_memory_snapshot("Test Start")

        loop.run_until_complete(test.test_image_handler())

        if test.profiler:
            test.profiler.take_memory_snapshot("Test Complete")

    finally:
        profiler.disable()
        loop.close()

        # Save profiling data
        profile_output = "profile_output_hypfer.prof"
        profiler.dump_stats(profile_output)

        # Print legacy profiling results
        print("\n" + "=" * 80)
        print("📊 LEGACY CPROFILE RESULTS (Top 50 functions)")
        print("=" * 80)
        stats = pstats.Stats(profile_output)
        stats.strip_dirs().sort_stats("cumulative").print_stats(50)

        # Generate comprehensive profiling report
        if test.profiler:
            test.profiler.generate_report()


if __name__ == "__main__":
    __main__()
