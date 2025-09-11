from __future__ import annotations

import asyncio
import logging
import cProfile
import pstats
import tracemalloc
import psutil
import gc
import time
from typing import Dict, List, Tuple

import sys
import os

# Add the parent directory to the path so we can import the SCR module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SCR.valetudo_map_parser.config.colors import ColorsManagement
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.rand256_handler import ReImageHandler
from SCR.valetudo_map_parser.config.rand256_parser import RRMapParser as Rand256Parser


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Ensures DEBUG logs are displayed
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s (line %(lineno)d) - %(message)s",
)

_LOGGER = logging.getLogger(__name__)


class PerformanceProfiler:
    """Comprehensive profiling for memory and CPU usage analysis."""

    def __init__(self, enable_memory_profiling: bool = True, enable_cpu_profiling: bool = True):
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

        self.memory_stats.append({
            'label': label,
            'timestamp': time.time(),
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': process.memory_percent(),
        })

        _LOGGER.debug(f"📊 Memory snapshot '{label}': RSS={memory_info.rss / 1024 / 1024:.1f}MB")

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
        self.timing_stats.append({
            'label': label,
            'duration_ms': duration * 1000,
            'timestamp': start_time
        })
        _LOGGER.info(f"⏱️  {label}: {duration * 1000:.1f}ms")

    def analyze_memory_usage(self) -> None:
        """Analyze memory usage patterns and print detailed report."""
        if not self.enable_memory_profiling or len(self.memory_snapshots) < 2:
            return

        print("\n" + "="*80)
        print("📊 MEMORY USAGE ANALYSIS")
        print("="*80)

        # Memory usage over time
        print("\n🔍 Memory Usage Timeline:")
        for i, stats in enumerate(self.memory_stats):
            print(f"  {i+1:2d}. {stats['label']:30s} | RSS: {stats['rss_mb']:6.1f}MB | VMS: {stats['vms_mb']:6.1f}MB | {stats['percent']:4.1f}%")

        # Memory growth analysis
        if len(self.memory_stats) >= 2:
            start_rss = self.memory_stats[0]['rss_mb']
            peak_rss = max(stats['rss_mb'] for stats in self.memory_stats)
            end_rss = self.memory_stats[-1]['rss_mb']

            print(f"\n📈 Memory Growth Analysis:")
            print(f"   Start RSS: {start_rss:.1f}MB")
            print(f"   Peak RSS:  {peak_rss:.1f}MB (+{peak_rss - start_rss:.1f}MB)")
            print(f"   End RSS:   {end_rss:.1f}MB ({'+' if end_rss > start_rss else ''}{end_rss - start_rss:.1f}MB from start)")

        # Top memory allocations
        if len(self.memory_snapshots) >= 2:
            print(f"\n🔥 Top Memory Allocations (comparing first vs last snapshot):")
            first_snapshot = self.memory_snapshots[0][1]
            last_snapshot = self.memory_snapshots[-1][1]

            top_stats = last_snapshot.compare_to(first_snapshot, 'lineno')[:10]
            for index, stat in enumerate(top_stats):
                print(f"   {index+1:2d}. {stat}")

    def analyze_cpu_usage(self) -> None:
        """Analyze CPU usage patterns and print detailed report."""
        if not self.enable_cpu_profiling or not self.cpu_profiles:
            return

        print("\n" + "="*80)
        print("⚡ CPU USAGE ANALYSIS")
        print("="*80)

        for label, profiler in self.cpu_profiles:
            print(f"\n🔍 CPU Profile: {label}")
            print("-" * 50)

            # Create stats object
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')

            # Print top 15 functions by cumulative time
            print("Top 15 functions by cumulative time:")
            stats.print_stats(15)

    def analyze_timing_patterns(self) -> None:
        """Analyze timing patterns across operations."""
        if not self.timing_stats:
            return

        print("\n" + "="*80)
        print("⏱️  TIMING ANALYSIS")
        print("="*80)

        # Group by operation type
        timing_groups = {}
        for stat in self.timing_stats:
            operation = stat['label'].split(' ')[0]  # Get first word as operation type
            if operation not in timing_groups:
                timing_groups[operation] = []
            timing_groups[operation].append(stat['duration_ms'])

        print("\n📊 Timing Summary by Operation:")
        for operation, durations in timing_groups.items():
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            print(f"   {operation:20s} | Avg: {avg_duration:6.1f}ms | Min: {min_duration:6.1f}ms | Max: {max_duration:6.1f}ms | Count: {len(durations)}")

    def generate_report(self) -> None:
        """Generate comprehensive performance report."""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE PERFORMANCE REPORT")
        print("="*80)

        self.analyze_memory_usage()
        self.analyze_cpu_usage()
        self.analyze_timing_patterns()

        # Garbage collection stats
        print(f"\n🗑️  Garbage Collection Stats:")
        gc_stats = gc.get_stats()
        for i, stats in enumerate(gc_stats):
            print(f"   Generation {i}: Collections={stats['collections']}, Collected={stats['collected']}, Uncollectable={stats['uncollectable']}")

        print("\n" + "="*80)


class TestRandImageHandler:
    def __init__(self, enable_profiling: bool = True):
        self.test_data = None
        self.image = None

        # Initialize profiler
        self.profiler = PerformanceProfiler(
            enable_memory_profiling=enable_profiling,
            enable_cpu_profiling=enable_profiling
        ) if enable_profiling else None

    def setUp(self):
        """Set up test data for first and last bin files."""
        _LOGGER.debug("Setting up test data for first and last bin files...")

        if self.profiler:
            self.profiler.take_memory_snapshot("Test Setup Start")

        # Test with first and last bin files
        # ("map_data_20250729_084141.bin", "LAST FILE - Multi-room with segments, vacuum at dock")
        self.test_files = [
            ("map_data_20250728_185945.bin", "FIRST FILE - Single room, vacuum at dock"),
            ("map_data_20250728_193950.bin", "FIRST FILE - Single room, vacuum at dock"),
            ("map_data_20250728_194519.bin", "FIRST FILE - Single room, vacuum at dock"),
            ("map_data_20250728_204538.bin", "FIRST FILE - Single room, vacuum at dock"),
            ("map_data_20250728_204552.bin", "FIRST FILE - Single room, vacuum at dock"),
            ("map_data_20250729_084141.bin", "LAST FILE - Multi-room with segments, vacuum at dock"),
        ]

        self.test_data_sets = []

        for filename, description in self.test_files:
            payload_file = os.path.join(os.path.dirname(__file__), filename)

            if not os.path.exists(payload_file):
                raise FileNotFoundError(f"Test payload file not found: {payload_file}")

            with open(payload_file, "rb") as f:
                payload = f.read()

            _LOGGER.debug(f"Loaded {filename}: {len(payload)} bytes")

            # Only use the new rand256 parser for both files
            import time
            self.new_rand256_parser = Rand256Parser()

            # Measure new_rand256 parser performance
            start_time = time.time()
            new_rand256_json = self.new_rand256_parser.parse_data(payload, pixels=True)
            parse_time = time.time() - start_time
            parsed_data = new_rand256_json
            self.test_data_sets.append({
                'filename': filename,
                'description': description,
                'payload': payload,
                'data': parsed_data,
                'json': new_rand256_json,
                'parse_time': parse_time
            })

            _LOGGER.debug(f"Parsed {filename} in {parse_time:.4f}s")

        # Display data for both files
        print("\n" + "="*80)
        print("NEW_RAND256_PARSER DATA COMPARISON")
        print("="*80)

        for i, dataset in enumerate(self.test_data_sets):
            print(f"\n📁 {dataset['description']}")
            print(f"   File: {dataset['filename']}")
            print(f"   Size: {len(dataset['payload']):,} bytes")
            print(f"   Parse time: {dataset['parse_time']:.4f} seconds")
            print(f"   JSON length: {len(dataset['json']):,} characters")

            data = dataset['data']
            if data:
                robot = data.get('robot', [0, 0])
                robot_angle = data.get('robot_angle', 0)
                charger = data.get('charger', [0, 0])
                path_data = data.get('path', {})
                path_points = len(path_data.get('points', []))
                path_angle = path_data.get('current_angle', 0)

                # Segments info
                image_data = data.get('image', {})
                segments = image_data.get('segments', {})
                segment_count = segments.get('count', 0)
                segment_ids = segments.get('id', [])

                print(f"   🤖 Robot: {robot}, Angle: {robot_angle}°")
                print(f"   🔌 Charger: {charger}")
                print(f"   🛤️  Path: {path_points} points, Angle: {path_angle:.1f}°")
                print(f"   🏠 Segments: {segment_count} rooms {segment_ids}")

                # Check if robot is at charger (close positions)
                if robot and charger:
                    distance = ((robot[0] - charger[0])**2 + (robot[1] - charger[1])**2)**0.5
                    at_dock = distance < 500  # Within 500 units
                    print(f"   🏠 At dock: {'✅ YES' if at_dock else '❌ NO'} (distance: {distance:.0f})")
            else:
                print("   ❌ PARSING FAILED")

        print("="*80)

        # Use the first dataset for the image test
        self.test_data = self.test_data_sets[0]['data']
        self.current_file = self.test_data_sets[0]['filename']

        _LOGGER.debug("Test data loaded and compared.")

    async def simulate_ha_background_task(self, task_name: str, duration: float):
        """Simulate Home Assistant background tasks like sensors, automations, etc."""
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < duration:
            # Simulate sensor updates
            await asyncio.sleep(0.1)
            # Simulate some CPU work
            _ = sum(i * i for i in range(1000))
            # Yield control back to event loop
            await asyncio.sleep(0)
        _LOGGER.debug(f"Background task {task_name} completed after {duration}s")

    def _generate_single_image_sync(self):
        """Synchronous wrapper for image generation (for asyncio.to_thread)."""
        # This will be called in a thread pool, so we need a new event loop
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._generate_single_image())
        finally:
            loop.close()

    async def test_image_handler(self):
        """Test image generation with Home Assistant environment simulation."""
        _LOGGER.info("Testing with Home Assistant concurrent environment simulation...")

        for i, dataset in enumerate(self.test_data_sets):
            print(f"\n{'='*80}")
            print(f"TESTING IMAGE GENERATION: {dataset['description']}")
            print(f"File: {dataset['filename']}")
            print(f"{'='*80}")

            # Set current test data
            self.test_data = dataset['data']
            self.current_file = dataset['filename']

            # Simulate Home Assistant concurrent environment
            background_tasks = []

            # Simulate various HA background operations
            background_tasks.append(self.simulate_ha_background_task("sensor_updates", 2.0))
            background_tasks.append(self.simulate_ha_background_task("automation_engine", 1.5))
            background_tasks.append(self.simulate_ha_background_task("state_machine", 1.8))
            background_tasks.append(self.simulate_ha_background_task("websocket_handler", 2.2))
            background_tasks.append(self.simulate_ha_background_task("recorder", 1.3))

            _LOGGER.info(f"🏠 Starting image generation {i+1}/{len(self.test_data_sets)} with HA simulation...")

            try:
                # Use asyncio.to_thread to simulate HA's approach
                image_task = asyncio.to_thread(self._generate_single_image_sync)

                # Run image generation concurrently with background tasks
                start_time = asyncio.get_event_loop().time()
                results = await asyncio.gather(
                    image_task,
                    *background_tasks,
                    return_exceptions=True
                )
                end_time = asyncio.get_event_loop().time()

                # Check if image generation succeeded
                image_result = results[0]
                if isinstance(image_result, Exception):
                    _LOGGER.error(f"❌ Image generation failed: {image_result}")
                    raise image_result

                _LOGGER.info(f"✅ Image {i+1}/{len(self.test_data_sets)} completed in {end_time - start_time:.3f}s with concurrent load")

            except Exception as e:
                _LOGGER.error(f"❌ Test failed for {dataset['filename']}: {e}")
                raise

    async def _generate_single_image(self):
        """Generate image for the current test data."""
        _LOGGER.info(f"Generating image for {self.current_file}...")

        # Start profiling for this image generation
        start_time = time.time()
        cpu_profiler = None
        if self.profiler:
            self.profiler.take_memory_snapshot(f"Before Image Gen - {self.current_file}")
            cpu_profiler = self.profiler.start_cpu_profile(f"Image Generation - {self.current_file}")

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
            'alpha_move': 100.0,
            'alpha_no_go': 125.0,
            'alpha_robot': 255.0,
            'alpha_text': 255.0,
            'alpha_wall': 115.0,  # Testing with a lower alpha value
            'alpha_zone_clean': 125.0,
            'aspect_ratio': '16, 9',
            'auto_zoom': True,
            'image_auto_zoom': True,  # Explicitly set image_auto_zoom to True
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
            'robot_size': 15,
            'offset_top': 0,
            'offset_bottom': 0,
            'offset_left': 10,
            'offset_right': 0,
            'rotate_image': '90',
            'margins': '100',
            'show_vac_status': True,
            'vac_status_font': 'SCR/valetudo_map_parser/config/fonts/FiraSans.ttf',
            'vac_status_position': True,
            'vac_status_size': 50.0,
            'enable_www_snapshots': False,
            'get_svg_file': False,
            'trims_data': {
                'trim_left': 0,
                'trim_up': 0,
                'trim_right': 0,
                'trim_down': 0
            },
            'disable_floor': False,
            'disable_wall': False,
            'disable_robot': False,
            'disable_charger': False,
            'disable_virtual_walls': False,
            'disable_restricted_areas': False,
            'disable_no_mop_areas': False,
            'disable_obstacles': False,
            'disable_path': False,
            'disable_predicted_path': False,
            'disable_go_to_target': False,
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
        shared.vacuum_state = "cleaning"
        shared.user_language = "it"

        # The room IDs in the test data are 16-20, but the handler uses an internal ID (0-4)
        # We need to set up the active zones array to match the internal IDs
        # Test with Living Room as active zone - zoom should be enabled
        shared.rand256_active_zone = [0] * 5  # Disable all rooms first
        shared.rand256_active_zone[4] = 0  # Enable Living Room (array index 4, segment ID 16)

        _LOGGER.debug(f"Shared instance trims: {shared.trims}")

        colors = ColorsManagement(shared)
        colors.set_initial_colours(device_info)

        _LOGGER.debug(f"Colors initialized: {shared.user_colors}")

        # Create a handler instance
        handler = ReImageHandler(shared)

        # Try to generate an image from the JSON data
        try:
            _LOGGER.info("Attempting to generate image from JSON data...")
            # Test with PNG output (WebP functionality kept in library but not used in test)
            self.image = await handler.async_get_image(self.test_data)
            _LOGGER.info("Successfully generated image from JSON data")
            if self.image is None:
                _LOGGER.error("Failed to generate image from JSON data")
                return
        except Exception as e:
            _LOGGER.warning(f"Error generating image from JSON: {e}")

        # Check if image generation was successful
        if self.image is None:
            _LOGGER.error("TEST FAILED: Image generation returned None")
            return
        else:
            _LOGGER.info("TEST PASSED: Image generated successfully")
            # Image should be PIL Image (not WebP bytes)
        if hasattr(self.image, 'size'):
            _LOGGER.info(f"PIL image size: {self.image.size}")
        else:
            _LOGGER.warning(f"Unexpected image type: {type(self.image)}")
        _LOGGER.info(f"Trims update: {shared.trims.to_dict()}")
        _LOGGER.info(f"Calibration_data: {handler.get_calibration_data()}")
        _LOGGER.info(await handler.get_rooms_attributes({
            "spots":[{"name":"test_point","coordinates":[25566,27289]}],
            "zones":[{"name":"test_zone","coordinates":[[20809,25919,22557,26582,1]]}],
            "rooms":[{"name":"Bathroom","id":19},
                     {"name":"Bedroom","id":20},
                     {"name":"Entrance","id":18},
                     {"name":"Kitchen","id":17},
                     {"name":"Living Room","id":16}],
                     "updated":1746298038728})
                     )
        # Get the robot room detection result
        robot_room_result = await handler.async_get_robot_in_room(25400, 25580, 0)
        _LOGGER.info(f"Robot in room: {robot_room_result}")

        # Check if the robot is in a room
        if robot_room_result and "in_room" in robot_room_result:
            room_name = robot_room_result["in_room"]
            _LOGGER.info(f"Robot is in room: {room_name}")

            # Check if the room has an ID in the robot_in_room property
            if handler.robot_in_room and "id" in handler.robot_in_room:
                room_id = handler.robot_in_room["id"]
                _LOGGER.info(f"Room ID: {room_id}")

                # Make sure the active zones array has enough elements
                if len(handler.shared.rand256_active_zone) > room_id:
                    _LOGGER.info(f"Active zone for room {room_id}: {handler.shared.rand256_active_zone[room_id]}")
                else:
                    _LOGGER.info(f"Room ID {room_id} is out of range for active zones array (length: {len(handler.shared.rand256_active_zone)})")
            else:
                _LOGGER.info("Robot is in a room but room_id is not set")
        else:
            _LOGGER.info("Robot is not in any room")

        # Check if image_auto_zoom is set correctly
        _LOGGER.info(f"image_auto_zoom: {handler.shared.image_auto_zoom}")

        # Check if all zooming conditions are met
        _LOGGER.info(f"Zoom conditions: zoom={handler.zooming}, vacuum_state={handler.shared.vacuum_state}, image_auto_zoom={handler.shared.image_auto_zoom}")

        _LOGGER.info(f"Zooming enabled: {handler.zooming}")
        # Show the image if successful
        if self.image is not None:
            print(f"\n🖼️  PROCESSING IMAGE: {self.current_file}")

            # Display PIL image directly without saving to disk
            if hasattr(self.image, 'size'):
                print(f"   📐 Image size: {self.image.size}")

                robot_in_room = getattr(handler, 'robot_in_room', 'Unknown')
                print(f"   🤖 Robot in room: {robot_in_room}")

                # Display the image directly
                self.image.show()
            else:
                print(f"   ❌ Unexpected image type: {type(self.image)}")
        else:
            print(f"\n❌ IMAGE GENERATION FAILED: {self.current_file}")

        # End profiling for this image generation
        end_time = time.time()
        if self.profiler:
            if cpu_profiler:
                self.profiler.stop_cpu_profile(cpu_profiler)
            self.profiler.take_memory_snapshot(f"After Image Gen - {self.current_file}")
            self.profiler.time_operation(f"Image Generation - {self.current_file}", start_time, end_time)


def __main__():
    # Enable comprehensive profiling (disable CPU profiling to avoid conflicts with main cProfile)
    test = TestRandImageHandler(enable_profiling=True)
    # Disable CPU profiling in the custom profiler to avoid conflicts
    if test.profiler:
        test.profiler.enable_cpu_profiling = False

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
        profile_output = "profile_output_rand.prof"
        profiler.dump_stats(profile_output)

        # Print legacy profiling results
        print("\n" + "="*80)
        print("📊 LEGACY CPROFILE RESULTS (Top 50 functions)")
        print("="*80)
        stats = pstats.Stats(profile_output)
        stats.strip_dirs().sort_stats("cumulative").print_stats(50)

        # Generate comprehensive profiling report
        if test.profiler:
            test.profiler.generate_report()


if __name__ == "__main__":
    __main__()
