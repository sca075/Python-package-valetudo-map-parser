from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import cProfile
import pstats
import tracemalloc
import psutil
import gc
import time
from typing import Dict, List, Tuple

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from SCR package
from SCR.valetudo_map_parser.config.colors import ColorsManagement
from SCR.valetudo_map_parser.config.shared import CameraSharedManager
from SCR.valetudo_map_parser.config.types import RoomStore
from SCR.valetudo_map_parser.hypfer_handler import HypferMapImageHandler

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Ensures DEBUG logs are displayed
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s (line %(lineno)d) - %(message)s",
)

_LOGGER = logging.getLogger(__name__)

# ----- Test Configuration -----
FRAME_COUNT = 10  # Set to 1/10/25/50/100 as needed
ENABLE_PROFILER = True  # Master switch for profiler usage
ENABLE_CPU_TIMING = False  # Lightweight per-frame CPU timing (process CPU time)
ENABLE_MEMORY_PROFILING = False  # Use tracemalloc snapshots
SNAPSHOT_EVERY_FRAME = False  # If False, snapshot only first and last frame
ENABLE_LEGACY_CPROFILE = False  # Legacy cProfile around the whole run
# ------------------------------


class PerformanceProfiler:
    """Comprehensive profiling for memory and CPU usage analysis."""

    def __init__(self, enable_memory_profiling: bool = ENABLE_MEMORY_PROFILING, enable_cpu_profiling: bool = ENABLE_CPU_TIMING):
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
        print("🎯 HYPFER COMPREHENSIVE PERFORMANCE REPORT")
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


class TestImageHandler:
    def __init__(self, enable_profiling: bool = True):
        self.test_data = None
        self.image = None

        # Initialize profiler
        self.profiler = PerformanceProfiler(
            enable_memory_profiling=ENABLE_MEMORY_PROFILING and ENABLE_PROFILER,
            enable_cpu_profiling=ENABLE_CPU_TIMING and ENABLE_PROFILER,
        ) if ENABLE_PROFILER else None
        # Always-on lightweight accumulators for per-frame stats (work even if profiler is off)
        self.wall_times_ms: list[float] = []
        self.cpu_times_ms: list[float] = []

    def set_up(self):
        """Set up test data with profiling."""
        if self.profiler:
            self.profiler.take_memory_snapshot("Test Setup Start")

        # Load the test.json file
        test_file_path = os.path.join(os.path.dirname(__file__), "glossyhardtofindnarwhal.json") #glossyhardtofindnarwhal
        _LOGGER.info(f"Loading test data from {test_file_path}")

        with open(test_file_path, "r") as file:
            self.test_data = json.load(file)
            _LOGGER.debug("Test data loaded.")

        if self.profiler:
            self.profiler.take_memory_snapshot("Test Setup Complete")

    async def test_image_handler(self):
        _LOGGER.info("Starting test_image_handler...")
        device_info = {
            'platform': 'mqtt_vacuum_camera',
            'unique_id': 'rockrobo_camera',
            'vacuum_config_entry': '9abe4e81972b00d1682c2363d3584041',
            'vacuum_map': 'valetudo/rockrobo',
            'vacuum_identifiers': {('mqtt', 'rockrobo')},
            'coordinator': "<custom_components.mqtt_vacuum_camera.coordinator.MQTTVacuumCoordinator object>",
            'is_rand256': False,
            'unsub_options_update_listener': "<function ConfigEntry.add_update_listener.<locals>.<lambda>>",
            'alpha_background': 255.0,
            'alpha_charger': 255.0,
            'alpha_go_to': 255.0,
            'alpha_move': 50.0,  # Higher alpha for better visibility
            'alpha_no_go': 125.0,
            'alpha_robot': 255.0,
            'alpha_text': 255.0,
            'alpha_wall': 150.0,  # Testing with a lower alpha value
            'alpha_zone_clean': 125.0,
            'aspect_ratio': '16, 9',
            'auto_zoom': True,
            'zoom_lock_ratio': True,
            'color_background': [0, 125, 255],
            'color_charger': [255, 128, 0],
            'color_go_to': [0, 255, 0],
            'color_move': [238, 247, 255],  # More vibrant blue for better visibility
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
            'alpha_room_0': 155.0,
            'alpha_room_1': 155.0,
            'alpha_room_2': 155.0,
            'alpha_room_3': 155.0,
            'alpha_room_4': 155.0,
            'alpha_room_5': 155.0,
            'alpha_room_6': 155.0,
            'alpha_room_7': 155.0,
            'alpha_room_8': 255.0,
            'alpha_room_9': 255.0,
            'alpha_room_10': 255.0,
            'alpha_room_11': 255.0,
            'alpha_room_12': 255.0,
            'alpha_room_13': 255.0,
            'alpha_room_14': 255.0,
            'alpha_room_15': 255.0,
            'robot_size': 20,
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
                "floor": "1",
                'trim_left': 1980,
                'trim_up': 1650,
                'trim_right': 3974,
                'trim_down': 3474
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
        file_name = "test_hypfer"
        shared_data = CameraSharedManager(file_name, device_info)
        shared = shared_data.get_instance()
        shared.vacuum_state = "docked"
        shared.vacuum_connection = True
        shared.vacuum_battery = 100
        shared.vacuum_ips = "192.168.8.1"

        # Active zones will be populated from the JSON data automatically
        # No need to manually set them here

        _LOGGER.debug(f"Shared instance trims: {shared.trims}")

        colors = ColorsManagement(shared)
        colors.set_initial_colours(device_info)

        _LOGGER.debug(f"Colors initialized: {shared.user_colors}")

        handler = HypferMapImageHandler(shared)

        # ImageDraw is already initialized in the handler constructor
        # Active zones will be populated from the JSON data during image processing

        # Lightweight per-frame CPU timing process handle
        proc = psutil.Process(os.getpid()) if ENABLE_CPU_TIMING else None

        # Run image generation N times to observe handler performance
        for i in range(FRAME_COUNT):
            iteration = i + 1

            # Optional sparse memory snapshot before frame
            if self.profiler and ENABLE_MEMORY_PROFILING:
                if SNAPSHOT_EVERY_FRAME or iteration in (1, FRAME_COUNT):
                    self.profiler.take_memory_snapshot(f"Before Image Generation #{iteration}")

            # Start timing (wall + CPU)
            cpu0 = proc.cpu_times() if proc else None
            start_time = time.time()

            # Get the image (PIL format)
            self.image = await handler.async_get_image(self.test_data, bytes_format=True)
            if shared.binary_image is None:
                _LOGGER.warning("❌ Binary image is None")
            else:
                _LOGGER.info(f"Image size: {len(shared.binary_image)} bytes")

            # End timing
            end_time = time.time()
            cpu1 = proc.cpu_times() if proc else None

            # Record timings (always capture into local accumulators)
            wall_ms = (end_time - start_time) * 1000.0
            self.wall_times_ms.append(wall_ms)

            cpu_used = None
            if proc and cpu0 and cpu1:
                cpu_used = (cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)
                wall = max(end_time - start_time, 1e-9)
                core_util = (cpu_used / wall) * 100.0
                cpu_ms = cpu_used * 1000.0
                self.cpu_times_ms.append(cpu_ms)
                _LOGGER.info(f"CPU/frame: {cpu_used:.3f}s | core-util: {core_util:.1f}% of one core")

            # Optional profiler bookkeeping
            if self.profiler:
                if ENABLE_MEMORY_PROFILING and (SNAPSHOT_EVERY_FRAME or iteration in (1, FRAME_COUNT)):
                    self.profiler.take_memory_snapshot(f"After Image Generation #{iteration}")
                self.profiler.time_operation(f"Image #{iteration}", start_time, end_time)
                if cpu_used is not None:
                    self.profiler.timing_stats.append({'label': f"CPU #{iteration}", 'duration_ms': cpu_ms, 'timestamp': start_time})

        # After loop: measure data sampling times separately
        t0=time.time(); calibration_data = handler.get_calibration_data(); t1=time.time()
        if self.profiler: self.profiler.time_operation("Calib", t0, t1)
        _LOGGER.info(f"Calibration_data: {calibration_data}")

        _LOGGER.info(f"PIL image size: {self.image.size}")
        store = RoomStore(file_name)
        t2=time.time(); rooms_data = await handler.async_get_rooms_attributes(); t3=time.time()
        if self.profiler: self.profiler.time_operation("Rooms", t2, t3)
        _LOGGER.info(f"Room Properties: {rooms_data}")

        # Robot in room timing (using existing robot_pos)
        rp = handler.robot_pos or {}
        rx, ry, ra = rp.get("x"), rp.get("y"), rp.get("angle")
        t4=time.time(); _ = await handler.imd.async_get_robot_in_room(robot_y=ry, robot_x=rx, angle=ra); t5=time.time()
        if self.profiler: self.profiler.time_operation("RobotRoom", t4, t5)



        _LOGGER.info(f"Calibration_data: {handler.get_calibration_data()}")
        _LOGGER.info(f"PIL image size: {self.image.size}")
        rooms_data = await handler.async_get_rooms_attributes()
        _LOGGER.info(f"Room Properties: {rooms_data}")
        count = store.get_rooms_count()
        _LOGGER.info(f"Room Store Properties: {count}")
        rooms = store.get_rooms()
        instaces = RoomStore.get_all_instances()
        _LOGGER.info(f"Room Store Rooms {instaces}: {rooms}")

        # Debug: Show RoomStore format like your real vacuum
        room_store_format = {}
        for room_id, room_data in rooms.items():
            room_store_format[room_id] = room_data['name']
        _LOGGER.info(f"RoomStore format (like your vacuum): {room_store_format}")

        # Debug: Show the room keys order
        room_keys = list(rooms.keys())
        _LOGGER.info(f"Room keys order: {room_keys}")

        # Debug: Show active zones mapping
        _LOGGER.info(f"Active zones: {handler.active_zones}")
        for i, active in enumerate(handler.active_zones or []):
            if i < len(room_keys):
                _LOGGER.info(f"Position {i}: Segment ID '{room_keys[i]}' ({rooms[room_keys[i]]['name']}) = active: {bool(active)}")
            else:
                _LOGGER.info(f"Position {i}: OUT_OF_BOUNDS = active: {bool(active)}")

        _LOGGER.info("=== TESTING YOUR VACUUM SCENARIO ===")

        _LOGGER.info(f"Trims update: {shared.trims.to_dict()}")
        calibration_data = handler.get_calibration_data()
        _LOGGER.info(f"Calibration Data: {calibration_data}")
        robot_x = handler.robot_pos.get("x")
        robot_y = handler.robot_pos.get("y")
        robot_angle = handler.robot_pos.get("angle")
        _LOGGER.info(f"Robot Position: ({robot_x}, {robot_y}, {robot_angle})")
        # Test the robot detection function
        result = await handler.imd.async_get_robot_in_room(robot_y=robot_y, robot_x=robot_x, angle=robot_angle)
        _LOGGER.info(f"Robot in room: {result}")

        # Check if zooming is enabled
        _LOGGER.info(f"Zooming enabled: {handler.imd.img_h.zooming}")

        # Check if all zooming conditions are met
        _LOGGER.info(f"Zoom conditions: zoom={handler.imd.img_h.zooming}, vacuum_state={handler.shared.vacuum_state}, image_auto_zoom={handler.shared.image_auto_zoom}")

        # Debug: Check the device_info auto_zoom setting
        _LOGGER.info(f"Device info auto_zoom: {device_info.get('auto_zoom')}")
        _LOGGER.info(f"Shared image_auto_zoom: {handler.shared.image_auto_zoom}")
        _LOGGER.info(f"Camera Attributes: {handler.shared.generate_attributes()}")
        # Debug: Check the zooming flags
        _LOGGER.info(f"Handler zooming: {handler.zooming}")
        _LOGGER.info(f"ImageDraw img_h zooming: {handler.imd.img_h.zooming}")
        _LOGGER.info(f"Active zones: {handler.imd.img_h.active_zones}")

        # Generate attributes to populate obstacles_data
        if shared.obstacles_pos:
            _LOGGER.info(f"Obstacles positions found: {shared.obstacles_pos}")
            # Call generate_attributes to populate obstacles_data
            attrs = shared.generate_attributes()
            _LOGGER.info(f"Generated attributes: {attrs}")
        else:
            _LOGGER.info("No obstacles positions found in the map data")

        _LOGGER.info(f"Obstacles data: {shared.obstacles_data}")


def __main__():
    # Enable comprehensive profiling (disable CPU profiling to avoid conflicts with main cProfile)
    test = TestImageHandler(enable_profiling=True)
    # Disable CPU profiling in the custom profiler to avoid conflicts
    if test.profiler:
        test.profiler.enable_cpu_profiling = False

    test.set_up()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Legacy cProfile for compatibility
    profiler = cProfile.Profile() if ENABLE_LEGACY_CPROFILE else None
    if profiler:
        profiler.enable()

    try:
        if test.profiler:
            test.profiler.take_memory_snapshot("Test Start")

        loop.run_until_complete(test.test_image_handler())

        if test.profiler:
            test.profiler.take_memory_snapshot("Test Complete")

    finally:
        if profiler:
            profiler.disable()
        loop.close()
        if test.image:
            test.image.show()

        if profiler:
            # Save profiling data
            profile_output = "hypfer_profile_output.prof"
            profiler.dump_stats(profile_output)

            # Print legacy profiling results
            print("\n" + "="*80)
            print("📊 LEGACY CPROFILE RESULTS (Top 50 functions)")
            print("="*80)
            stats = pstats.Stats(profile_output)
            stats.strip_dirs().sort_stats("cumulative").print_stats(50)

        # Generate comprehensive profiling report
        # Summarize adjusted stats (remove warm-up frame for steady-state)
        def _avg(values: list[float]) -> float:
            return sum(values) / max(len(values), 1)
    
        if hasattr(test, 'wall_times_ms') and test.wall_times_ms:
            steady_wall = test.wall_times_ms[1:] if len(test.wall_times_ms) > 1 else test.wall_times_ms
            steady_cpu = test.cpu_times_ms[1:] if len(test.cpu_times_ms) > 1 else test.cpu_times_ms
            print("\n=== ADJUSTED (steady-state, excl. warm-up) ===")
            if steady_wall:
                print(f"Image avg (ms): {_avg(steady_wall):.1f} | min: {min(steady_wall):.1f} | max: {max(steady_wall):.1f} | n={len(steady_wall)}")
            if steady_cpu:
                print(f"CPU avg (ms):   {_avg(steady_cpu):.1f} | min: {min(steady_cpu):.1f} | max: {max(steady_cpu):.1f} | n={len(steady_cpu)}")
    
            if test.profiler:
                test.profiler.generate_report()



if __name__ == "__main__":
    __main__()
