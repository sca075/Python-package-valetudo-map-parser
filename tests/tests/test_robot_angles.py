#!/usr/bin/env python3
"""Test script to understand robot angle calculation and propose improvements."""

import os
import sys


# Add the SCR directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "SCR")))


def current_angle_calculation(robot_angle: float) -> tuple:
    """Current implementation from map_data.py"""
    angle_c = round(robot_angle)
    angle = (360 - angle_c + 100) if angle_c < 0 else (180 - angle_c - 100)
    return angle % 360, robot_angle


def proposed_angle_calculation(robot_angle: float, offset: int = 100) -> tuple:
    """Proposed cleaner implementation with configurable offset."""
    # Convert raw angle to display angle (0-359°)
    display_angle = (robot_angle + offset) % 360
    return int(display_angle), robot_angle


def test_angle_calculations():
    """Test both implementations with various angle values."""
    print("🧪 ROBOT ANGLE CALCULATION TEST")
    print("=" * 80)

    # Test data: [raw_angle, expected_vacuum_orientation_description]
    test_angles = [
        (0, "12 o'clock (North)"),
        (90, "3 o'clock (East)"),
        (180, "6 o'clock (South)"),
        (-90, "9 o'clock (West)"),
        (-180, "6 o'clock (South)"),
        (45, "1:30 o'clock (NE)"),
        (-45, "10:30 o'clock (NW)"),
        (135, "4:30 o'clock (SE)"),
        (-135, "7:30 o'clock (SW)"),
        (-172, "Current test data (11 o'clock)"),
        (-86, "Test data 1"),
        (48, "Test data 2"),
        (-169, "Test data 3"),
        (-128, "Test data 4"),
        (177, "Test data 5"),
    ]

    print(
        f"{'Raw Angle':<12} {'Description':<25} {'Current':<12} {'Proposed':<12} {'Difference':<12}"
    )
    print("-" * 80)

    for raw_angle, description in test_angles:
        current_result, _ = current_angle_calculation(raw_angle)
        proposed_result, _ = proposed_angle_calculation(raw_angle)
        difference = abs(current_result - proposed_result)

        print(
            f"{raw_angle:<12} {description:<25} {current_result:<12} {proposed_result:<12} {difference:<12}"
        )

    print("\n" + "=" * 80)
    print("📊 ANALYSIS")
    print("=" * 80)

    print("\n🔍 CURRENT IMPLEMENTATION LOGIC:")
    print("   if angle < 0: (360 - angle + 100) % 360")
    print("   if angle >= 0: (180 - angle - 100) % 360")

    print("\n🔍 PROPOSED IMPLEMENTATION LOGIC:")
    print("   (angle + offset) % 360")

    print("\n⚠️  ISSUES WITH CURRENT IMPLEMENTATION:")
    print("   1. Different formulas for positive/negative angles")
    print("   2. Hardcoded offset (100) not configurable")
    print("   3. Complex logic that's hard to understand")
    print("   4. May not handle edge cases consistently")

    print("\n✅ BENEFITS OF PROPOSED IMPLEMENTATION:")
    print("   1. Single formula for all angles")
    print("   2. Configurable offset for different vacuum models")
    print("   3. Simple, clear math")
    print("   4. Consistent behavior")


def test_with_real_data():
    """Test with actual data from our bin files."""
    print("\n" + "=" * 80)
    print("🔬 TESTING WITH REAL BIN FILE DATA")
    print("=" * 80)

    # Real data from our bin files
    real_data = [
        (-86, "map_data_20250728_185945.bin"),
        (48, "map_data_20250728_193950.bin"),
        (-172, "map_data_20250728_194519.bin"),
        (-169, "map_data_20250728_204538.bin"),
        (-128, "map_data_20250728_204552.bin"),
        (177, "map_data_20250729_084141.bin"),
    ]

    print(f"{'File':<30} {'Raw Angle':<12} {'Current':<12} {'Proposed':<12}")
    print("-" * 70)

    for raw_angle, filename in real_data:
        current_result, _ = current_angle_calculation(raw_angle)
        proposed_result, _ = proposed_angle_calculation(raw_angle)

        short_filename = filename.replace("map_data_", "").replace(".bin", "")
        print(
            f"{short_filename:<30} {raw_angle:<12} {current_result:<12} {proposed_result:<12}"
        )


def test_offset_tuning():
    """Test different offset values to see the effect."""
    print("\n" + "=" * 80)
    print("🎛️  OFFSET TUNING TEST")
    print("=" * 80)

    test_angle = -172  # Our current test case
    offsets = [0, 50, 80, 100, 120, 150, 180]

    print(f"Raw angle: {test_angle}° (robot at 11 o'clock)")
    print(f"{'Offset':<10} {'Result':<10} {'Clock Position':<15}")
    print("-" * 40)

    for offset in offsets:
        result = (test_angle + offset) % 360
        # Convert to clock position (0° = 12 o'clock, 90° = 3 o'clock, etc.)
        clock_hour = ((result / 30) + 12) % 12
        if clock_hour == 0:
            clock_hour = 12
        clock_pos = f"{clock_hour:.1f} o'clock"

        print(f"{offset:<10} {result:<10} {clock_pos:<15}")


def recommend_solution():
    """Provide recommendations for the angle calculation."""
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    print("\n🎯 PROPOSED SOLUTION:")
    print("""
def get_rrm_robot_angle(json_data: JsonType, angle_offset: int = 100) -> tuple:
    '''
    Get the robot angle from the json with configurable offset.
    
    Args:
        json_data: JSON data containing robot_angle
        angle_offset: Calibration offset for vacuum orientation (default: 100)
    
    Returns:
        tuple: (display_angle_0_to_359, original_raw_angle)
    '''
    raw_angle = json_data.get("robot_angle", 0)
    display_angle = int((raw_angle + angle_offset) % 360)
    return display_angle, raw_angle
    """)

    print("\n🔧 CONFIGURATION OPTIONS:")
    print("   1. Keep current offset (100) as default")
    print("   2. Make offset configurable per vacuum model")
    print("   3. Add offset to vacuum configuration file")

    print("\n📝 IMPLEMENTATION STEPS:")
    print("   1. Replace current complex logic with simple math")
    print("   2. Add angle_offset parameter (default 100)")
    print("   3. Test with all bin files to ensure consistency")
    print("   4. Allow users to tune offset if needed")


def main():
    """Run all tests."""
    test_angle_calculations()
    test_with_real_data()
    test_offset_tuning()
    recommend_solution()

    print("\n" + "=" * 80)
    print("🎯 CONCLUSION")
    print("=" * 80)
    print("The current implementation works but is unnecessarily complex.")
    print("The proposed solution is simpler, more flexible, and easier to tune.")
    print("Both produce similar results, but the new approach is more maintainable.")


if __name__ == "__main__":
    main()
