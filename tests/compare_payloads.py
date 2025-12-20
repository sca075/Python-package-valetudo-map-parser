#!/usr/bin/env python3
"""Compare multiple payloads to find robot angle pattern."""

import os
import struct
import sys


# Add the SCR directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "SCR")))

from valetudo_map_parser.config.rand25_parser import RRMapParser


def analyze_payload(payload_file: str, description: str):
    """Analyze a single payload file."""
    print(f"\n{'=' * 60}")
    print(f"ANALYZING: {description}")
    print(f"File: {payload_file}")
    print(f"{'=' * 60}")

    if not os.path.exists(payload_file):
        print(f"File not found: {payload_file}")
        return None

    with open(payload_file, "rb") as f:
        payload = f.read()

    print(f"Payload size: {len(payload)} bytes")

    # Parse with current parser
    parser = RRMapParser()
    result = parser.parse_data(payload, pixels=False)

    if result:
        robot_pos = result.get("robot", [0, 0])
        robot_angle = result.get("robot_angle", 0)
        path_data = result.get("path", {})
        path_points = len(path_data.get("points", []))
        path_angle = path_data.get("current_angle", 0)

        print("Parser Results:")
        print(f"  Robot position: {robot_pos}")
        print(f"  Robot angle: {robot_angle}")
        print(f"  Path points: {path_points}")
        print(f"  Path current_angle: {path_angle}")
    else:
        print("Parser failed!")
        return None

    # Find robot position block
    offset = 0x14  # Start after header
    robot_block_data = None

    while offset < len(payload) - 8:
        try:
            type_ = struct.unpack("<H", payload[offset : offset + 2])[0]
            hlength = struct.unpack("<H", payload[offset + 2 : offset + 4])[0]
            length = struct.unpack("<I", payload[offset + 4 : offset + 8])[0]

            if type_ == 8:  # Robot position block
                block_data_start = offset + 8
                if block_data_start + 8 <= len(payload):
                    x = struct.unpack(
                        "<I", payload[block_data_start : block_data_start + 4]
                    )[0]
                    y = struct.unpack(
                        "<I", payload[block_data_start + 4 : block_data_start + 8]
                    )[0]
                    robot_block_data = {
                        "position": [x, y],
                        "raw_bytes": payload[offset : offset + hlength + length].hex(),
                    }
                    print(f"Robot block raw data: {robot_block_data['raw_bytes']}")
                break

            offset = offset + hlength + length

        except (struct.error, IndexError):
            break

    return {
        "file": payload_file,
        "description": description,
        "robot_pos": robot_pos,
        "robot_angle": robot_angle,
        "path_angle": path_angle,
        "path_points": path_points,
        "robot_block_data": robot_block_data,
    }


def main():
    """Compare all three payloads."""
    payloads = [
        (
            "tests/map_data_20250728_185945.bin",
            "Original payload (robot at 12 o'clock?)",
        ),
        ("tests/map_data_20250728_193950.bin", "Second payload (robot at 11 o'clock)"),
        (
            "tests/map_data_20250728_194519.bin",
            "Third payload (robot at different angle)",
        ),
    ]

    results = []

    for payload_file, description in payloads:
        result = analyze_payload(payload_file, description)
        if result:
            results.append(result)

    # Compare results
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")

    print(
        f"{'Payload':<20} {'Robot Position':<20} {'Robot Angle':<12} {'Path Angle':<12} {'Path Points':<12}"
    )
    print("-" * 80)

    for result in results:
        pos_str = f"[{result['robot_pos'][0]}, {result['robot_pos'][1]}]"
        print(
            f"{result['description'][:19]:<20} {pos_str:<20} {result['robot_angle']:<12} {result['path_angle']:<12.1f} {result['path_points']:<12}"
        )

    # Look for patterns
    print(f"\n{'=' * 60}")
    print("PATTERN ANALYSIS")
    print(f"{'=' * 60}")

    if len(results) >= 2:
        print("Position changes:")
        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]
            dx = curr["robot_pos"][0] - prev["robot_pos"][0]
            dy = curr["robot_pos"][1] - prev["robot_pos"][1]
            print(
                f"  {prev['description'][:15]} -> {curr['description'][:15]}: dx={dx}, dy={dy}"
            )

        print("\nAngle changes:")
        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]
            angle_diff = curr["robot_angle"] - prev["robot_angle"]
            path_diff = curr["path_angle"] - prev["path_angle"]
            print(
                f"  {prev['description'][:15]} -> {curr['description'][:15]}: robot_angle_diff={angle_diff}, path_angle_diff={path_diff:.1f}"
            )

    # Check if robot angle correlates with position or path
    print("\nHypothesis: Robot angle might be calculated from position or path data")
    for result in results:
        x, y = result["robot_pos"]
        # Try to calculate angle from position (relative to some center point)
        # This is just a guess - we'd need to know the reference point
        print(
            f"  {result['description'][:20]}: pos=[{x}, {y}], reported_angle={result['robot_angle']}"
        )


if __name__ == "__main__":
    main()
