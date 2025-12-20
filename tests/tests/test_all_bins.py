#!/usr/bin/env python3
"""Test new_rand256_parser with all available .bin files."""

import json
import os
import sys
import time
from typing import Any, Dict


# Add the SCR directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "SCR")))

from valetudo_map_parser.config.new_rand256_parser import (
    RRMapParser as NewRand256Parser,
)
from valetudo_map_parser.config.rand25_parser import RRMapParser as Rand25Parser
from valetudo_map_parser.config.rand256_parser import RRMapParser as Rand256Parser


def test_parser_with_file(filename: str) -> Dict[str, Any]:
    """Test all three parsers with a single file."""
    print(f"\n{'=' * 80}")
    print(f"TESTING: {filename}")
    print(f"{'=' * 80}")

    filepath = os.path.join("..", filename)
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return {"error": f"File not found: {filepath}"}

    # Load payload
    with open(filepath, "rb") as f:
        payload = f.read()

    print(f"📁 File size: {len(payload):,} bytes")

    results = {}

    # Test each parser
    parsers = [
        ("RAND25", Rand25Parser()),
        ("RAND256", Rand256Parser()),
        ("NEW_RAND256", NewRand256Parser()),
    ]

    for parser_name, parser in parsers:
        try:
            start_time = time.time()
            result = parser.parse_data(payload, pixels=False)
            parse_time = time.time() - start_time

            if result is None:
                print(f"❌ {parser_name}: FAILED - returned None")
                results[parser_name] = {
                    "error": "Parser returned None",
                    "time": parse_time,
                }
                continue

            # For new parser, result is JSON string, parse it back
            if parser_name == "NEW_RAND256" and isinstance(result, str):
                try:
                    parsed_result = json.loads(result)
                    json_length = len(result)
                except json.JSONDecodeError as e:
                    print(f"❌ {parser_name}: FAILED - Invalid JSON: {e}")
                    results[parser_name] = {
                        "error": f"Invalid JSON: {e}",
                        "time": parse_time,
                    }
                    continue
            else:
                parsed_result = result
                json_length = 0

            # Extract key data
            robot = parsed_result.get("robot", [0, 0])
            robot_angle = parsed_result.get("robot_angle", 0)
            charger = parsed_result.get("charger", [0, 0])
            path_data = parsed_result.get("path", {})
            path_points = len(path_data.get("points", []))
            path_angle = path_data.get("current_angle", 0)
            image_data = parsed_result.get("image", {})
            segments = image_data.get("segments", {})
            segment_count = segments.get("count", 0)
            segment_ids = segments.get("id", [])

            results[parser_name] = {
                "success": True,
                "time": parse_time,
                "json_length": json_length,
                "robot": robot,
                "robot_angle": robot_angle,
                "charger": charger,
                "path_points": path_points,
                "path_angle": path_angle,
                "segment_count": segment_count,
                "segment_ids": segment_ids,
            }

            print(f"✅ {parser_name}: SUCCESS ({parse_time:.4f}s)")
            print(f"   Robot: {robot}, Angle: {robot_angle}")
            print(f"   Path: {path_points} points, Angle: {path_angle:.1f}°")
            print(f"   Segments: {segment_count} ({segment_ids})")
            if json_length > 0:
                print(f"   JSON: {json_length:,} characters")

        except Exception as e:
            print(f"❌ {parser_name}: EXCEPTION - {e}")
            results[parser_name] = {"error": str(e), "time": 0}

    return results


def compare_results(results: Dict[str, Dict[str, Any]], filename: str):
    """Compare results between parsers."""
    print(f"\n📊 COMPARISON FOR {filename}:")
    print("-" * 60)

    # Check if all parsers succeeded
    successful_parsers = [
        name for name, result in results.items() if result.get("success")
    ]
    failed_parsers = [
        name for name, result in results.items() if not result.get("success")
    ]

    if failed_parsers:
        print(f"❌ FAILED PARSERS: {', '.join(failed_parsers)}")

    if len(successful_parsers) < 2:
        print("❌ Not enough successful parsers to compare")
        return

    # Compare data between successful parsers
    base_parser = successful_parsers[0]
    base_result = results[base_parser]

    print("📈 PERFORMANCE COMPARISON:")
    for parser_name in successful_parsers:
        result = results[parser_name]
        time_diff = (
            ((result["time"] / base_result["time"] - 1) * 100)
            if base_result["time"] > 0
            else 0
        )
        print(f"   {parser_name}: {result['time']:.4f}s ({time_diff:+.1f}%)")

    print("\n🔍 DATA COMPARISON:")
    data_fields = [
        "robot",
        "robot_angle",
        "charger",
        "path_points",
        "path_angle",
        "segment_count",
        "segment_ids",
    ]

    all_match = True
    for field in data_fields:
        values = [
            results[parser][field]
            for parser in successful_parsers
            if field in results[parser]
        ]
        if len(set(str(v) for v in values)) == 1:
            print(f"   ✅ {field}: {values[0]} (ALL MATCH)")
        else:
            print(f"   ❌ {field}: MISMATCH")
            for parser in successful_parsers:
                if field in results[parser]:
                    print(f"      {parser}: {results[parser][field]}")
            all_match = False

    if all_match:
        print("\n🎉 ALL DATA MATCHES PERFECTLY!")
    else:
        print("\n⚠️  DATA MISMATCHES FOUND!")


def main():
    """Test all .bin files."""
    print("🧪 TESTING NEW_RAND256_PARSER WITH ALL BIN FILES")
    print("=" * 80)

    # Find all .bin files
    bin_files = [f for f in os.listdir("..") if f.endswith(".bin")]
    bin_files.sort()

    print(f"📁 Found {len(bin_files)} .bin files:")
    for f in bin_files:
        print(f"   - {f}")

    all_results = {}

    # Test each file
    for filename in bin_files:
        results = test_parser_with_file(filename)
        all_results[filename] = results
        compare_results(results, filename)

    # Overall summary
    print(f"\n{'=' * 80}")
    print("📋 OVERALL SUMMARY")
    print(f"{'=' * 80}")

    total_files = len(bin_files)
    successful_files = 0
    performance_improvements = []

    for filename, results in all_results.items():
        if "NEW_RAND256" in results and results["NEW_RAND256"].get("success"):
            successful_files += 1

            # Calculate performance improvement vs RAND25
            if "RAND25" in results and results["RAND25"].get("success"):
                old_time = results["RAND25"]["time"]
                new_time = results["NEW_RAND256"]["time"]
                if old_time > 0:
                    improvement = ((old_time - new_time) / old_time) * 100
                    performance_improvements.append(improvement)

    print(
        f"✅ NEW_RAND256 SUCCESS RATE: {successful_files}/{total_files} ({successful_files / total_files * 100:.1f}%)"
    )

    if performance_improvements:
        avg_improvement = sum(performance_improvements) / len(performance_improvements)
        min_improvement = min(performance_improvements)
        max_improvement = max(performance_improvements)
        print("🚀 PERFORMANCE IMPROVEMENT:")
        print(f"   Average: {avg_improvement:.1f}% faster")
        print(f"   Range: {min_improvement:.1f}% to {max_improvement:.1f}% faster")

    print("\n🎯 CONCLUSION:")
    if successful_files == total_files:
        print("   ✅ NEW_RAND256_PARSER WORKS PERFECTLY WITH ALL FILES!")
        print("   ✅ READY FOR PRODUCTION USE!")
    else:
        print(
            f"   ⚠️  NEW_RAND256_PARSER FAILED ON {total_files - successful_files} FILES"
        )
        print("   🔧 NEEDS INVESTIGATION BEFORE PRODUCTION USE")


if __name__ == "__main__":
    main()
