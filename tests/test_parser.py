#!/usr/bin/env python3
"""Test script to compare rand25_parser vs rand256_parser with real vacuum data."""

import json
import os
import sys
from pathlib import Path


# Add the SCR directory to Python path
current_dir = Path(__file__).parent
scr_path = current_dir.parent / "SCR"
sys.path.insert(0, str(scr_path))

from backups.rand256_parser_backup import RRMapParser as Rand25Parser
from valetudo_map_parser.config.rand256_parser import RRMapParser as Rand256Parser


def load_payload(payload_file: str) -> bytes:
    """Load a saved payload file."""
    print(f"Loading payload from: {payload_file}")
    with open(payload_file, "rb") as f:
        data = f.read()
    print(f"Loaded {len(data)} bytes")
    return data


def test_parsers():
    """Test both parsers with the saved map data."""
    # Look for the map data file
    payload_file = "map_data_20250728_185945.bin"

    # Try different possible locations
    possible_paths = [
        payload_file,
        f"tests/{payload_file}",
        f"../{payload_file}",
        f"/tmp/vacuum_payloads/{payload_file}",
        "tests/map_data_20250728_185945.bin",
    ]

    payload_path = None
    for path in possible_paths:
        if os.path.exists(path):
            payload_path = path
            break

    if not payload_path:
        print(f"Could not find payload file: {payload_file}")
        print("Tried these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        return

    # Load the payload
    try:
        payload = load_payload(payload_path)
    except Exception as e:
        print(f"Error loading payload: {e}")
        return

    print(f"\n{'=' * 60}")
    print("TESTING PARSERS WITH REAL VACUUM DATA")
    print(f"{'=' * 60}")

    results = {}

    # Test rand25_parser (current)
    print(f"\n{'=' * 20} RAND25 PARSER (Current) {'=' * 20}")
    rand25 = Rand25Parser()
    try:
        result25 = rand25.parse_data(payload, pixels=True)
        if result25:
            print("✅ rand25 parser succeeded")

            # Extract key data
            robot_data = result25.get("robot", [])
            robot_angle = result25.get("robot_angle", 0)
            charger_data = result25.get("charger", [])
            image_data = result25.get("image", {})

            print(f"Robot position: {robot_data}")
            print(f"Robot angle: {robot_angle}")
            print(f"Charger position: {charger_data}")
            print(f"Image dimensions: {image_data.get('dimensions', {})}")
            print(
                f"Segments found: {len(image_data.get('segments', {}).get('id', []))}"
            )

            results["rand25"] = {
                "success": True,
                "robot": robot_data,
                "robot_angle": robot_angle,
                "charger": charger_data,
                "image_dimensions": image_data.get("dimensions", {}),
                "segments_count": len(image_data.get("segments", {}).get("id", [])),
                "segments_ids": image_data.get("segments", {}).get("id", []),
                "full_data": result25,
            }
        else:
            print("❌ rand25 parser returned None")
            results["rand25"] = {"success": False, "error": "Parser returned None"}
    except Exception as e:
        print(f"❌ ERROR in rand25 parser: {e}")
        results["rand25"] = {"success": False, "error": str(e)}

    # Test rand256_parser (new)
    print(f"\n{'=' * 20} RAND256 PARSER (New) {'=' * 20}")
    rand256 = Rand256Parser()
    try:
        result256 = rand256.parse_data(payload, pixels=True)
        if result256:
            print("✅ rand256 parser succeeded")

            # Extract key data
            robot_data = result256.get("robot", [])
            robot_angle = result256.get("robot_angle", 0)
            charger_data = result256.get("charger", [])
            image_data = result256.get("image", {})

            print(f"Robot position: {robot_data}")
            print(f"Robot angle: {robot_angle}")
            print(f"Charger position: {charger_data}")
            print(f"Image dimensions: {image_data.get('dimensions', {})}")
            print(
                f"Segments found: {len(image_data.get('segments', {}).get('id', []))}"
            )

            results["rand256"] = {
                "success": True,
                "robot": robot_data,
                "robot_angle": robot_angle,
                "charger": charger_data,
                "image_dimensions": image_data.get("dimensions", {}),
                "segments_count": len(image_data.get("segments", {}).get("id", [])),
                "segments_ids": image_data.get("segments", {}).get("id", []),
                "full_data": result256,
            }
        else:
            print("❌ rand256 parser returned None")
            results["rand256"] = {"success": False, "error": "Parser returned None"}
    except Exception as e:
        print(f"❌ ERROR in rand256 parser: {e}")
        results["rand256"] = {"success": False, "error": str(e)}

    # Compare results
    print(f"\n{'=' * 25} COMPARISON {'=' * 25}")

    if results["rand25"]["success"] and results["rand256"]["success"]:
        r25 = results["rand25"]
        r256 = results["rand256"]

        print("\n🔍 ROBOT POSITION:")
        if r25["robot"] == r256["robot"]:
            print(f"  ✅ MATCH: {r25['robot']}")
        else:
            print("  ⚠️  DIFFER:")
            print(f"    rand25:  {r25['robot']}")
            print(f"    rand256: {r256['robot']}")

        print("\n🔍 ROBOT ANGLE:")
        if r25["robot_angle"] == r256["robot_angle"]:
            print(f"  ✅ MATCH: {r25['robot_angle']}")
        else:
            print("  ⚠️  DIFFER:")
            print(f"    rand25:  {r25['robot_angle']}")
            print(f"    rand256: {r256['robot_angle']}")

        print("\n🔍 CHARGER POSITION:")
        if r25["charger"] == r256["charger"]:
            print(f"  ✅ MATCH: {r25['charger']}")
        else:
            print("  ⚠️  DIFFER:")
            print(f"    rand25:  {r25['charger']}")
            print(f"    rand256: {r256['charger']}")

        print("\n🔍 IMAGE DIMENSIONS:")
        if r25["image_dimensions"] == r256["image_dimensions"]:
            print(f"  ✅ MATCH: {r25['image_dimensions']}")
        else:
            print("  ⚠️  DIFFER:")
            print(f"    rand25:  {r25['image_dimensions']}")
            print(f"    rand256: {r256['image_dimensions']}")

        print("\n🔍 SEGMENTS:")
        if r25["segments_ids"] == r256["segments_ids"]:
            print(f"  ✅ MATCH: {r25['segments_count']} segments")
            print(f"    IDs: {r25['segments_ids']}")
        else:
            print("  ⚠️  DIFFER:")
            print(
                f"    rand25:  {r25['segments_count']} segments, IDs: {r25['segments_ids']}"
            )
            print(
                f"    rand256: {r256['segments_count']} segments, IDs: {r256['segments_ids']}"
            )

    # Save results to JSON file
    output_file = "tests/test_rand256.json"
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"\n❌ Error saving results: {e}")

    print(f"\n{'=' * 60}")
    print("TEST COMPLETE")
    print(f"{'=' * 60}")

    return results


def main():
    """Main test function."""
    print("🧪 VACUUM MAP PARSER COMPARISON TEST")
    print("=" * 60)

    results = test_parsers()

    # Summary
    if results:
        print("\n📊 SUMMARY:")
        for parser_name, result in results.items():
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            print(f"  {parser_name.upper()}: {status}")
            if not result["success"]:
                print(f"    Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
