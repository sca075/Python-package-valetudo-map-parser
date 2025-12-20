#!/usr/bin/env python3
"""Test script to profile and compare rand25_parser vs rand256_parser processing times."""

import os
import statistics
import sys
import time
from pathlib import Path


# Add the SCR directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "SCR"))

from backups.new_rand256_parser import RRMapParser as Rand256Parser
from backups.rand256_parser_backup import RRMapParser as Rand25Parser


def load_payload(payload_file: str) -> bytes:
    """Load a saved payload file."""
    with open(payload_file, "rb") as f:
        return f.read()


def profile_parser(
    parser, parser_name: str, payload: bytes, pixels: bool = False, runs: int = 5
) -> dict:
    """Profile a parser with multiple runs and return timing statistics."""
    print(f"\n🔍 Profiling {parser_name} ({runs} runs)...")

    times = []
    results = []
    errors = []

    for run in range(runs):
        try:
            start_time = time.perf_counter()
            result = parser.parse_data(payload, pixels=pixels)
            end_time = time.perf_counter()

            parse_time = end_time - start_time
            times.append(parse_time)
            results.append(result is not None)

            print(f"  Run {run + 1}: {parse_time:.4f}s {'✅' if result else '❌'}")

        except Exception as e:
            print(f"  Run {run + 1}: ERROR - {e}")
            errors.append(str(e))

    if not times:
        return {
            "parser": parser_name,
            "success": False,
            "error": "All runs failed",
            "errors": errors,
        }

    # Calculate statistics
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    median_time = statistics.median(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0

    success_rate = sum(results) / len(results) * 100

    print("  📊 Results:")
    print(f"    Average: {avg_time:.4f}s")
    print(f"    Min:     {min_time:.4f}s")
    print(f"    Max:     {max_time:.4f}s")
    print(f"    Median:  {median_time:.4f}s")
    print(f"    Std Dev: {std_dev:.4f}s")
    print(f"    Success: {success_rate:.1f}%")

    return {
        "parser": parser_name,
        "success": True,
        "runs": runs,
        "times": times,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "median_time": median_time,
        "std_dev": std_dev,
        "success_rate": success_rate,
        "errors": errors,
    }


def compare_parsers(payload_file: str, runs: int = 5):
    """Profile and compare both parsers."""
    print(f"\n{'=' * 60}")
    print("PARSER PERFORMANCE PROFILING")
    print(f"{'=' * 60}")
    print(f"Payload file: {payload_file}")

    # Load the payload
    payload = load_payload(payload_file)
    print(f"Payload size: {len(payload):,} bytes")

    # Profile both parsers
    rand25_stats = profile_parser(
        Rand25Parser(), "RAND25", payload, pixels=True, runs=runs
    )
    rand256_stats = profile_parser(
        Rand256Parser(), "RAND256", payload, pixels=True, runs=runs
    )

    # Compare performance
    print(f"\n{'=' * 30} COMPARISON {'=' * 30}")

    if rand25_stats["success"] and rand256_stats["success"]:
        rand25_avg = rand25_stats["avg_time"]
        rand256_avg = rand256_stats["avg_time"]

        # Use a small threshold to determine if times are essentially equal
        threshold = 0.0001  # 0.1ms threshold
        time_diff = abs(rand25_avg - rand256_avg)

        if time_diff <= threshold:
            print("🤝 Both parsers have IDENTICAL performance")
            print(f"   RAND25:  {rand25_avg:.4f}s (avg)")
            print(f"   RAND256: {rand256_avg:.4f}s (avg)")
            print(
                f"   Difference: {time_diff:.6f}s (within {threshold:.4f}s threshold)"
            )
        elif rand25_avg < rand256_avg:
            speedup = rand256_avg / rand25_avg
            print(f"🏆 RAND25 is FASTER by {speedup:.2f}x")
            print(f"   RAND25:  {rand25_avg:.4f}s (avg)")
            print(f"   RAND256: {rand256_avg:.4f}s (avg)")
            print(f"   Difference: {time_diff:.6f}s")
        else:  # rand256_avg < rand25_avg
            speedup = rand25_avg / rand256_avg
            print(f"🏆 RAND256 is FASTER by {speedup:.2f}x")
            print(f"   RAND256: {rand256_avg:.4f}s (avg)")
            print(f"   RAND25:  {rand25_avg:.4f}s (avg)")
            print(f"   Difference: {time_diff:.6f}s")

        # Show detailed comparison
        print("\n📈 Detailed Performance:")
        print(f"   {'Metric':<12} {'RAND25':<12} {'RAND256':<12} {'Winner'}")
        print(f"   {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")

        metrics = [
            ("Average", "avg_time"),
            ("Minimum", "min_time"),
            ("Maximum", "max_time"),
            ("Median", "median_time"),
            ("Std Dev", "std_dev"),
        ]

        for metric_name, metric_key in metrics:
            r25_val = rand25_stats[metric_key]
            r256_val = rand256_stats[metric_key]

            # Use threshold for determining winner
            threshold = 0.0001 if metric_key != "std_dev" else 0.00001
            diff = abs(r25_val - r256_val)

            if diff <= threshold:
                winner = "TIE"
            elif r25_val < r256_val:
                winner = "RAND25"
            else:
                winner = "RAND256"

            print(f"   {metric_name:<12} {r25_val:<12.4f} {r256_val:<12.4f} {winner}")

    return rand25_stats, rand256_stats


def test_with_pixels(payload_file: str, runs: int = 3):
    """Test parsers with pixel data enabled (more intensive)."""
    print(f"\n{'=' * 60}")
    print("PARSER PROFILING WITH PIXEL DATA")
    print(f"{'=' * 60}")
    print(f"Payload file: {payload_file}")

    payload = load_payload(payload_file)
    print(f"Payload size: {len(payload):,} bytes")

    # Profile with pixel data (more intensive)
    rand25_stats = profile_parser(
        Rand25Parser(), "RAND25 (pixels=True)", payload, pixels=True, runs=runs
    )
    rand256_stats = profile_parser(
        Rand256Parser(), "RAND256 (pixels=True)", payload, pixels=True, runs=runs
    )

    return rand25_stats, rand256_stats


def main():
    """Main profiling function."""
    payload_dir = "."
    runs = 5  # Number of runs for profiling

    if not os.path.exists(payload_dir):
        print(f"Payload directory {payload_dir} doesn't exist.")
        print("Run your vacuum first to generate payload files.")
        return

    # Find all payload files
    payload_files = [f for f in os.listdir(payload_dir) if f.endswith(".bin")]

    if not payload_files:
        print(f"No payload files found in {payload_dir}")
        print("Run your vacuum first to generate payload files.")
        return

    # Sort by timestamp (newest first)
    payload_files.sort(reverse=True)

    print(f"Found {len(payload_files)} payload files:")
    for i, f in enumerate(payload_files[:5]):  # Show first 5
        print(f"  {i + 1}. {f}")

    all_results = []

    # Test with the most recent payload (basic parsing)
    latest_payload = os.path.join(payload_dir, payload_files[0])
    print("\n🚀 Testing basic parsing (pixels=False)...")
    rand25_basic, rand256_basic = compare_parsers(latest_payload, runs)
    all_results.append(("Basic Parsing", rand25_basic, rand256_basic))

    # Test with pixel data (more intensive)
    print("\n🚀 Testing with pixel data (pixels=True)...")
    rand25_pixels, rand256_pixels = test_with_pixels(latest_payload, runs=3)
    all_results.append(("With Pixels", rand25_pixels, rand256_pixels))

    # Test with additional files if available
    if len(payload_files) > 1:
        print("\n🚀 Testing with second payload file...")
        second_payload = os.path.join(payload_dir, payload_files[1])
        rand25_second, rand256_second = compare_parsers(second_payload, runs)
        all_results.append(("Second File", rand25_second, rand256_second))

    # Summary report
    print(f"\n{'=' * 60}")
    print("FINAL PERFORMANCE SUMMARY")
    print(f"{'=' * 60}")

    for test_name, r25_stats, r256_stats in all_results:
        print(f"\n📋 {test_name}:")
        if r25_stats["success"] and r256_stats["success"]:
            r25_avg = r25_stats["avg_time"]
            r256_avg = r256_stats["avg_time"]

            # Use threshold for final summary too
            threshold = 0.0001
            time_diff = abs(r25_avg - r256_avg)

            if time_diff <= threshold:
                winner = "TIE (identical performance)"
            elif r25_avg < r256_avg:
                speedup = r256_avg / r25_avg
                winner = f"RAND25 ({speedup:.2f}x faster)"
            else:
                speedup = r25_avg / r256_avg
                winner = f"RAND256 ({speedup:.2f}x faster)"

            print(f"   RAND25:  {r25_avg:.4f}s ± {r25_stats['std_dev']:.4f}s")
            print(f"   RAND256: {r256_avg:.4f}s ± {r256_stats['std_dev']:.4f}s")
            print(f"   Winner:  {winner}")
        else:
            print("   ❌ Test failed - check individual results above")

    print(f"\n{'=' * 60}")
    print("PROFILING COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
