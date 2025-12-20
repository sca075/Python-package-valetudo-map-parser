"""
Test script to convert Rand256 pixel format to Hypfer compressed format.

This demonstrates how to compress the huge Rand256 pixel lists into
the same compact format used by Hypfer vacuums.

Rand256 format: [30358, 30359, 30360, ...] - individual pixel indices
Hypfer format: [x, y, length, x, y, length, ...] - compressed runs
"""

import json


def compress_rand_to_hypfer(
    pixel_indices: list,
    image_width: int,
    image_height: int,
    image_top: int = 0,
    image_left: int = 0,
) -> list:
    """
    Convert Rand256 pixel indices to Hypfer compressed format.
    
    Args:
        pixel_indices: List of pixel indices [30358, 30359, 30360, ...]
        image_width: Width of the image
        image_height: Height of the image
        image_top: Top offset
        image_left: Left offset
    
    Returns:
        Flat list in Hypfer format: [x, y, length, x, y, length, ...]
    """
    if not pixel_indices:
        return []
    
    compressed = []
    
    # Convert indices to (x, y) coordinates and group consecutive runs
    prev_x = prev_y = None
    run_start_x = run_y = None
    run_length = 0
    
    for idx in pixel_indices:
        # Convert pixel index to x, y coordinates
        # Same formula as in from_rrm_to_compressed_pixels
        x = (idx % image_width) + image_left
        y = ((image_height - 1) - (idx // image_width)) + image_top
        
        if run_start_x is None:
            # Start first run
            run_start_x, run_y, run_length = x, y, 1
        elif y == run_y and x == prev_x + 1:
            # Continue current run (same row, consecutive x)
            run_length += 1
        else:
            # End current run, start new one
            compressed.extend([run_start_x, run_y, run_length])
            run_start_x, run_y, run_length = x, y, 1
        
        prev_x, prev_y = x, y
    
    # Add final run
    if run_start_x is not None:
        compressed.extend([run_start_x, run_y, run_length])
    
    return compressed


def main():
    """Test the compression on segment 20 from rand.json."""
    
    # Load rand.json
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rand_json_path = os.path.join(script_dir, "rand.json")

    with open(rand_json_path, "r") as f:
        rand_data = json.load(f)
    
    # Get image dimensions
    image_data = rand_data["image"]
    dimensions = image_data["dimensions"]
    position = image_data["position"]
    
    image_width = dimensions["width"]
    image_height = dimensions["height"]
    image_top = position["top"]
    image_left = position["left"]
    
    print(f"Image dimensions: {image_width}x{image_height}")
    print(f"Image position: top={image_top}, left={image_left}")
    print()
    
    # Get segment 20 data
    segments = image_data["segments"]
    segment_id = 20
    pixel_indices = segments[f"pixels_seg_{segment_id}"]
    
    print(f"Segment {segment_id}:")
    print(f"  Original format (Rand256): {len(pixel_indices)} individual pixel indices")
    print(f"  First 10 indices: {pixel_indices[:10]}")
    print(f"  Memory size (approx): {len(pixel_indices) * 8} bytes (assuming 8 bytes per int)")
    print()
    
    # Compress to Hypfer format
    compressed = compress_rand_to_hypfer(
        pixel_indices,
        image_width,
        image_height,
        image_top,
        image_left
    )
    
    print(f"  Compressed format (Hypfer): {len(compressed)} values")
    print(f"  Number of runs: {len(compressed) // 3}")
    print(f"  First 3 runs (x, y, length): {compressed[:9]}")
    print(f"  Memory size (approx): {len(compressed) * 8} bytes")
    print()
    
    # Calculate compression ratio
    original_size = len(pixel_indices)
    compressed_size = len(compressed)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    print(f"Compression ratio: {ratio:.2f}x")
    print(f"Memory reduction: {(1 - compressed_size/original_size) * 100:.1f}%")
    print()
    
    # Verify the compression is correct by reconstructing pixels
    print("Verifying compression...")
    reconstructed = []
    for i in range(0, len(compressed), 3):
        x, y, length = compressed[i], compressed[i+1], compressed[i+2]
        for j in range(length):
            # Convert back to pixel index
            pixel_x = x + j - image_left
            pixel_y = (image_height - 1) - (y - image_top)
            pixel_idx = pixel_y * image_width + pixel_x
            reconstructed.append(pixel_idx)
    
    # Check if reconstruction matches original
    if reconstructed == pixel_indices:
        print("✓ Compression verified! Reconstructed pixels match original.")
    else:
        print("✗ Compression error! Reconstructed pixels don't match.")
        print(f"  Original: {len(pixel_indices)} pixels")
        print(f"  Reconstructed: {len(reconstructed)} pixels")
        # Show first difference
        for i, (orig, recon) in enumerate(zip(pixel_indices, reconstructed)):
            if orig != recon:
                print(f"  First difference at index {i}: {orig} != {recon}")
                break
    
    print()
    print("=" * 60)
    print("Summary:")
    print(f"  This compression would reduce Rand256 memory usage from")
    print(f"  ~126MB/frame to ~{126 * compressed_size / original_size:.1f}MB/frame")
    print(f"  Making it comparable to Hypfer's ~12MB/frame")
    print()

    # Show the data in dictionary format
    print("=" * 60)
    print("CONVERTED DATA IN DICTIONARY FORMAT:")
    print("=" * 60)
    print()

    # Create a dictionary similar to Hypfer format
    converted_segment = {
        "segment_id": segment_id,
        "format": "hypfer_compressed",
        "compressedPixels": compressed,
        "pixel_count": len(pixel_indices),
        "compressed_count": len(compressed),
        "run_count": len(compressed) // 3,
    }

    print("Segment data:")
    print(json.dumps(converted_segment, indent=2))
    print()

    # Show first few runs in readable format
    print("First 5 runs (human-readable):")
    for i in range(0, min(15, len(compressed)), 3):
        x, y, length = compressed[i], compressed[i+1], compressed[i+2]
        print(f"  Run {i//3 + 1}: x={x}, y={y}, length={length} pixels")
    print()

    # Show what the full converted JSON structure would look like
    print("=" * 60)
    print("FULL CONVERTED STRUCTURE (like Hypfer):")
    print("=" * 60)
    print()

    converted_full = {
        "image": {
            "dimensions": {
                "width": image_width,
                "height": image_height
            },
            "position": {
                "top": image_top,
                "left": image_left
            },
            "segments": {
                "count": 1,
                "id": [segment_id],
                f"compressedPixels_{segment_id}": compressed
            }
        }
    }

    print(json.dumps(converted_full, indent=2))


if __name__ == "__main__":
    main()

