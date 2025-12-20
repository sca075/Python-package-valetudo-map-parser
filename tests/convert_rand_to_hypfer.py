"""
Complete conversion script: Rand256 JSON → Hypfer JSON format.

Converts all segments, paths, robot position, charger, etc.
"""

import json
import os


def compress_pixels(pixel_indices, image_width, image_height, image_top=0, image_left=0):
    """Convert Rand256 pixel indices to Hypfer compressed format."""
    if not pixel_indices:
        return []
    
    compressed = []
    prev_x = prev_y = None
    run_start_x = run_y = None
    run_length = 0
    
    for idx in pixel_indices:
        x = (idx % image_width) + image_left
        y = ((image_height - 1) - (idx // image_width)) + image_top
        
        if run_start_x is None:
            run_start_x, run_y, run_length = x, y, 1
        elif y == run_y and x == prev_x + 1:
            run_length += 1
        else:
            compressed.extend([run_start_x, run_y, run_length])
            run_start_x, run_y, run_length = x, y, 1
        
        prev_x, prev_y = x, y
    
    if run_start_x is not None:
        compressed.extend([run_start_x, run_y, run_length])
    
    return compressed


def calculate_dimensions(compressed_pixels):
    """Calculate min/max/mid/avg dimensions from compressed pixels."""
    if not compressed_pixels:
        return None
    
    x_coords = []
    y_coords = []
    pixel_count = 0
    
    for i in range(0, len(compressed_pixels), 3):
        x, y, length = compressed_pixels[i], compressed_pixels[i+1], compressed_pixels[i+2]
        for j in range(length):
            x_coords.append(x + j)
            y_coords.append(y)
            pixel_count += 1
    
    return {
        "x": {
            "min": min(x_coords),
            "max": max(x_coords),
            "mid": (min(x_coords) + max(x_coords)) // 2,
            "avg": sum(x_coords) // len(x_coords)
        },
        "y": {
            "min": min(y_coords),
            "max": max(y_coords),
            "mid": (min(y_coords) + max(y_coords)) // 2,
            "avg": sum(y_coords) // len(y_coords)
        },
        "pixelCount": pixel_count
    }


def convert_rand_to_hypfer(rand_json_path, output_path):
    """Convert complete Rand256 JSON to Hypfer format."""
    
    # Load Rand256 JSON
    with open(rand_json_path, 'r') as f:
        rand_data = json.load(f)
    
    # Extract image data
    image = rand_data["image"]
    dimensions = image["dimensions"]
    position = image["position"]
    segments_data = image["segments"]
    
    image_width = dimensions["width"]
    image_height = dimensions["height"]
    image_top = position["top"]
    image_left = position["left"]
    
    # Calculate total map size (Hypfer uses absolute coordinates)
    # Assuming pixelSize = 5 (standard for most vacuums)
    pixel_size = 5
    map_size_x = (image_width + image_left) * pixel_size
    map_size_y = (image_height + image_top) * pixel_size
    
    # Convert floor layer
    layers = []
    total_area = 0

    if "pixels" in image and "floor" in image["pixels"]:
        floor_pixels = image["pixels"]["floor"]
        compressed_floor = compress_pixels(
            floor_pixels,
            image_width,
            image_height,
            image_top,
            image_left
        )

        dims_floor = calculate_dimensions(compressed_floor)
        if dims_floor:
            total_area += dims_floor["pixelCount"] * (pixel_size ** 2)

        floor_layer = {
            "__class": "MapLayer",
            "metaData": {},
            "type": "floor",
            "pixels": [],
            "dimensions": dims_floor if dims_floor else {},
            "compressedPixels": compressed_floor
        }
        layers.append(floor_layer)

    # Convert wall layer
    if "pixels" in image and "walls" in image["pixels"]:
        wall_pixels = image["pixels"]["walls"]
        compressed_walls = compress_pixels(
            wall_pixels,
            image_width,
            image_height,
            image_top,
            image_left
        )

        dims_walls = calculate_dimensions(compressed_walls)

        wall_layer = {
            "__class": "MapLayer",
            "metaData": {},
            "type": "wall",
            "pixels": [],
            "dimensions": dims_walls if dims_walls else {},
            "compressedPixels": compressed_walls
        }
        layers.append(wall_layer)

    # Convert segments
    segment_ids = segments_data["id"]

    for seg_id in segment_ids:
        pixel_key = f"pixels_seg_{seg_id}"
        if pixel_key not in segments_data:
            continue
        
        pixel_indices = segments_data[pixel_key]
        
        # Compress pixels
        compressed = compress_pixels(
            pixel_indices,
            image_width,
            image_height,
            image_top,
            image_left
        )
        
        # Calculate dimensions
        dims = calculate_dimensions(compressed)
        if dims:
            total_area += dims["pixelCount"] * (pixel_size ** 2)
        
        # Create layer in Hypfer format
        layer = {
            "__class": "MapLayer",
            "metaData": {
                "segmentId": str(seg_id),
                "active": False,
                "source": "regular",
                "name": f"Room {seg_id}",
                "area": dims["pixelCount"] * (pixel_size ** 2) if dims else 0
            },
            "type": "segment",
            "pixels": [],
            "dimensions": dims if dims else {},
            "compressedPixels": compressed
        }
        
        layers.append(layer)
    
    # Convert path (divide by 10)
    path_points = []
    if "path" in rand_data and "points" in rand_data["path"]:
        for point in rand_data["path"]["points"]:
            path_points.extend([point[0] // 10, point[1] // 10])
    
    # Create path entity
    entities = []
    if path_points:
        entities.append({
            "__class": "PathMapEntity",
            "metaData": {},
            "type": "path",
            "points": path_points
        })
    
    # Convert robot position (divide by 10)
    if "robot" in rand_data and rand_data["robot"]:
        robot_pos = rand_data["robot"]
        entities.append({
            "__class": "PointMapEntity",
            "metaData": {
                "angle": rand_data.get("robot_angle", 0)
            },
            "type": "robot_position",
            "points": [robot_pos[0] // 10, robot_pos[1] // 10]
        })

    # Convert charger position (divide by 10)
    if "charger" in rand_data and rand_data["charger"]:
        charger_pos = rand_data["charger"]
        entities.append({
            "__class": "PointMapEntity",
            "metaData": {},
            "type": "charger_location",
            "points": [charger_pos[0] // 10, charger_pos[1] // 10]
        })
    
    # Convert virtual walls
    if "virtual_walls" in rand_data and rand_data["virtual_walls"]:
        for wall in rand_data["virtual_walls"]:
            entities.append({
                "__class": "LineMapEntity",
                "metaData": {},
                "type": "virtual_wall",
                "points": wall
            })
    
    # Convert forbidden zones
    if "forbidden_zones" in rand_data and rand_data["forbidden_zones"]:
        for zone in rand_data["forbidden_zones"]:
            entities.append({
                "__class": "PolygonMapEntity",
                "metaData": {},
                "type": "no_go_area",
                "points": zone
            })
    
    # Create Hypfer JSON structure
    hypfer_data = {
        "__class": "ValetudoMap",
        "metaData": {
            "version": 2,
            "nonce": "converted-from-rand256",
            "totalLayerArea": total_area
        },
        "size": {
            "x": map_size_x,
            "y": map_size_y
        },
        "pixelSize": pixel_size,
        "layers": layers,
        "entities": entities
    }
    
    # Save converted JSON
    with open(output_path, 'w') as f:
        json.dump(hypfer_data, f, indent=2)
    
    return hypfer_data


def main():
    """Convert rand.json to Hypfer format."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rand_json = os.path.join(script_dir, "rand.json")
    output_json = os.path.join(script_dir, "rand_converted.json")
    
    print("Converting Rand256 JSON to Hypfer format...")
    print(f"Input: {rand_json}")
    print(f"Output: {output_json}")
    print()
    
    result = convert_rand_to_hypfer(rand_json, output_json)
    
    print("Conversion complete!")
    print()
    print(f"Segments converted: {len(result['layers'])}")
    print(f"Entities created: {len(result['entities'])}")
    print(f"Total layer area: {result['metaData']['totalLayerArea']}")
    print(f"Map size: {result['size']['x']} x {result['size']['y']}")
    print()
    
    # Show compression stats
    with open(rand_json, 'r') as f:
        original = json.load(f)
    
    original_pixels = 0
    compressed_pixels = 0
    
    for seg_id in original["image"]["segments"]["id"]:
        pixel_key = f"pixels_seg_{seg_id}"
        if pixel_key in original["image"]["segments"]:
            original_pixels += len(original["image"]["segments"][pixel_key])
    
    for layer in result["layers"]:
        compressed_pixels += len(layer["compressedPixels"])
    
    print(f"Original pixel data: {original_pixels} values")
    print(f"Compressed pixel data: {compressed_pixels} values")
    print(f"Compression ratio: {original_pixels / compressed_pixels:.2f}x")
    print(f"Memory reduction: {(1 - compressed_pixels/original_pixels) * 100:.1f}%")
    print()
    print(f"✅ Converted JSON saved to: {output_json}")


if __name__ == "__main__":
    main()

