import asyncio
import time

import numpy as np
from scipy import ndimage

from SCR.valetudo_map_parser.config.auto_crop import AutoCrop
from SCR.valetudo_map_parser.config.utils import BaseHandler


class DummyHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.file_name = "benchmark"
        self.shared = type(
            "obj",
            (object,),
            {
                "trims": type(
                    "obj",
                    (object,),
                    {
                        "to_dict": lambda: {
                            "trim_up": 0,
                            "trim_left": 0,
                            "trim_right": 0,
                            "trim_down": 0,
                        }
                    },
                ),
                "offset_top": 0,
                "offset_down": 0,
                "offset_left": 0,
                "offset_right": 0,
            },
        )


# Original implementation for comparison
async def original_image_margins(
    image_array: np.ndarray, detect_colour: tuple
) -> tuple[int, int, int, int]:
    """Original implementation of the image margins function"""
    nonzero_coords = np.column_stack(np.where(image_array != list(detect_colour)))
    # Calculate the trim box based on the first and last occurrences
    min_y, min_x, _ = np.min(nonzero_coords, axis=0)
    max_y, max_x, _ = np.max(nonzero_coords, axis=0)
    del nonzero_coords
    return min_y, min_x, max_x, max_y


# Optimized implementation (similar to what we added to auto_crop.py)
async def optimized_image_margins(
    image_array: np.ndarray, detect_colour: tuple
) -> tuple[int, int, int, int]:
    """Optimized implementation using scipy.ndimage"""
    # Create a binary mask where True = non-background pixels
    mask = ~np.all(image_array == list(detect_colour), axis=2)

    # Use scipy.ndimage.find_objects to efficiently find the bounding box
    labeled_mask = mask.astype(np.int8)  # Convert to int8 (smallest integer type)
    objects = ndimage.find_objects(labeled_mask)

    if not objects:  # No objects found
        return 0, 0, image_array.shape[1], image_array.shape[0]

    # Extract the bounding box coordinates from the slice objects
    y_slice, x_slice = objects[0]
    min_y, max_y = y_slice.start, y_slice.stop - 1
    min_x, max_x = x_slice.start, x_slice.stop - 1

    return min_y, min_x, max_x, max_y


async def benchmark():
    # Create test images of different sizes to simulate real-world scenarios
    image_sizes = [(2000, 2000, 4), (4000, 4000, 4), (8000, 8000, 4)]
    background_color = (0, 125, 255, 255)  # Background color
    iterations = 5

    for size in image_sizes:
        print(f"\n=== Testing with image size {size[0]}x{size[1]} ===\n")

        # Create image with background color
        image = np.full(size, background_color, dtype=np.uint8)

        # Add a non-background rectangle in the middle (40% of image size)
        rect_size_x = int(size[1] * 0.4)
        rect_size_y = int(size[0] * 0.4)
        start_x = (size[1] - rect_size_x) // 2
        start_y = (size[0] - rect_size_y) // 2
        image[start_y : start_y + rect_size_y, start_x : start_x + rect_size_x] = (
            255,
            0,
            0,
            255,
        )

        # Create AutoCrop instance
        handler = DummyHandler()
        auto_crop = AutoCrop(handler)

        # Benchmark the original implementation
        print(
            f"Running benchmark for ORIGINAL implementation ({iterations} iterations)..."
        )
        original_total_time = 0

        for i in range(iterations):
            start_time = time.time()
            min_y, min_x, max_x, max_y = await original_image_margins(
                image, background_color
            )
            end_time = time.time()

            elapsed = end_time - start_time
            original_total_time += elapsed

            print(f"Iteration {i + 1}: {elapsed:.6f} seconds")

        original_avg_time = original_total_time / iterations
        print(f"Original implementation average: {original_avg_time:.6f} seconds")

        # Benchmark the optimized implementation
        print(
            f"\nRunning benchmark for OPTIMIZED implementation ({iterations} iterations)..."
        )
        optimized_total_time = 0

        for i in range(iterations):
            start_time = time.time()
            min_y, min_x, max_x, max_y = await optimized_image_margins(
                image, background_color
            )
            end_time = time.time()

            elapsed = end_time - start_time
            optimized_total_time += elapsed

            print(f"Iteration {i + 1}: {elapsed:.6f} seconds")

        optimized_avg_time = optimized_total_time / iterations
        print(f"Optimized implementation average: {optimized_avg_time:.6f} seconds")

        # Calculate and display improvement
        if original_avg_time > 0:
            improvement = (
                (original_avg_time - optimized_avg_time) / original_avg_time * 100
            )
            print(f"\nImprovement: {improvement:.2f}% faster")
            print(
                f"Original: {original_avg_time:.6f}s vs Optimized: {optimized_avg_time:.6f}s"
            )


if __name__ == "__main__":
    asyncio.run(benchmark())
