#!/usr/bin/env python3
"""
Example demonstrating the usage of async_get_pil_image function
for both Hypfer and Rand256 handlers.

This example shows how to:
1. Initialize handlers with shared data
2. Use the unified async_get_pil_image function
3. Access processed images from shared data
4. Check image update timestamps
"""

import asyncio
import datetime
import sys
from pathlib import Path


# Add the SCR directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "SCR"))

from valetudo_map_parser.config.shared import CameraSharedManager
from valetudo_map_parser.hypfer_handler import HypferMapImageHandler
from valetudo_map_parser.rand256_handler import ReImageHandler


async def example_hypfer_usage():
    """Example usage with Hypfer handler."""
    print("=== Hypfer Handler Example ===")

    # Initialize shared data manager
    device_info = {
        "auto_zoom": False,
        "margins": 100,
        "rotate_image": 0,
        "aspect_ratio": "None",
    }

    shared_manager = CameraSharedManager("test_hypfer", device_info)
    shared = shared_manager.get_instance()

    # Initialize Hypfer handler
    handler = HypferMapImageHandler(shared)

    # Example JSON data (you would get this from your vacuum)
    example_json = {"size": {"x": 1024, "y": 1024}, "entities": [], "layers": []}

    # Use the unified async_get_pil_image function
    print("Processing image with async_get_pil_image...")
    pil_image = await handler.async_get_image(example_json)

    if pil_image:
        print(f"✅ Image processed successfully: {pil_image.size}")
        print(f"📷 Image stored in shared.new_image: {shared.new_image is not None}")
        print(
            f"🕒 Last updated: {datetime.datetime.fromtimestamp(shared.image_last_updated)}"
        )

        # Process another image to see backup functionality
        print("\nProcessing second image to demonstrate backup...")
        pil_image2 = await handler.async_get_image(example_json)

        if pil_image2:
            print(f"✅ Second image processed: {pil_image2.size}")
            print(
                f"💾 Previous image backed up to shared.last_image: {shared.last_image is not None}"
            )
            print(
                f"🕒 Updated timestamp: {datetime.datetime.fromtimestamp(shared.image_last_updated)}"
            )
    else:
        print("❌ Failed to process image")


async def example_rand256_usage():
    """Example usage with Rand256 handler."""
    print("\n=== Rand256 Handler Example ===")

    # Initialize shared data manager
    device_info = {
        "auto_zoom": False,
        "margins": 100,
        "rotate_image": 0,
        "aspect_ratio": "None",
    }

    shared_manager = CameraSharedManager("test_rand256", device_info)
    shared = shared_manager.get_instance()

    # Initialize Rand256 handler
    handler = ReImageHandler(shared)

    # Example JSON data for Rand256 (you would get this from your vacuum)
    example_json = {
        "image": {
            "dimensions": {"x": 1024, "y": 1024},
            "pixels": {"floor": [], "wall": []},
        },
        "entities": [],
    }

    # Example destinations for Rand256
    destinations = ["room1", "room2", "kitchen"]

    # Use the unified async_get_pil_image function
    print("Processing image with async_get_pil_image...")
    pil_image = await handler.async_get_image(example_json, destinations)

    if pil_image:
        print(f"✅ Image processed successfully: {pil_image.size}")
        print(f"📷 Image stored in shared.new_image: {shared.new_image is not None}")
        print(
            f"🕒 Last updated: {datetime.datetime.fromtimestamp(shared.image_last_updated)}"
        )

        # Process another image to see backup functionality
        print("\nProcessing second image to demonstrate backup...")
        pil_image2 = await handler.async_get_image(example_json, destinations)

        if pil_image2:
            print(f"✅ Second image processed: {pil_image2.size}")
            print(
                f"💾 Previous image backed up to shared.last_image: {shared.last_image is not None}"
            )
            print(
                f"🕒 Updated timestamp: {datetime.datetime.fromtimestamp(shared.image_last_updated)}"
            )
    else:
        print("❌ Failed to process image")


async def demonstrate_shared_data_management():
    """Demonstrate shared data management across multiple handlers."""
    print("\n=== Shared Data Management Demo ===")

    # Create two different handlers with different shared instances
    device_info = {"auto_zoom": False, "margins": 100}

    # Hypfer handler
    hypfer_shared_manager = CameraSharedManager("vacuum_1", device_info)
    hypfer_shared = hypfer_shared_manager.get_instance()
    hypfer_handler = HypferMapImageHandler(hypfer_shared)

    # Rand256 handler
    rand256_shared_manager = CameraSharedManager("vacuum_2", device_info)
    rand256_shared = rand256_shared_manager.get_instance()
    rand256_handler = ReImageHandler(rand256_shared)

    print("Initial state:")
    print(f"Hypfer shared.new_image: {hypfer_shared.new_image}")
    print(f"Rand256 shared.new_image: {rand256_shared.new_image}")

    # Process images with both handlers
    hypfer_json = {"size": {"x": 512, "y": 512}, "entities": [], "layers": []}
    rand256_json = {
        "image": {
            "dimensions": {"x": 512, "y": 512},
            "pixels": {"floor": [], "wall": []},
        },
        "entities": [],
    }

    # Process concurrently
    results = await asyncio.gather(
        hypfer_handler.async_get_pil_image(hypfer_json),
        rand256_handler.async_get_pil_image(rand256_json, ["room1"]),
        return_exceptions=True,
    )

    print("\nAfter processing:")
    print(f"Hypfer result: {'✅ Success' if results[0] else '❌ Failed'}")
    print(f"Rand256 result: {'✅ Success' if results[1] else '❌ Failed'}")
    print(f"Hypfer shared.new_image: {hypfer_shared.new_image is not None}")
    print(f"Rand256 shared.new_image: {rand256_shared.new_image is not None}")

    if hypfer_shared.image_last_updated > 0:
        print(
            f"Hypfer last updated: {datetime.datetime.fromtimestamp(hypfer_shared.image_last_updated)}"
        )
    if rand256_shared.image_last_updated > 0:
        print(
            f"Rand256 last updated: {datetime.datetime.fromtimestamp(rand256_shared.image_last_updated)}"
        )


async def main():
    """Main example function."""
    print("🚀 async_get_pil_image Function Examples")
    print("=" * 50)

    try:
        await example_hypfer_usage()
        await example_rand256_usage()
        await demonstrate_shared_data_management()

        print("\n✅ All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
