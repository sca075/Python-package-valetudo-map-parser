"""Memory profiling test for mvcrender C extensions."""
import tracemalloc
import gc
import numpy as np
from mvcrender.autocrop import AutoCrop
from mvcrender.blend import blend_mask_inplace, sample_and_blend_color, get_blended_color
from mvcrender.draw import line_u8, circle_u8, polygon_u8, polyline_u8


class DummyShared:
    def __init__(self):
        self.trims = type("T", (),
                          {"to_dict": lambda self: {"trim_up": 0, "trim_down": 0, "trim_left": 0, "trim_right": 0}})()
        self.offset_top = 0;
        self.offset_down = 0;
        self.offset_left = 0;
        self.offset_right = 0
        self.vacuum_state = "cleaning";
        self.image_auto_zoom = True
        self.image_ref_width = 0;
        self.image_ref_height = 0


class DummyBaseHandler:
    def __init__(self):
        self.crop_img_size = [0, 0]
        self.crop_area = None
        self.shared = None
        self.file_name = "memory_test"
        self.robot_position = (200, 150, 0)
        self.robot_pos = {"in_room": None}


class DummyHandler(DummyBaseHandler, AutoCrop):
    def __init__(self, shared=None):
        DummyBaseHandler.__init__(self)
        self.shared = shared
        AutoCrop.__init__(self, self)
        self.max_frames = 0
        self.room_propriety = None
        self.rooms_pos = []
        self.img_size = (0, 0)


print("=" * 70)
print("Memory Profiling Test - mvcrender C Extensions")
print("=" * 70)

# Start memory tracking
tracemalloc.start()

# Test parameters
H, W = 5700, 5700  # Large image as in production
ITERATIONS = 100

print(f"\nTest configuration:")
print(f"  Image size: {H}x{W} RGBA")
print(f"  Iterations: {ITERATIONS}")
print(f"  Memory per image: {H * W * 4 / 1024 / 1024:.2f} MB")

# Initialize handler
handler = DummyHandler(DummyShared())

# Baseline memory
gc.collect()
baseline_current, baseline_peak = tracemalloc.get_traced_memory()
print(f"\nBaseline memory:")
print(f"  Current: {baseline_current / 1024 / 1024:.2f} MB")
print(f"  Peak: {baseline_peak / 1024 / 1024:.2f} MB")

# Test 1: AutoCrop with rotation (most complex operation)
print(f"\n{'=' * 70}")
print("Test 1: AutoCrop with rotation ({ITERATIONS} iterations)")
print(f"{'=' * 70}")

for i in range(ITERATIONS):
    # Create fresh image each iteration
    img = np.zeros((H, W, 4), dtype=np.uint8)
    img[..., 3] = 255
    img[:, :, :3] = (93, 109, 126)
    img[500:2500, 800:3200, :3] = (120, 200, 255)

    # Process
    result = handler.auto_trim_and_zoom_image(
        img, (93, 109, 126, 255),
        margin_size=10,
        rotate=90,
        zoom=False,
        rand256=True,
    )

    # Explicitly delete to help GC
    del result
    del img

    # Check memory every 10 iterations
    if (i + 1) % 10 == 0:
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        print(f"  Iteration {i + 1:3d}: Current={current / 1024 / 1024:6.2f} MB, Peak={peak / 1024 / 1024:6.2f} MB")

gc.collect()
test1_current, test1_peak = tracemalloc.get_traced_memory()
print(f"\nTest 1 final memory:")
print(
    f"  Current: {test1_current / 1024 / 1024:.2f} MB (delta: {(test1_current - baseline_current) / 1024 / 1024:+.2f} MB)")
print(f"  Peak: {test1_peak / 1024 / 1024:.2f} MB")

# Test 2: Blending operations - blend_mask_inplace
print(f"\n{'=' * 70}")
print(f"Test 2: blend_mask_inplace ({ITERATIONS} iterations)")
print(f"{'=' * 70}")

for i in range(ITERATIONS):
    img = np.zeros((H, W, 4), dtype=np.uint8)
    img[..., 3] = 255
    img[:, :, :3] = (93, 109, 126)

    # Create mask
    mask = np.zeros((H, W), dtype=bool)
    mask[1000:2000, 1000:2000] = True

    # Blend
    blend_mask_inplace(img, mask, (255, 0, 0, 128))

    del img
    del mask

    if (i + 1) % 10 == 0:
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        print(f"  Iteration {i + 1:3d}: Current={current / 1024 / 1024:6.2f} MB, Peak={peak / 1024 / 1024:6.2f} MB")

gc.collect()
test2_current, test2_peak = tracemalloc.get_traced_memory()
print(f"\nTest 2 final memory:")
print(
    f"  Current: {test2_current / 1024 / 1024:.2f} MB (delta: {(test2_current - test1_current) / 1024 / 1024:+.2f} MB)")
print(f"  Peak: {test2_peak / 1024 / 1024:.2f} MB")

# Test 2b: sample_and_blend_color
print(f"\n{'=' * 70}")
print(f"Test 2b: sample_and_blend_color ({ITERATIONS} iterations)")
print(f"{'=' * 70}")

for i in range(ITERATIONS):
    img = np.zeros((H, W, 4), dtype=np.uint8)
    img[..., 3] = 255
    img[:, :, :3] = (93, 109, 126)

    # Sample and blend at many points
    color = (255, 128, 0, 128)
    for y in range(1000, 2000, 10):
        for x in range(1000, 2000, 10):
            r, g, b, a = sample_and_blend_color(img, x, y, color)
            img[y, x] = [r, g, b, a]

    del img

    if (i + 1) % 10 == 0:
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        print(f"  Iteration {i + 1:3d}: Current={current / 1024 / 1024:6.2f} MB, Peak={peak / 1024 / 1024:6.2f} MB")

gc.collect()
test2b_current, test2b_peak = tracemalloc.get_traced_memory()
print(f"\nTest 2b final memory:")
print(
    f"  Current: {test2b_current / 1024 / 1024:.2f} MB (delta: {(test2b_current - test2_current) / 1024 / 1024:+.2f} MB)")
print(f"  Peak: {test2b_peak / 1024 / 1024:.2f} MB")

# Test 2c: get_blended_color
print(f"\n{'=' * 70}")
print(f"Test 2c: get_blended_color ({ITERATIONS} iterations)")
print(f"{'=' * 70}")

for i in range(ITERATIONS):
    img = np.zeros((H, W, 4), dtype=np.uint8)
    img[..., 3] = 255
    img[:, :, :3] = (93, 109, 126)

    # Get blended color for line segments
    color = (255, 0, 128, 128)
    for j in range(100):
        x0, y0 = 1000 + j * 10, 1000
        x1, y1 = 2000, 1000 + j * 10
        r, g, b, a = get_blended_color(x0, y0, x1, y1, img, color)
        # Use the color (simulate drawing)
        if 0 <= y0 < H and 0 <= x0 < W:
            img[y0, x0] = [r, g, b, a]

    del img

    if (i + 1) % 10 == 0:
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        print(f"  Iteration {i + 1:3d}: Current={current / 1024 / 1024:6.2f} MB, Peak={peak / 1024 / 1024:6.2f} MB")

gc.collect()
test2c_current, test2c_peak = tracemalloc.get_traced_memory()
print(f"\nTest 2c final memory:")
print(
    f"  Current: {test2c_current / 1024 / 1024:.2f} MB (delta: {(test2c_current - test2b_current) / 1024 / 1024:+.2f} MB)")
print(f"  Peak: {test2c_peak / 1024 / 1024:.2f} MB")

# Test 3: Drawing operations
print(f"\n{'=' * 70}")
print(f"Test 3: Drawing operations ({ITERATIONS} iterations)")
print(f"{'=' * 70}")

for i in range(ITERATIONS):
    img = np.zeros((H, W, 4), dtype=np.uint8)
    img[..., 3] = 255

    # Draw various shapes
    line_u8(img, 0, 0, H - 1, W - 1, (255, 0, 0, 255), 5)
    circle_u8(img, H // 2, W // 2, 500, (0, 255, 0, 255), -1)

    xs = np.array([1000, 2000, 3000, 2000], dtype=np.int32)
    ys = np.array([1000, 1000, 2000, 2000], dtype=np.int32)
    polygon_u8(img, xs, ys, (0, 0, 255, 255), 3, (255, 255, 0, 128))

    # Polyline
    xs2 = np.array([500, 1000, 1500, 2000, 2500], dtype=np.int32)
    ys2 = np.array([500, 1000, 500, 1000, 500], dtype=np.int32)
    polyline_u8(img, xs2, ys2, (255, 0, 255, 255), 3)

    del img
    del xs
    del ys
    del xs2
    del ys2

    if (i + 1) % 10 == 0:
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        print(f"  Iteration {i + 1:3d}: Current={current / 1024 / 1024:6.2f} MB, Peak={peak / 1024 / 1024:6.2f} MB")

gc.collect()
test3_current, test3_peak = tracemalloc.get_traced_memory()
print(f"\nTest 3 final memory:")
print(
    f"  Current: {test3_current / 1024 / 1024:.2f} MB (delta: {(test3_current - test2c_current) / 1024 / 1024:+.2f} MB)")
print(f"  Peak: {test3_peak / 1024 / 1024:.2f} MB")

# Final summary
print(f"\n{'=' * 70}")
print("MEMORY LEAK ANALYSIS")
print(f"{'=' * 70}")

memory_growth = test3_current - baseline_current
memory_per_iteration = memory_growth / (ITERATIONS * 5)  # 5 test sections now

print(f"\nTotal memory growth: {memory_growth / 1024 / 1024:.2f} MB")
print(f"Memory per iteration: {memory_per_iteration / 1024:.2f} KB")

if memory_per_iteration < 10:  # Less than 10KB per iteration
    print("\n✅ PASS: No significant memory leaks detected")
    print("   Memory growth is within acceptable bounds for Python overhead")
elif memory_per_iteration < 100:  # Less than 100KB per iteration
    print("\n⚠️  WARNING: Small memory growth detected")
    print("   May be Python overhead, but worth monitoring")
else:
    print("\n❌ FAIL: Significant memory leak detected!")
    print("   Memory is growing beyond acceptable bounds")

# Stop tracking
tracemalloc.stop()

print(f"\n{'=' * 70}")
print("Test complete!")
print(f"{'=' * 70}")

