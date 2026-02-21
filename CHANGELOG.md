# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.5] - 2026-02-21

### 🎉 Stable Release

This is a stable release based on 0.2.5b3 with all beta features tested and verified.

### 📝 Changes from Beta
- **0.2.5b3**: Reverted experimental automatic content type selection feature
  - Feature was removed to maintain library simplicity
  - Content type selection remains manual via `set_content_type()` method

### ✨ Features Included
- All features from 0.2.4 release
- Stable multi-floor support
- Mop path customization for Hypfer vacuums
- Calibration rotation handling
- Memory leak fixes

---

## [0.2.5b3] - 2026-02-21

### 🔄 Changes
- **Reverted**: Automatic content type selection based on camera mode
  - Feature was experimental and removed before stable release
  - Content type selection remains manual for library flexibility

---

## [0.2.5b2] - 2026-02-19

### ✨ New Features
- **Image Format Configuration**: Added support for configurable image output formats
  - Added `ALLOWED_IMAGE_FORMAT` constant to define supported formats (PIL, PNG, JPEG)
  - Made `image_format` private (`_image_format`) with public `set_content_type()` method
  - Added `get_content_type()` method for accessing current format
  - Exported `ALLOWED_IMAGE_FORMAT` and `pil_to_pil_bytes` in `__init__.py`

### 🔧 Improvements
- **Binary Conversion**: Enhanced `_convert_to_binary()` with match-case for all formats
  - `image/jpeg` → JPEG bytes conversion
  - `image/png` → PNG bytes conversion
  - `image/pil` → PIL bytes conversion
  - Added `pil_to_pil_bytes()` utility function
- **Code Quality**: Used proper encapsulation with getter methods instead of direct attribute access

---

## [0.2.5b1] - 2026-02-18

### ✨ New Features
- **JPEG Format Support**: Added JPEG output format for image generation
  - New `pil_to_jpeg_bytes()` function with RGB conversion and quality control
  - Configurable via `shared.image_format = "image/jpeg"`
  - Default quality: 85, optimized for web delivery

### 🐛 Bug Fixes
- **API Consistency**: Fixed `data["image"]` structure to always be populated
  - Previously missing when `bytes_format=False`
  - Now always includes `binary`, `size`, and `streaming` keys
  - Ensures consistent API for mqtt_vacuum_camera integration

### 🔧 Improvements
- **Direct PIL Storage**: When `bytes_format=False`, stores PIL Image object directly
  - Enables efficient PIL→JPEG conversion in MJPEG streaming
  - Reduces unnecessary PNG conversion overhead
- **Public API**: Exported `pil_to_jpeg_bytes` and `pil_to_png_bytes` utility functions

### 📝 Testing
- Added `test_bytes_format_consistency.py` to verify API consistency

---

## [0.2.5b0] - 2026-02-17

### ✨ New Features
- **Customizable Obstacle Image Links**: Added configuration for custom obstacle image endpoints
  - New constants: `CONF_OBSTACLE_LINK_IP`, `CONF_OBSTACLE_LINK_PORT`, `CONF_OBSTACLE_LINK_PROTOCOL`
  - Allows using different IP/port/protocol than vacuum's main connection
  - Falls back to `vacuum_ips` if custom parameters not provided
  - Exported in `__init__.py` for mqtt_vacuum_camera integration

### 🔧 Improvements
- **Obstacle Link Composition**: Updated `_compose_obstacle_links()` to accept custom parameters
  - Supports HTTPS protocol for secure obstacle image retrieval
  - Configurable port for non-standard setups
  - Separate IP address for obstacle image server

---

## [0.2.4] - 2026-02-08

### 🐛 Critical Bug Fixes

#### Calibration Rotation Bug (Fixed in 0.2.4b3)
- **Fixed**: Calibration points now correctly update when map rotation changes
- **Issue**: Calibration points were showing identical coordinates across all rotations (270°, 0°, 90°)
- **Root Cause #1**: `get_vacuum_points()` was reordering points instead of using already-rotated crop_area values
- **Root Cause #2**: Calibration data was only calculated once and never recalculated when rotation changed
- **Impact**: Home Assistant coordinate mapping features (click-to-go, zone selection) now work correctly at all rotation angles

#### Rotation Change Handling (Fixed in 0.2.4b3)
- **Fixed**: Prevents "Invalid crop region" errors when user changes rotation with saved floor data
- **Issue**: Old trims from one rotation (e.g., 270°) were incompatible with different rotation (e.g., 0°)
- **Solution**: Added rotation tracking to `FloorData` - detects rotation changes and resets trims for recalculation
- **Implementation**:
  - Added `rotation: int` field to `FloorData` class
  - `update_trims()` now saves current rotation with trims
  - On reload, compares saved rotation with current rotation
  - If different, resets trims to defaults and lets auto-crop recalculate
- **Impact**: Smooth handling of rotation changes without manual intervention

#### Segment Alignment Bug (Fixed in 0.2.4b1)
- **Fixed**: Critical alignment bug in `_extract_segment_metadata()` that could cause data corruption
- **Issue**: Active list could become misaligned with segment entries on conversion failures
- **Solution**: Ensure active_list always appends a value (0 on failure) to maintain alignment

### ✨ New Features

#### Multi-Floor Support (0.2.2)
- Added `FloorData` class with methods for managing multiple floors:
  - `add_floor()` - Add a new floor with trims data
  - `update_floor()` - Update existing floor data
  - `remove_floor()` - Remove a floor from configuration
  - `update_trims()` - Update trims for current floor
  - `clear()` - Reset all floor data
- Floor-specific trim data now persisted in `floors_trims` dictionary
- Enhanced `CameraShared` class with floor management methods

#### Mop Path Customization (0.2.2)
- **Hypfer vacuums only**: Configurable mop path visualization
- New configuration options:
  - `mop_path_width` - Control path width when mopping (default: `robot_size - 2`)
  - `color_mop_move` - Customize mop path color (RGB, default: `[238, 247, 255]`)
  - `alpha_mop_move` - Control transparency (0-255, default: `100.0`)
- Mop path automatically uses configured settings when `mop_mode` is enabled
- Visual distinction between normal cleaning and mopping operations

#### Dock State Display (0.2.4b1)
- Added `ATTR_DOCK_STATE` constant and attribute
- Dock state now displayed in status text (e.g., "docked mop cleaning")
- Only shown when dock is performing operations (not idle)
- Exported in `__init__.py` for external use

### 🔧 Improvements

#### Compatibility
- **Python 3.12+ Support** (0.2.2): Changed requirement from Python 3.13 to 3.12 for Home Assistant compatibility
- **mvcrender 0.1.0**: Updated dependency with calibration rotation fixes ([Release v0.1.0](https://github.com/sca075/mvcrender/releases/tag/v0.1.0))

#### Code Quality
- **Pylint Score**: Achieved 10.00/10 rating
- **Refactoring**: Reduced code complexity in multiple modules:
  - `drawable_elements.py`: Reduced branches from 14 to 10
  - `map_data.py`: Reduced branches from 16 to 6, nested blocks from 7 to 3
  - Extracted helper methods for better maintainability
- **Exception Handling**: Replaced broad `Exception` catches with specific exceptions
- **Type Safety**: Improved type conversions for device_info values

#### Performance
- Image generation: ~450ms average
- Optimized path rendering
- Reduced memory overhead

### 🔄 Changed

#### Data Type Handling (0.2.3)
- All trim values now converted to integers in `TrimsData` and `TrimCropData`
- Numeric device_info values (offsets, sizes) converted to integers
- Prevents "float object cannot be interpreted as an integer" errors

#### Robot Size Initialization (0.2.2)
- Changed default from `None` to `25` to prevent TypeError in arithmetic operations
- Safer initialization before `update_shared_data()` is called

### 🗑️ Removed
- Cleaned up commented-out code blocks
- Removed duplicate configuration keys in test files

### 📝 Documentation
- Added integration guide for MQTT Vacuum Camera
- Updated README with recent features and version info
- Comprehensive changelog with clear categorization

---

## Configuration Example

For MQTT Vacuum Camera integration, add these options to your device configuration:

```yaml
# Mop path configuration (Hypfer vacuums only)
mop_path_width: 16              # Width of path when mopping (default: robot_size - 2)
color_mop_move: [238, 247, 255] # RGB color for mop path (default: light blue)
alpha_mop_move: 100.0           # Transparency (0-255, default: 100.0)
```

---

## [0.2.0 - 0.2.1] - Previous Releases

See [GitHub Releases](https://github.com/sca075/Python-package-valetudo-map-parser/releases) for version history.

