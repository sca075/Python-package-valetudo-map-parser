# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Mop Path Visualization** (Hypfer vacuums only):
  - Added `mop_path_width` configuration option to control the width of the path when vacuum is in mop mode
  - Added `color_mop_move` configuration option to customize the mop path color (RGB)
  - Added `alpha_mop_move` configuration option to control the mop path transparency (default: 100.0)
  - Mop path automatically uses configured width and color when `mop_mode` is enabled
  - Default mop path width is calculated as `robot_size - 2` if not explicitly configured
  - Default mop path alpha is 100.0 (semi-transparent) for better visibility

### Changed
- Enhanced path rendering to support different visual styles for normal cleaning vs. mopping operations
- Updated color management system to include mop-specific color configuration

### Technical Details
- Added `COLOR_MOP_MOVE` and `ALPHA_MOP_MOVE` constants to `const.py`
- Extended `COLORS` list to include `"mop_move"` color
- Updated `colors.py` to initialize mop path colors with proper defaults
- Modified `hypfer_draw.py` to use mop-specific width and color when in mop mode
- Added `mop_path_width` to `CameraShared` class for configuration storage

### Notes
- This feature is only available for Hypfer-compatible vacuums
- Rand256 vacuums continue to use standard path rendering without mop mode support

## [0.2.2] - 2024-XX-XX

### Previous releases
See [GitHub Releases](https://github.com/sca075/Python-package-valetudo-map-parser/releases) for version history prior to this changelog.

---

## Configuration Example

For MQTT Vacuum Camera integration, add these new options to your device configuration:

```yaml
# Mop path configuration (Hypfer vacuums only)
mop_path_width: 16              # Width of path when mopping (default: robot_size - 2)
color_mop_move: [238, 247, 255] # RGB color for mop path (default: same as path color)
alpha_mop_move: 100.0           # Transparency for mop path (0-255, default: 100.0)
```

These options work in conjunction with the existing `mop_mode` setting to provide visual distinction between normal cleaning and mopping operations.

