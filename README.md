# Python-package-valetudo-map-parser

---
### What is it:
❗This is an _unofficial_ project and is not created, maintained, or in any sense linked to [valetudo.cloud](https://valetudo.cloud)

A Python library that converts Valetudo vacuum JSON map data into PIL (Python Imaging Library) images. This package is primarily developed for and used in the [MQTT Vacuum Camera](https://github.com/sca075/mqtt_vacuum_camera) project.

---

### Features:
- Processes map data from Valetudo-compatible robot vacuums
- Supports both Hypfer and Rand256 vacuum data formats
- Renders comprehensive map visualizations including:
  - Walls and obstacles
  - Robot position and cleaning path
  - Room segments and boundaries
  - Cleaning zones
  - Virtual restrictions
  - LiDAR data
- Provides auto-cropping and dynamic zooming
- Supports image rotation and aspect ratio management
- Enables custom color schemes
- Handles multilingual labels
- Implements thread-safe data sharing

### Installation:
```bash
pip install valetudo_map_parser
```

### Requirements:
- Python 3.12 or higher
- Dependencies:
  - Pillow (PIL) for image processing
  - NumPy for array operations

### Usage:
The library is configured using a dictionary format. See our [sample code](https://github.com/sca075/Python-package-valetudo-map-parser/blob/main/tests/test.py) for implementation examples.

Key functionalities:
- Decodes raw data from Rand256 format
- Processes JSON data from compatible vacuums
- Returns Pillow PNG images
- Provides calibration and room property extraction
- Supports asynchronous operations

### Development Status:
Current version: 0.1.9.b41
- Full functionality available in versions >= 0.1.9
- Actively maintained and enhanced
- Uses Poetry for dependency management
- Implements comprehensive testing
- Enforces code quality through ruff, isort, and pylint

### Contributing:
Contributions are welcome! You can help by:
- Submitting code improvements
- Enhancing documentation
- Reporting issues
- Suggesting new features

### Disclaimer:
This project is provided "as is" without warranty of any kind. Users assume all risks associated with its use.

### License:
Apache-2.0

---
For more information about Valetudo, visit [valetudo.cloud](https://valetudo.cloud)
Integration with Home Assistant: [MQTT Vacuum Camera](https://github.com/sca075/mqtt_vacuum_camera)
