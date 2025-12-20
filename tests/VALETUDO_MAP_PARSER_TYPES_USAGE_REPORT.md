# Valetudo Map Parser Types Usage Report
**Generated:** 2025-10-18  
**Purpose:** Comprehensive analysis of all valetudo_map_parser types, classes, and constants currently in use

---

## Executive Summary

This report documents all imports and usages of the `valetudo_map_parser` library throughout the MQTT Vacuum Camera integration codebase. The library is used across 8 main files with 13 distinct imports.

---

## 1. Import Summary by Category

### 1.1 Configuration & Shared Data
- **`CameraShared`** - Main shared configuration object
- **`CameraSharedManager`** - Manager for CameraShared instances
- **`ColorsManagement`** - Color configuration for maps

### 1.2 Type Definitions
- **`JsonType`** - Type alias for JSON data
- **`PilPNG`** - Type alias for PIL Image objects
- **`RoomStore`** - Storage and management of room data
- **`UserLanguageStore`** - Storage for user language preferences

### 1.3 Image Handlers
- **`HypferMapImageHandler`** - Handler for Hypfer/Valetudo firmware maps
- **`ReImageHandler`** - Handler for Rand256 firmware maps

### 1.4 Parsers
- **`RRMapParser`** - Parser for Rand256 binary map data

### 1.5 Utilities
- **`ResizeParams`** - Parameters for image resizing
- **`async_resize_image`** - Async function to resize images
- **`get_default_font_path`** - Function to get default font path

---

## 2. Detailed Usage by File

### 2.1 `__init__.py`
**Location:** `custom_components/mqtt_vacuum_camera/__init__.py`

**Imports:**
```python
from valetudo_map_parser import get_default_font_path
from valetudo_map_parser.config.shared import CameraShared, CameraSharedManager
```

**Usage:**
- **Line 23-24:** Import statements
- **Line 66:** Type hint `tuple[Optional[CameraShared], Optional[str]]`
- **Line 75:** Create instance: `CameraSharedManager(file_name, dict(device_info))`
- **Line 76:** Get shared instance: `shared_manager.get_instance()`
- **Line 77:** Set font path: `shared.vacuum_status_font = f"{get_default_font_path()}/FiraSans.ttf"`
- **Line 83:** Parameter type: `shared: CameraShared`

**Purpose:** Initialize shared configuration and font paths for the integration

---

### 2.2 `coordinator.py`
**Location:** `custom_components/mqtt_vacuum_camera/coordinator.py`

**Imports:**
```python
from valetudo_map_parser.config.shared import CameraShared, CameraSharedManager
```

**Usage:**
- **Line 16:** Import statement
- **Line 33:** Parameter type: `shared: Optional[CameraShared]`
- **Line 50:** Attribute type: `self.shared_manager: Optional[CameraSharedManager]`
- **Line 52-54:** Access shared properties: `self.shared`, `self.shared.is_rand`, `self.shared.file_name`
- **Line 96:** Access shared property: `self.shared.current_room`

**Purpose:** Coordinator uses CameraShared to maintain state across the integration

---

### 2.3 `camera.py`
**Location:** `custom_components/mqtt_vacuum_camera/camera.py`

**Imports:**
```python
from valetudo_map_parser.config.colors import ColorsManagement
from valetudo_map_parser.config.utils import ResizeParams, async_resize_image
```

**Usage:**
- **Line 26-27:** Import statements
- **Line 84:** Store shared reference: `self._shared = coordinator.shared`
- **Line 113:** Create colors instance: `self._colours = ColorsManagement(self._shared)`
- **Line 114:** Initialize colors: `self._colours.set_initial_colours(device_info)`
- **Line 407:** Reset trims: `self._shared.reset_trims()`
- **Line 520:** Create resize params: `resize_data = ResizeParams(...)`
- **Line 527:** Resize image: `await async_resize_image(pil_img, resize_data)`

**Purpose:** Manage camera colors and image resizing operations

---

### 2.4 `utils/camera/camera_processing.py`
**Location:** `custom_components/mqtt_vacuum_camera/utils/camera/camera_processing.py`

**Imports:**
```python
from valetudo_map_parser.config.types import JsonType, PilPNG
from valetudo_map_parser.hypfer_handler import HypferMapImageHandler
from valetudo_map_parser.rand256_handler import ReImageHandler
```

**Usage:**
- **Line 18-20:** Import statements
- **Line 35:** Create Hypfer handler: `self._map_handler = HypferMapImageHandler(camera_shared)`
- **Line 36:** Create Rand256 handler: `self._re_handler = ReImageHandler(camera_shared)`
- **Line 42:** Method signature: `async def async_process_valetudo_data(self, parsed_json: JsonType) -> PilPNG | None`
- **Line 49-51:** Process Hypfer image: `pil_img, data = await self._map_handler.async_get_image(m_json=parsed_json, bytes_format=True)`
- **Line 65:** Get frame number: `self._map_handler.get_frame_number()`
- **Line 71:** Method signature: `async def async_process_rand256_data(self, parsed_json: JsonType) -> PilPNG | None`
- **Line 78-82:** Process Rand256 image: `pil_img, data = await self._re_handler.async_get_image(m_json=parsed_json, destinations=self._shared.destinations, bytes_format=True)`
- **Line 94:** Method signature: `def run_process_valetudo_data(self, parsed_json: JsonType)`
- **Line 117:** Get frame number: `self._map_handler.get_frame_number()`

**Purpose:** Core image processing using library handlers for both firmware types

---

### 2.5 `utils/connection/connector.py`
**Location:** `custom_components/mqtt_vacuum_camera/utils/connection/connector.py`

**Imports:**
```python
from valetudo_map_parser.config.types import RoomStore
```

**Usage:**
- **Line 12:** Import statement
- **Line 71:** Attribute type: `room_store: Any` (stores RoomStore instance)
- **Line 136:** Initialize room store: `room_store=RoomStore(camera_shared.file_name)`
- **Line 257:** Set rooms: `self.connector_data.room_store.set_rooms(self.mqtt_data.mqtt_segments)`

**Purpose:** Manage room data from MQTT segments

---

### 2.6 `utils/connection/decompress.py`
**Location:** `custom_components/mqtt_vacuum_camera/utils/connection/decompress.py`

**Imports:**
```python
from valetudo_map_parser.config.rand256_parser import RRMapParser
```

**Usage:**
- **Line 12:** Import statement
- **Line 48:** Create parser instance: `self._parser = RRMapParser()`
- **Line 75-76:** Parse Rand256 data: `await self._thread_pool.run_in_executor("decompression", self._parser.parse_data, decompressed, True)`

**Purpose:** Parse decompressed Rand256 binary map data

---

### 2.7 `utils/room_manager.py`
**Location:** `custom_components/mqtt_vacuum_camera/utils/room_manager.py`

**Imports:**
```python
from valetudo_map_parser.config.types import RoomStore
```

**Usage:**
- **Line 18:** Import statement
- **Line 129:** Create room store: `rooms = RoomStore(vacuum_id)`
- **Line 130:** Get room data: `room_data = rooms.get_rooms()`

**Purpose:** Retrieve room data for translation and naming operations

---

### 2.8 `utils/language_cache.py`
**Location:** `custom_components/mqtt_vacuum_camera/utils/language_cache.py`

**Imports:**
```python
from valetudo_map_parser.config.types import UserLanguageStore
```

**Usage:**
- **Line 18:** Import statement
- **Line 64:** Create instance: `user_language_store = UserLanguageStore()`
- **Line 65:** Check initialization: `await UserLanguageStore.is_initialized()`
- **Line 69:** Get all languages: `all_languages = await user_language_store.get_all_languages()`
- **Line 125-127:** Set user language: `await user_language_store.set_user_language(user_id, language)`
- **Line 137:** Mark as initialized (via method call)
- **Line 174:** Create instance: `user_language_store = UserLanguageStore()`
- **Line 175:** Get user language: `language = await user_language_store.get_user_language(active_user_id)`
- **Line 191-193:** Set user language: `await user_language_store.set_user_language(active_user_id, language)`
- **Line 341:** Set initialization flag: `setattr(UserLanguageStore, "_initialized", True)`

**Purpose:** Cache and manage user language preferences using library storage

---

### 2.9 `options_flow.py`
**Location:** `custom_components/mqtt_vacuum_camera/options_flow.py`

**Imports:**
```python
from valetudo_map_parser.config.types import RoomStore
```

**Usage:**
- **Line 21:** Import statement
- **Line 838:** Create room store: `rooms_data = RoomStore(self.file_name)`
- **Line 839:** Get rooms: `rooms_data.get_rooms()`

**Purpose:** Access room data for configuration flow options

---

## 3. Type Categories and Their Purposes

### 3.1 Core Configuration Types
| Type | Module | Purpose | Usage Count |
|------|--------|---------|-------------|
| `CameraShared` | `config.shared` | Main shared state object | 5 files |
| `CameraSharedManager` | `config.shared` | Singleton manager for CameraShared | 2 files |

### 3.2 Data Storage Types
| Type | Module | Purpose | Usage Count |
|------|--------|---------|-------------|
| `RoomStore` | `config.types` | Room data storage | 3 files |
| `UserLanguageStore` | `config.types` | User language storage | 1 file |

### 3.3 Type Aliases
| Type | Module | Purpose | Usage Count |
|------|--------|---------|-------------|
| `JsonType` | `config.types` | JSON data type alias | 1 file |
| `PilPNG` | `config.types` | PIL Image type alias | 1 file |

### 3.4 Image Processing Types
| Type | Module | Purpose | Usage Count |
|------|--------|---------|-------------|
| `HypferMapImageHandler` | `hypfer_handler` | Hypfer map processor | 1 file |
| `ReImageHandler` | `rand256_handler` | Rand256 map processor | 1 file |
| `ColorsManagement` | `config.colors` | Color configuration | 1 file |

### 3.5 Parser Types
| Type | Module | Purpose | Usage Count |
|------|--------|---------|-------------|
| `RRMapParser` | `config.rand256_parser` | Rand256 binary parser | 1 file |

### 3.6 Utility Types
| Type | Module | Purpose | Usage Count |
|------|--------|---------|-------------|
| `ResizeParams` | `config.utils` | Image resize parameters | 1 file |
| `async_resize_image` | `config.utils` | Async resize function | 1 file |
| `get_default_font_path` | (root) | Font path utility | 1 file |

---

## 4. Recommendations for Library Refactoring

### 4.1 Suggested const.py Structure
Based on usage patterns, here's a recommended structure for separating types from constants:

```python
# valetudo_map_parser/const.py
"""Constants for valetudo_map_parser library."""

# Default paths
DEFAULT_FONT_PATH = "path/to/fonts"
DEFAULT_FONT_FILE = "FiraSans.ttf"

# Image processing constants
DEFAULT_IMAGE_FORMAT = "PNG"
DEFAULT_COMPRESSION = 6

# Parser constants
RAND256_MAGIC_NUMBER = 0x72726D
HYPFER_COMPRESSION_TYPE = "zlib"

# Color constants (if applicable)
DEFAULT_FLOOR_COLOR = "#FFFFFF"
DEFAULT_WALL_COLOR = "#000000"
```

### 4.2 Suggested types.py Structure
```python
# valetudo_map_parser/types.py
"""Type definitions for valetudo_map_parser library."""

from typing import Dict, Any, Union
from PIL import Image

# Type aliases
JsonType = Dict[str, Any]
PilPNG = Image.Image

# Storage classes
class RoomStore:
    """Room data storage."""
    pass

class UserLanguageStore:
    """User language storage."""
    pass

# Parameter classes
class ResizeParams:
    """Parameters for image resizing."""
    pass
```

### 4.3 Migration Impact Analysis

**High Impact (Core Dependencies):**
- `CameraShared` - Used in 5 files, central to integration
- `RoomStore` - Used in 3 files for room management
- Image handlers - Critical for map rendering

**Medium Impact:**
- `ColorsManagement` - Used in camera.py
- `RRMapParser` - Used in decompress.py
- Storage utilities - Used in specific modules

**Low Impact:**
- Type aliases (`JsonType`, `PilPNG`) - Easy to update
- Utility functions - Single usage points

---

## 5. Current Module Structure

```
valetudo_map_parser/
├── __init__.py (get_default_font_path)
├── config/
│   ├── shared.py (CameraShared, CameraSharedManager)
│   ├── types.py (JsonType, PilPNG, RoomStore, UserLanguageStore)
│   ├── colors.py (ColorsManagement)
│   ├── utils.py (ResizeParams, async_resize_image)
│   └── rand256_parser.py (RRMapParser)
├── hypfer_handler.py (HypferMapImageHandler)
└── rand256_handler.py (ReImageHandler)
```

---

## 6. Summary Statistics

- **Total Files Using Library:** 8
- **Total Distinct Imports:** 13
- **Most Used Type:** `CameraShared` (5 files)
- **Most Used Module:** `config.types` (4 different types)
- **Critical Dependencies:** CameraShared, Image Handlers, RoomStore

---

## 7. Notes for Refactoring

1. **Backward Compatibility:** Consider maintaining import aliases during transition
2. **Type Separation:** Clear separation between types and constants will improve maintainability
3. **Import Paths:** Update all import statements when restructuring
4. **Testing:** Comprehensive testing needed after refactoring due to widespread usage
5. **Documentation:** Update all docstrings and type hints after changes

---

**End of Report**

