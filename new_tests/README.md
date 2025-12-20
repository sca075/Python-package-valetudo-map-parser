# Valetudo Map Parser Test Suite

This directory contains comprehensive pytest test suites for the `valetudo_map_parser` library.

## Structure

```
new_tests/
├── conftest.py                          # Pytest fixtures and configuration
├── config/                              # Tests for config module
│   ├── test_types.py                   # Tests for type classes (RoomStore, TrimsData, etc.)
│   ├── test_shared.py                  # Tests for CameraShared and CameraSharedManager
│   ├── test_colors.py                  # Tests for color management
│   ├── test_drawable.py                # Tests for drawable elements
│   └── test_status_text.py             # Tests for status text generation
├── handlers/                            # Tests for handler modules
│   └── (handler tests to be added)
├── integration/                         # Integration tests
│   └── test_basic_integration.py       # End-to-end workflow tests
└── test_map_data.py                    # Tests for map data processing
```

## Running Tests

### Run all tests
```bash
pytest new_tests/
```

### Run specific test file
```bash
pytest new_tests/config/test_types.py
```

### Run specific test class
```bash
pytest new_tests/config/test_types.py::TestRoomStore
```

### Run specific test
```bash
pytest new_tests/config/test_types.py::TestRoomStore::test_singleton_behavior
```

### Run with verbose output
```bash
pytest new_tests/ -v
```

### Run with coverage
```bash
pytest new_tests/ --cov=valetudo_map_parser --cov-report=html
```

## Test Coverage

The test suite covers:

### Config Module
- **types.py**: All dataclasses and singleton stores (RoomStore, UserLanguageStore, SnapshotStore, TrimCropData, TrimsData, FloorData)
- **shared.py**: CameraShared and CameraSharedManager classes
- **colors.py**: Color management and conversion
- **drawable.py**: Drawing utilities and element configuration
- **status_text**: Status text generation and translations

### Map Data
- **map_data.py**: JSON parsing, entity extraction, coordinate conversion

### Integration Tests
- End-to-end image generation for Hypfer vacuums
- End-to-end image generation for Rand256 vacuums
- Multi-vacuum support
- Room detection and storage

## Test Data

Tests use sample data from the `tests/` directory:
- **Hypfer JSON samples**: `test.json`, `glossyhardtofindnarwhal.json`, `l10_carpet.json`
- **Rand256 binary samples**: `map_data_*.bin` files

## Fixtures

Common fixtures are defined in `conftest.py`:
- `hypfer_json_data`: Loads Hypfer JSON test data
- `rand256_bin_data`: Loads Rand256 binary test data
- `camera_shared`: Creates a CameraShared instance
- `room_store`: Creates a RoomStore instance
- `test_image`: Creates a test PIL Image
- `device_info`: Sample device information
- `vacuum_id`: Test vacuum identifier

## Adding New Tests

1. Create a new test file in the appropriate directory
2. Import necessary modules and fixtures
3. Create test classes and methods following pytest conventions
4. Use descriptive test names that explain what is being tested
5. Include docstrings for test classes and methods
6. Use fixtures from `conftest.py` where applicable

## Best Practices

- Keep tests fast and focused
- Test one thing per test method
- Use parametrized tests for testing multiple inputs
- Clean up resources (images, files) after tests
- Mock external dependencies when appropriate
- Test both success and failure cases
- Test edge cases and boundary conditions

