"""Tests for map_data.py module."""

import json

import pytest

from valetudo_map_parser.map_data import HyperMapData, ImageData, RandImageData


class TestImageData:
    """Tests for ImageData class."""

    def test_find_layers_empty(self):
        """Test find_layers with empty data."""
        result_dict, result_list = ImageData.find_layers({}, None, None)
        assert result_dict == {}
        assert result_list == []

    def test_find_layers_with_map_layer(self):
        """Test find_layers with MapLayer data."""
        json_obj = {
            "__class": "MapLayer",
            "type": "floor",
            "compressedPixels": [1, 2, 3],
            "metaData": {},
        }
        result_dict, result_list = ImageData.find_layers(json_obj, None, None)
        assert "floor" in result_dict
        assert result_dict["floor"] == [[1, 2, 3]]

    def test_find_layers_with_segment(self):
        """Test find_layers with segment layer."""
        json_obj = {
            "__class": "MapLayer",
            "type": "segment",
            "compressedPixels": [1, 2, 3],
            "metaData": {"segmentId": "16", "active": True},
        }
        result_dict, result_list = ImageData.find_layers(json_obj, None, None)
        assert "segment" in result_dict
        assert 1 in result_list  # active=True converted to 1

    def test_find_layers_nested(self):
        """Test find_layers with nested structure."""
        json_obj = {
            "layers": [
                {"__class": "MapLayer", "type": "floor", "compressedPixels": [1, 2, 3]},
                {"__class": "MapLayer", "type": "wall", "compressedPixels": [4, 5, 6]},
            ]
        }
        result_dict, result_list = ImageData.find_layers(json_obj, None, None)
        assert "floor" in result_dict
        assert "wall" in result_dict

    def test_find_points_entities_empty(self):
        """Test find_points_entities with empty data."""
        result = ImageData.find_points_entities({})
        assert result == {}

    def test_find_points_entities_with_robot(self):
        """Test find_points_entities with robot position."""
        json_obj = {
            "__class": "PointMapEntity",
            "type": "robot_position",
            "points": [100, 200],
            "metaData": {"angle": 90},
        }
        result = ImageData.find_points_entities(json_obj)
        assert "robot_position" in result
        assert len(result["robot_position"]) == 1

    def test_find_paths_entities_empty(self):
        """Test find_paths_entities with empty data."""
        result = ImageData.find_paths_entities({})
        assert result == {}

    def test_find_paths_entities_with_path(self):
        """Test find_paths_entities with path data."""
        json_obj = {
            "__class": "PathMapEntity",
            "type": "path",
            "points": [10, 20, 30, 40],
        }
        result = ImageData.find_paths_entities(json_obj)
        assert "path" in result
        assert len(result["path"]) == 1

    def test_find_zone_entities_empty(self):
        """Test find_zone_entities with empty data."""
        result = ImageData.find_zone_entities({})
        assert result == {}

    def test_find_zone_entities_with_zone(self):
        """Test find_zone_entities with zone data."""
        json_obj = {
            "__class": "PolygonMapEntity",
            "type": "no_go_area",
            "points": [10, 20, 30, 40, 50, 60, 70, 80],
        }
        result = ImageData.find_zone_entities(json_obj)
        assert "no_go_area" in result
        assert len(result["no_go_area"]) == 1

    def test_get_obstacles(self):
        """Test getting obstacles from entities."""
        entities = {
            "obstacle": [
                {
                    "points": [100, 200],
                    "metaData": {"label": "shoe", "id": "obstacle_1"},
                }
            ]
        }
        obstacles = ImageData.get_obstacles(entities)
        assert len(obstacles) == 1
        assert obstacles[0]["label"] == "shoe"
        assert obstacles[0]["points"] == {"x": 100, "y": 200}


class TestRandImageData:
    """Tests for RandImageData class."""

    def test_get_rrm_image_size(self):
        """Test getting image size from RRM data."""
        json_data = {
            "image": {"dimensions": {"width": 1024, "height": 1024}}
        }
        width, height = RandImageData.get_rrm_image_size(json_data)
        assert width == 1024
        assert height == 1024

    def test_get_rrm_image_size_empty(self):
        """Test getting image size with empty data."""
        width, height = RandImageData.get_rrm_image_size({})
        assert width == 0
        assert height == 0

    def test_get_rrm_segments_ids(self):
        """Test getting segment IDs from RRM data."""
        json_data = {"image": {"segments": {"id": [16, 17, 18]}}}
        seg_ids = RandImageData.get_rrm_segments_ids(json_data)
        assert seg_ids == [16, 17, 18]

    def test_get_rrm_segments_ids_no_data(self):
        """Test getting segment IDs with no data."""
        seg_ids = RandImageData.get_rrm_segments_ids({})
        # Returns empty list when no data, not None
        assert seg_ids == []


class TestHyperMapData:
    """Tests for HyperMapData dataclass."""

    def test_initialization_empty(self):
        """Test HyperMapData initialization with no data."""
        map_data = HyperMapData()
        assert map_data.json_data is None
        assert map_data.json_id is None
        assert map_data.obstacles == {}

    def test_initialization_with_data(self, hypfer_json_data):
        """Test HyperMapData initialization with JSON data."""
        map_data = HyperMapData(json_data=hypfer_json_data, json_id="test_id")
        assert map_data.json_data == hypfer_json_data
        assert map_data.json_id == "test_id"

