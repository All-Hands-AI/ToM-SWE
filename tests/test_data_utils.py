"""Tests for data utilities."""

import unittest
import os
import tempfile
import json
from pathlib import Path

from utils.data_utils import (
    load_json,
    save_json,
    get_user_data_files,
    get_analysis_files,
    extract_user_id,
)


class TestDataUtils(unittest.TestCase):
    """Test case for data utilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        
        # Create some test files
        self.test_data = {"key": "value", "nested": {"inner": 42}}
        self.user_file = self.data_dir / "user_123.json"
        self.analysis_file = self.data_dir / "analysis_123.json"
        
        with open(self.user_file, "w") as f:
            json.dump(self.test_data, f)
        
        with open(self.analysis_file, "w") as f:
            json.dump(self.test_data, f)

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def test_load_json(self):
        """Test loading JSON data."""
        # Test loading existing file
        data = load_json(self.user_file)
        self.assertEqual(data, self.test_data)
        
        # Test loading with string path
        data = load_json(str(self.user_file))
        self.assertEqual(data, self.test_data)
        
        # Test loading non-existent file
        data = load_json(self.data_dir / "nonexistent.json")
        self.assertEqual(data, {})
        
        # Test loading invalid JSON
        invalid_file = self.data_dir / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("not valid json")
        
        data = load_json(invalid_file)
        self.assertEqual(data, {})

    def test_save_json(self):
        """Test saving JSON data."""
        # Test saving to a new file
        new_file = self.data_dir / "new.json"
        result = save_json(self.test_data, new_file)
        
        self.assertTrue(result)
        self.assertTrue(new_file.exists())
        
        # Verify the content
        with open(new_file, "r") as f:
            data = json.load(f)
        
        self.assertEqual(data, self.test_data)
        
        # Test saving with string path
        another_file = self.data_dir / "another.json"
        result = save_json(self.test_data, str(another_file))
        
        self.assertTrue(result)
        self.assertTrue(another_file.exists())
        
        # Test saving to a directory that doesn't exist
        nested_dir = self.data_dir / "nested" / "deep"
        nested_file = nested_dir / "nested.json"
        
        result = save_json(self.test_data, nested_file)
        
        self.assertTrue(result)
        self.assertTrue(nested_file.exists())

    def test_get_user_data_files(self):
        """Test getting user data files."""
        # Create additional user files
        (self.data_dir / "user_456.json").touch()
        (self.data_dir / "user_789.json").touch()
        
        # Create non-user files
        (self.data_dir / "other.json").touch()
        
        # Get user data files
        files = get_user_data_files(self.data_dir)
        
        # Check the result
        self.assertEqual(len(files), 3)
        self.assertIn(self.user_file, files)
        
        # Test with non-existent directory
        files = get_user_data_files(self.data_dir / "nonexistent")
        self.assertEqual(files, [])

    def test_get_analysis_files(self):
        """Test getting analysis files."""
        # Create additional analysis files
        (self.data_dir / "analysis_456.json").touch()
        (self.data_dir / "analysis_789.json").touch()
        
        # Create non-analysis files
        (self.data_dir / "other.json").touch()
        
        # Get analysis files
        files = get_analysis_files(self.data_dir)
        
        # Check the result
        self.assertEqual(len(files), 3)
        self.assertIn(self.analysis_file, files)
        
        # Test with non-existent directory
        files = get_analysis_files(self.data_dir / "nonexistent")
        self.assertEqual(files, [])

    def test_extract_user_id(self):
        """Test extracting user ID from file path."""
        # Test with user file
        user_id = extract_user_id(self.user_file)
        self.assertEqual(user_id, "123")
        
        # Test with analysis file
        user_id = extract_user_id(self.analysis_file)
        self.assertEqual(user_id, "123")
        
        # Test with string path
        user_id = extract_user_id(str(self.user_file))
        self.assertEqual(user_id, "123")
        
        # Test with other file
        other_file = self.data_dir / "other.json"
        user_id = extract_user_id(other_file)
        self.assertEqual(user_id, "")


if __name__ == "__main__":
    unittest.main()