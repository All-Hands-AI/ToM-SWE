"""Tests for core functionality."""

import unittest
from tom_swe.core import process_code

class TestCore(unittest.TestCase):
    """Test cases for core functionality."""
    
    def test_process_code_with_content(self):
        """Test processing code with content."""
        code = "def hello():\n    print('Hello, world!')"
        result = process_code(code)
        self.assertEqual(result["length"], len(code))
        self.assertEqual(result["lines"], 2)
    
    # Missing test for empty code snippet
    
if __name__ == "__main__":
    unittest.main()