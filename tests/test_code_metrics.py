"""Tests for code metrics utilities."""

import unittest
from utils.code_metrics import (
    count_lines,
    count_functions,
    count_classes,
    calculate_cyclomatic_complexity,
    extract_imports,
    calculate_complexity_score,
)


class TestCodeMetrics(unittest.TestCase):
    """Test case for code metrics utilities."""

    def test_count_lines(self):
        """Test counting lines of code."""
        # Empty code
        self.assertEqual(count_lines(""), 0)
        
        # Single line
        self.assertEqual(count_lines("print('hello')"), 1)
        
        # Multiple lines
        code = "def hello():\n    print('hello')\n\nhello()"
        self.assertEqual(count_lines(code), 3)
        
        # Lines with only whitespace should be ignored
        code = "def hello():\n    \n    print('hello')\n\nhello()"
        self.assertEqual(count_lines(code), 3)

    def test_count_functions(self):
        """Test counting function definitions."""
        # No functions
        self.assertEqual(count_functions("x = 1"), 0)
        
        # One function
        self.assertEqual(count_functions("def hello(): pass"), 1)
        
        # Multiple functions
        code = "def hello(): pass\ndef goodbye(): pass"
        self.assertEqual(count_functions(code), 2)
        
        # Nested functions
        code = "def outer():\n    def inner(): pass\n    return inner"
        self.assertEqual(count_functions(code), 2)
        
        # Invalid Python code (should use regex fallback)
        code = "def hello() print('hello')"
        self.assertEqual(count_functions(code), 1)

    def test_count_classes(self):
        """Test counting class definitions."""
        # No classes
        self.assertEqual(count_classes("x = 1"), 0)
        
        # One class
        self.assertEqual(count_classes("class Test: pass"), 1)
        
        # Multiple classes
        code = "class Test1: pass\nclass Test2: pass"
        self.assertEqual(count_classes(code), 2)
        
        # Nested classes
        code = "class Outer:\n    class Inner: pass"
        self.assertEqual(count_classes(code), 2)
        
        # Invalid Python code (should use regex fallback)
        code = "class Test print('hello')"
        self.assertEqual(count_classes(code), 1)

    def test_calculate_cyclomatic_complexity(self):
        """Test calculating cyclomatic complexity."""
        # Simple function
        code = "def hello(): print('hello')"
        self.assertEqual(calculate_cyclomatic_complexity(code), 1)
        
        # Function with if statement
        code = "def hello():\n    if True:\n        print('hello')"
        self.assertEqual(calculate_cyclomatic_complexity(code), 2)
        
        # Function with if-else
        code = "def hello():\n    if True:\n        print('hello')\n    else:\n        print('goodbye')"
        self.assertEqual(calculate_cyclomatic_complexity(code), 2)
        
        # Function with loop
        code = "def hello():\n    for i in range(10):\n        print(i)"
        self.assertEqual(calculate_cyclomatic_complexity(code), 2)
        
        # Function with boolean operators
        code = "def hello():\n    if True and False:\n        print('hello')"
        self.assertEqual(calculate_cyclomatic_complexity(code), 2)
        
        # Complex function
        code = """
def complex():
    for i in range(10):
        if i % 2 == 0:
            print('even')
        else:
            print('odd')
            
        if i > 5 and i < 8:
            print('middle')
        """
        self.assertEqual(calculate_cyclomatic_complexity(code), 5)
        
        # Invalid Python code
        code = "def hello() if True: print('hello')"
        self.assertEqual(calculate_cyclomatic_complexity(code), 1)

    def test_extract_imports(self):
        """Test extracting import statements."""
        # No imports
        self.assertEqual(extract_imports("x = 1"), [])
        
        # Simple import
        self.assertEqual(extract_imports("import os"), ["import os"])
        
        # Multiple imports
        code = "import os\nimport sys"
        imports = extract_imports(code)
        self.assertEqual(len(imports), 2)
        self.assertIn("import os", imports)
        self.assertIn("import sys", imports)
        
        # From import
        self.assertEqual(extract_imports("from os import path"), ["from os import path"])
        
        # Multiple from imports
        code = "from os import path, environ"
        imports = extract_imports(code)
        self.assertEqual(len(imports), 1)
        self.assertIn("from os import path, environ", imports)
        
        # Invalid Python code
        code = "import os from sys import path"
        imports = extract_imports(code)
        self.assertEqual(len(imports), 2)

    def test_calculate_complexity_score(self):
        """Test calculating complexity score."""
        # Low complexity
        metrics = {
            "lines": 20,
            "functions": 2,
            "classes": 0,
            "cyclomatic_complexity": 3
        }
        self.assertEqual(calculate_complexity_score(metrics), "low")
        
        # Medium complexity
        metrics = {
            "lines": 100,
            "functions": 5,
            "classes": 2,
            "cyclomatic_complexity": 10
        }
        self.assertEqual(calculate_complexity_score(metrics), "medium")
        
        # High complexity
        metrics = {
            "lines": 300,
            "functions": 15,
            "classes": 8,
            "cyclomatic_complexity": 20
        }
        self.assertEqual(calculate_complexity_score(metrics), "high")
        
        # Missing metrics
        metrics = {
            "lines": 100
        }
        self.assertEqual(calculate_complexity_score(metrics), "low")


if __name__ == "__main__":
    unittest.main()