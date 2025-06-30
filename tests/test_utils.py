"""Tests for utility functions."""

import pytest
from tom_swe.utils import (
    count_lines, count_functions, count_classes,
    extract_imports, extract_function_names, extract_class_names,
    calculate_complexity
)


def test_count_lines():
    """Test line counting functionality."""
    assert count_lines("") == 0
    assert count_lines("single line") == 1
    assert count_lines("line 1\nline 2") == 2
    assert count_lines("line 1\nline 2\nline 3") == 3


def test_count_functions():
    """Test function counting functionality."""
    assert count_functions("") == 0
    assert count_functions("x = 1") == 0
    assert count_functions("def func(): pass") == 1
    assert count_functions("def func1():\n    pass\n\ndef func2():\n    pass") == 2
    assert count_functions("def func(x, y):\n    return x + y") == 1


def test_count_classes():
    """Test class counting functionality."""
    assert count_classes("") == 0
    assert count_classes("x = 1") == 0
    assert count_classes("class Test: pass") == 1
    assert count_classes("class Test1:\n    pass\n\nclass Test2:\n    pass") == 2
    assert count_classes("class Test(object):\n    def __init__(self):\n        pass") == 1


def test_extract_imports():
    """Test import extraction functionality."""
    assert extract_imports("") == []
    assert extract_imports("x = 1") == []
    
    code = "import os\nimport sys\nfrom datetime import datetime"
    imports = extract_imports(code)
    assert len(imports) == 3
    assert "import os" in imports
    assert "import sys" in imports
    assert "from datetime import datetime" in imports
    
    code = "import os, sys\nfrom datetime import datetime, timedelta"
    imports = extract_imports(code)
    assert len(imports) == 2
    assert "import os, sys" in imports
    assert "from datetime import datetime, timedelta" in imports


def test_extract_function_names():
    """Test function name extraction functionality."""
    assert extract_function_names("") == []
    assert extract_function_names("x = 1") == []
    
    code = "def func1(): pass\ndef func2(x): return x"
    names = extract_function_names(code)
    assert len(names) == 2
    assert "func1" in names
    assert "func2" in names
    
    code = "def __init__(self): pass"
    names = extract_function_names(code)
    assert len(names) == 1
    assert "__init__" in names


def test_extract_class_names():
    """Test class name extraction functionality."""
    assert extract_class_names("") == []
    assert extract_class_names("x = 1") == []
    
    code = "class Test1: pass\nclass Test2(object): pass"
    names = extract_class_names(code)
    assert len(names) == 2
    assert any(name[0] == "Test1" for name in names)
    assert any(name[0] == "Test2" for name in names)


def test_calculate_complexity():
    """Test complexity calculation functionality."""
    # Empty code
    result = calculate_complexity("")
    assert result["lines"] == 0
    assert result["functions"] == 0
    assert result["classes"] == 0
    assert result["complexity"] == "low"
    
    # Simple code (low complexity)
    code = "x = 1\ny = 2\nprint(x + y)"
    result = calculate_complexity(code)
    assert result["lines"] == 3
    assert result["functions"] == 0
    assert result["classes"] == 0
    assert result["complexity"] == "low"
    
    # Medium complexity
    code = """
def func1():
    pass

def func2():
    pass

def func3():
    pass

def func4():
    pass

def func5():
    pass

def func6():
    pass
"""
    result = calculate_complexity(code)
    assert result["functions"] == 6
    assert result["complexity"] == "medium"
    
    # High complexity
    code = """
class Class1:
    def __init__(self):
        pass

class Class2:
    def __init__(self):
        pass

class Class3:
    def __init__(self):
        pass

class Class4:
    def __init__(self):
        pass

class Class5:
    def __init__(self):
        pass

class Class6:
    def __init__(self):
        pass
"""
    result = calculate_complexity(code)
    assert result["classes"] == 6
    assert result["complexity"] == "high"