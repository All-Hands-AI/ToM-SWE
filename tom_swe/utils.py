"""Utility functions for ToM-SWE."""

import re
from typing import Dict, List, Any, Optional


def count_lines(code: str) -> int:
    """
    Count the number of lines in a code snippet.

    Args:
        code: The code snippet.

    Returns:
        The number of lines.
    """
    if not code:
        return 0
    return code.count('\n') + 1


def count_functions(code: str) -> int:
    """
    Count the number of function definitions in a code snippet.

    Args:
        code: The code snippet.

    Returns:
        The number of function definitions.
    """
    if not code:
        return 0
    # Simple regex to match function definitions in Python
    return len(re.findall(r'def\s+\w+\s*\(', code))


def count_classes(code: str) -> int:
    """
    Count the number of class definitions in a code snippet.

    Args:
        code: The code snippet.

    Returns:
        The number of class definitions.
    """
    if not code:
        return 0
    # Simple regex to match class definitions in Python
    return len(re.findall(r'class\s+\w+\s*(\(|:)', code))


def extract_imports(code: str) -> List[str]:
    """
    Extract import statements from a code snippet.

    Args:
        code: The code snippet.

    Returns:
        A list of import statements.
    """
    if not code:
        return []
    
    # Match import statements
    import_pattern = r'^(?:from\s+[\w.]+\s+import\s+(?:[\w.]+(?:\s*,\s*[\w.]+)*)|import\s+(?:[\w.]+(?:\s*,\s*[\w.]+)*))(?:\s+as\s+[\w.]+)?'
    return re.findall(import_pattern, code, re.MULTILINE)


def extract_function_names(code: str) -> List[str]:
    """
    Extract function names from a code snippet.

    Args:
        code: The code snippet.

    Returns:
        A list of function names.
    """
    if not code:
        return []
    
    # Match function definitions and extract the name
    function_pattern = r'def\s+(\w+)\s*\('
    return re.findall(function_pattern, code)


def extract_class_names(code: str) -> List[str]:
    """
    Extract class names from a code snippet.

    Args:
        code: The code snippet.

    Returns:
        A list of class names.
    """
    if not code:
        return []
    
    # Match class definitions and extract the name
    class_pattern = r'class\s+(\w+)\s*(\(|:)'
    return re.findall(class_pattern, code)


def calculate_complexity(code: str) -> Dict[str, Any]:
    """
    Calculate complexity metrics for a code snippet.

    Args:
        code: The code snippet.

    Returns:
        A dictionary of complexity metrics.
    """
    if not code:
        return {
            "lines": 0,
            "functions": 0,
            "classes": 0,
            "imports": [],
            "complexity": "low"
        }
    
    lines = count_lines(code)
    functions = count_functions(code)
    classes = count_classes(code)
    imports = extract_imports(code)
    
    # Simple complexity heuristic
    if lines > 100 or functions > 10 or classes > 5:
        complexity = "high"
    elif lines > 50 or functions > 5 or classes > 2:
        complexity = "medium"
    else:
        complexity = "low"
    
    return {
        "lines": lines,
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "complexity": complexity
    }