"""Code metrics utilities for ToM-SWE."""

import ast
import re
from typing import Dict, Any, List, Optional


def count_lines(code: str) -> int:
    """Count the number of non-empty lines in code.
    
    Args:
        code: The code to analyze.
        
    Returns:
        Number of non-empty lines.
    """
    if not code:
        return 0
    
    lines = code.strip().split("\n")
    non_empty_lines = [line for line in lines if line.strip()]
    return len(non_empty_lines)


def count_functions(code: str) -> int:
    """Count the number of function definitions in code.
    
    Args:
        code: The code to analyze.
        
    Returns:
        Number of function definitions.
    """
    try:
        tree = ast.parse(code)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        return len(functions)
    except SyntaxError:
        # Fall back to regex for invalid Python code
        pattern = r"def\s+\w+\s*\("
        matches = re.findall(pattern, code)
        return len(matches)


def count_classes(code: str) -> int:
    """Count the number of class definitions in code.
    
    Args:
        code: The code to analyze.
        
    Returns:
        Number of class definitions.
    """
    try:
        tree = ast.parse(code)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        return len(classes)
    except SyntaxError:
        # Fall back to regex for invalid Python code
        pattern = r"class\s+\w+"
        matches = re.findall(pattern, code)
        return len(matches)


def calculate_cyclomatic_complexity(code: str) -> int:
    """Calculate the cyclomatic complexity of code.
    
    Args:
        code: The code to analyze.
        
    Returns:
        Cyclomatic complexity score.
    """
    try:
        tree = ast.parse(code)
        complexity = 1  # Base complexity
        
        # Count branches that increase complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
                complexity += len(node.values) - 1
            
        return complexity
    except SyntaxError:
        # Return a default value for invalid Python code
        return 1


def extract_imports(code: str) -> List[str]:
    """Extract import statements from code.
    
    Args:
        code: The code to analyze.
        
    Returns:
        List of import statements.
    """
    try:
        tree = ast.parse(code)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(f"import {name.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(name.name for name in node.names)
                imports.append(f"from {module} import {names}")
                
        return imports
    except SyntaxError:
        # Fall back to regex for invalid Python code
        import_pattern = r"^\s*(import\s+.+|from\s+.+\s+import\s+.+)$"
        lines = code.split("\n")
        imports = [line for line in lines if re.match(import_pattern, line)]
        return imports


def calculate_complexity_score(metrics: Dict[str, Any]) -> str:
    """Calculate a complexity score based on metrics.
    
    Args:
        metrics: Dictionary of code metrics.
        
    Returns:
        Complexity score as string ("low", "medium", "high").
    """
    lines = metrics.get("lines", 0)
    functions = metrics.get("functions", 0)
    classes = metrics.get("classes", 0)
    cyclomatic_complexity = metrics.get("cyclomatic_complexity", 1)
    
    # Simple heuristic for complexity
    if lines <= 50 and functions <= 3 and classes <= 1 and cyclomatic_complexity <= 5:
        return "low"
    elif lines <= 200 and functions <= 10 and classes <= 5 and cyclomatic_complexity <= 15:
        return "medium"
    else:
        return "high"


def extract_function_names(code: str) -> List[str]:
    """Extract function names from code.
    
    Args:
        code: The code to analyze.
        
    Returns:
        List of function names.
    """
    if not code:
        return []
    
    try:
        tree = ast.parse(code)
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                names.append(node.name)
        return names
    except SyntaxError:
        # Fallback to regex if AST parsing fails
        pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, code)
        return matches


def extract_class_names(code: str) -> List[tuple]:
    """Extract class names and their base classes from code.
    
    Args:
        code: The code to analyze.
        
    Returns:
        List of tuples (class_name, base_classes).
    """
    if not code:
        return []
    
    try:
        tree = ast.parse(code)
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                classes.append((node.name, base_names))
        return classes
    except SyntaxError:
        # Fallback to regex if AST parsing fails
        pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(pattern, code)
        return [(match, []) for match in matches]


def calculate_complexity(code: str) -> Dict[str, Any]:
    """Calculate overall complexity metrics for code.
    
    Args:
        code: The code to analyze.
        
    Returns:
        Dictionary containing complexity metrics.
    """
    metrics = {
        "lines": count_lines(code),
        "functions": count_functions(code),
        "classes": count_classes(code),
        "cyclomatic_complexity": calculate_cyclomatic_complexity(code)
    }
    
    complexity_score = calculate_complexity_score(metrics)
    metrics["complexity"] = complexity_score
    
    return metrics