"""Core functionality for ToM-SWE code processing."""

from typing import Dict, Any, Optional
from utils.code_metrics import calculate_complexity


def process_code(code: str) -> Dict[str, Any]:
    """
    Process a code snippet and return analysis results.
    
    Args:
        code: The code snippet to process.
        
    Returns:
        Dictionary containing analysis results or error information.
        
    Raises:
        TypeError: If code is not a string.
    """
    if not isinstance(code, str):
        return {
            "error": "Code must be a string",
            "success": False
        }
    
    if not code or not code.strip():
        return {
            "error": "Code snippet cannot be empty",
            "success": False
        }
    
    try:
        # Calculate code complexity metrics
        metrics = calculate_complexity(code)
        
        return {
            "success": True,
            "metrics": metrics,
            "code_length": len(code),
            "line_count": len(code.splitlines())
        }
    except Exception as e:
        return {
            "error": f"Failed to process code: {str(e)}",
            "success": False
        }


def analyze_code_snippet(code: str, include_details: bool = False) -> Dict[str, Any]:
    """
    Analyze a code snippet with optional detailed analysis.
    
    Args:
        code: The code snippet to analyze.
        include_details: Whether to include detailed analysis.
        
    Returns:
        Dictionary containing analysis results.
    """
    result = process_code(code)
    
    if not result.get("success", False):
        return result
    
    if include_details:
        # Add more detailed analysis
        result["detailed_analysis"] = {
            "has_functions": result["metrics"]["functions"] > 0,
            "has_classes": result["metrics"]["classes"] > 0,
            "complexity_level": result["metrics"]["complexity"]
        }
    
    return result