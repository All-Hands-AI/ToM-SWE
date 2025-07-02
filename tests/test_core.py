"""Tests for tom_swe.core module."""

import pytest
from tom_swe.core import process_code, analyze_code_snippet


class TestProcessCode:
    """Test the process_code function."""
    
    def test_process_code_empty_string(self):
        """Test that process_code handles empty strings correctly."""
        result = process_code("")
        
        assert result["success"] is False
        assert "error" in result
        assert "empty" in result["error"].lower()
    
    def test_process_code_whitespace_only(self):
        """Test that process_code handles whitespace-only strings correctly."""
        result = process_code("   \n\t  ")
        
        assert result["success"] is False
        assert "error" in result
        assert "empty" in result["error"].lower()
    
    def test_process_code_none_input(self):
        """Test that process_code handles None input correctly."""
        result = process_code(None)
        
        assert result["success"] is False
        assert "error" in result
        assert "string" in result["error"].lower()
    
    def test_process_code_non_string_input(self):
        """Test that process_code handles non-string input correctly."""
        result = process_code(123)
        
        assert result["success"] is False
        assert "error" in result
        assert "string" in result["error"].lower()
    
    def test_process_code_valid_simple_code(self):
        """Test that process_code handles valid simple code correctly."""
        code = "print('Hello, World!')"
        result = process_code(code)
        
        assert result["success"] is True
        assert "metrics" in result
        assert "code_length" in result
        assert "line_count" in result
        assert result["code_length"] == len(code)
        assert result["line_count"] == 1
    
    def test_process_code_valid_multiline_code(self):
        """Test that process_code handles valid multiline code correctly."""
        code = """def hello():
    print('Hello, World!')
    return True"""
        result = process_code(code)
        
        assert result["success"] is True
        assert "metrics" in result
        assert result["line_count"] == 3
        assert result["metrics"]["functions"] == 1
    
    def test_process_code_valid_class_code(self):
        """Test that process_code handles code with classes correctly."""
        code = """class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value"""
        result = process_code(code)
        
        assert result["success"] is True
        assert result["metrics"]["classes"] == 1
        assert result["metrics"]["functions"] == 2  # __init__ and get_value
    
    def test_process_code_invalid_syntax(self):
        """Test that process_code handles invalid Python syntax gracefully."""
        code = "def invalid_function(\nprint('missing closing parenthesis')"
        result = process_code(code)
        
        # Should still succeed but with fallback metrics
        assert result["success"] is True
        assert "metrics" in result


class TestAnalyzeCodeSnippet:
    """Test the analyze_code_snippet function."""
    
    def test_analyze_code_snippet_empty_string(self):
        """Test that analyze_code_snippet handles empty strings correctly."""
        result = analyze_code_snippet("")
        
        assert result["success"] is False
        assert "error" in result
    
    def test_analyze_code_snippet_basic_analysis(self):
        """Test basic analysis without details."""
        code = "print('Hello, World!')"
        result = analyze_code_snippet(code)
        
        assert result["success"] is True
        assert "metrics" in result
        assert "detailed_analysis" not in result
    
    def test_analyze_code_snippet_detailed_analysis(self):
        """Test detailed analysis."""
        code = """def hello():
    print('Hello, World!')
    return True"""
        result = analyze_code_snippet(code, include_details=True)
        
        assert result["success"] is True
        assert "detailed_analysis" in result
        assert result["detailed_analysis"]["has_functions"] is True
        assert result["detailed_analysis"]["has_classes"] is False
        assert "complexity_level" in result["detailed_analysis"]
    
    def test_analyze_code_snippet_with_class_detailed(self):
        """Test detailed analysis with class code."""
        code = """class TestClass:
    def method(self):
        pass"""
        result = analyze_code_snippet(code, include_details=True)
        
        assert result["success"] is True
        assert result["detailed_analysis"]["has_functions"] is True
        assert result["detailed_analysis"]["has_classes"] is True
    
    def test_analyze_code_snippet_error_propagation(self):
        """Test that errors from process_code are properly propagated."""
        result = analyze_code_snippet(None, include_details=True)
        
        assert result["success"] is False
        assert "error" in result
        assert "detailed_analysis" not in result


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_process_code_very_long_code(self):
        """Test processing very long code snippets."""
        # Create a long code snippet
        code = "\n".join([f"variable_{i} = {i}" for i in range(1000)])
        result = process_code(code)
        
        assert result["success"] is True
        assert result["line_count"] == 1000
    
    def test_process_code_unicode_characters(self):
        """Test processing code with unicode characters."""
        code = "# This is a comment with unicode: 你好世界\nprint('Hello, 世界!')"
        result = process_code(code)
        
        assert result["success"] is True
        assert result["line_count"] == 2
    
    def test_process_code_special_characters(self):
        """Test processing code with special characters."""
        code = """# Special characters: !@#$%^&*()
text = "String with quotes and 'apostrophes'"
regex = r"\\d+\\.\\d+"
"""
        result = process_code(code)
        
        assert result["success"] is True
        assert result["line_count"] == 3