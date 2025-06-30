"""Type checking tests for utility modules."""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import unittest

from utils.code_metrics import (
    count_lines,
    count_functions,
    count_classes,
    calculate_cyclomatic_complexity,
)
from utils.llm_client import LLMClient
from utils.data_utils import load_json, save_json


class TestCodeMetricsTypes:
    """Test type annotations in the code_metrics module."""

    def test_count_lines(self) -> None:
        """Test count_lines with type annotations."""
        code: str = "def hello(): pass"
        lines: int = count_lines(code)
        
        # These should fail type checking
        # lines: str = count_lines(code)
        # lines: int = count_lines(123)

    def test_count_functions(self) -> None:
        """Test count_functions with type annotations."""
        code: str = "def hello(): pass"
        functions: int = count_functions(code)
        
        # These should fail type checking
        # functions: str = count_functions(code)
        # functions: int = count_functions(123)

    def test_count_classes(self) -> None:
        """Test count_classes with type annotations."""
        code: str = "class Test: pass"
        classes: int = count_classes(code)
        
        # These should fail type checking
        # classes: str = count_classes(code)
        # classes: int = count_classes(123)

    def test_calculate_cyclomatic_complexity(self) -> None:
        """Test calculate_cyclomatic_complexity with type annotations."""
        code: str = "def hello(): if True: pass"
        complexity: int = calculate_cyclomatic_complexity(code)
        
        # These should fail type checking
        # complexity: str = calculate_cyclomatic_complexity(code)
        # complexity: int = calculate_cyclomatic_complexity(123)


class TestLLMClientTypes:
    """Test type annotations in the llm_client module."""

    def test_llm_client_init(self) -> None:
        """Test initialization with type annotations."""
        # Test with default parameters
        client1: LLMClient = LLMClient()
        
        # Test with config
        config: Dict[str, Any] = {"provider": "openai", "model": "gpt-4"}
        client2: LLMClient = LLMClient(config)
        
        # This should fail type checking
        # client3: str = LLMClient()

    def test_generate_method(self) -> None:
        """Test generate method with type annotations."""
        client: LLMClient = LLMClient()
        
        prompt: str = "Generate some text"
        response: str = client.generate(prompt)
        
        # These should fail type checking
        # response: Dict[str, Any] = client.generate(prompt)
        # response: str = client.generate(123)

    def test_analyze_code_method(self) -> None:
        """Test analyze_code method with type annotations."""
        client: LLMClient = LLMClient()
        
        code: str = "def hello(): pass"
        task: str = "complexity"
        analysis: Dict[str, Any] = client.analyze_code(code, task)
        
        # These should fail type checking
        # analysis: str = client.analyze_code(code, task)
        # analysis: Dict[str, Any] = client.analyze_code(123, task)


class TestDataUtilsTypes:
    """Test type annotations in the data_utils module."""

    def test_load_json(self) -> None:
        """Test load_json with type annotations."""
        # Test with string path
        data1: Dict[str, Any] = load_json("/path/to/file.json")
        
        # Test with Path object
        path: Path = Path("/path/to/file.json")
        data2: Dict[str, Any] = load_json(path)
        
        # These should fail type checking
        # data3: str = load_json(path)
        # data4: Dict[str, Any] = load_json(123)

    def test_save_json(self) -> None:
        """Test save_json with type annotations."""
        data: Dict[str, Any] = {"key": "value"}
        
        # Test with string path
        result1: bool = save_json(data, "/path/to/file.json")
        
        # Test with Path object
        path: Path = Path("/path/to/file.json")
        result2: bool = save_json(data, path)
        
        # These should fail type checking
        # result3: str = save_json(data, path)
        # result4: bool = save_json("not a dict", path)
        # result5: bool = save_json(data, 123)