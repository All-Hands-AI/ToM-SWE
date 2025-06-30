"""Type checking tests for the analyzer module."""

from typing import Dict, Any, List, Optional
import unittest

from tom_module.analyzer import CodeAnalyzer


class TestAnalyzerTypes:
    """Test type annotations in the analyzer module."""

    def test_analyzer_init(self) -> None:
        """Test initialization with type annotations."""
        # Test with default parameters
        analyzer1: CodeAnalyzer = CodeAnalyzer()
        
        # Test with config
        config: Dict[str, Any] = {"provider": "openai", "model": "gpt-4"}
        analyzer2: CodeAnalyzer = CodeAnalyzer(config)
        
        # This should fail type checking
        # analyzer3: str = CodeAnalyzer()

    def test_analyze_method(self) -> None:
        """Test analyze method with type annotations."""
        analyzer: CodeAnalyzer = CodeAnalyzer()
        
        code: str = "def hello(): pass"
        result: Dict[str, Any] = analyzer.analyze(code)
        
        # These should fail type checking
        # result: str = analyzer.analyze(code)
        # result: Dict[str, Any] = analyzer.analyze(123)

    def test_understand_intent_method(self) -> None:
        """Test understand_intent method with type annotations."""
        analyzer: CodeAnalyzer = CodeAnalyzer()
        
        code: str = "def hello(): pass"
        intent: Dict[str, Any] = analyzer.understand_intent(code)
        
        # These should fail type checking
        # intent: str = analyzer.understand_intent(code)
        # intent: Dict[str, Any] = analyzer.understand_intent(123)

    def test_explain_code_method(self) -> None:
        """Test explain_code method with type annotations."""
        analyzer: CodeAnalyzer = CodeAnalyzer()
        
        code: str = "def hello(): pass"
        explanation: str = analyzer.explain_code(code)
        explanation_with_audience: str = analyzer.explain_code(code, audience="beginner")
        
        # These should fail type checking
        # explanation: Dict[str, Any] = analyzer.explain_code(code)
        # explanation: str = analyzer.explain_code(123)

    def test_suggest_improvements_method(self) -> None:
        """Test suggest_improvements method with type annotations."""
        analyzer: CodeAnalyzer = CodeAnalyzer()
        
        code: str = "def hello(): pass"
        suggestions: List[Dict[str, Any]] = analyzer.suggest_improvements(code)
        
        # These should fail type checking
        # suggestions: str = analyzer.suggest_improvements(code)
        # suggestions: List[Dict[str, Any]] = analyzer.suggest_improvements(123)

    def test_private_methods(self) -> None:
        """Test private methods with type annotations."""
        analyzer: CodeAnalyzer = CodeAnalyzer()
        
        code: str = "def hello(): pass"
        metrics: Dict[str, Any] = analyzer._calculate_metrics(code)
        ast_analysis: Dict[str, Any] = analyzer._analyze_ast(code)
        
        # These should fail type checking
        # metrics: str = analyzer._calculate_metrics(code)
        # ast_analysis: str = analyzer._analyze_ast(code)