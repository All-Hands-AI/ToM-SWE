"""Code analyzer module for ToM-SWE."""

import ast
import logging
from typing import Dict, Any, List, Optional, Tuple

from .intent_predictor import IntentPredictor
from .code_explainer import CodeExplainer
from utils.code_metrics import (
    count_lines,
    count_functions,
    count_classes,
    calculate_cyclomatic_complexity,
    calculate_complexity_score,
)

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """Analyzer for code using Theory of Mind principles."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the code analyzer.
        
        Args:
            llm_config: Configuration for the language model.
        """
        self.intent_predictor = IntentPredictor(llm_config)
        self.code_explainer = CodeExplainer(llm_config)
        logger.info("CodeAnalyzer initialized")

    def analyze(self, code: str) -> Dict[str, Any]:
        """Analyze code and return metrics and insights.
        
        Args:
            code: The code to analyze.
            
        Returns:
            Dict containing analysis results.
        """
        logger.info("Analyzing code snippet")
        
        # Basic metrics
        metrics = self._calculate_metrics(code)
        
        # AST-based analysis
        ast_analysis = self._analyze_ast(code)
        
        # Intent analysis
        intent_analysis = self.intent_predictor.predict_intent(code)
        
        # Combine all analyses
        analysis = {
            **metrics,
            **ast_analysis,
            "intent_analysis": intent_analysis,
        }
        
        logger.info("Code analysis completed")
        return analysis

    def understand_intent(self, code: str) -> Dict[str, Any]:
        """Understand the intent behind the code.
        
        Args:
            code: The code to analyze.
            
        Returns:
            Dict containing intent analysis.
        """
        return self.intent_predictor.predict_intent(code)

    def explain_code(self, code: str, audience: str = "general") -> str:
        """Explain the code for a specific audience.
        
        Args:
            code: The code to explain.
            audience: Target audience (e.g., "beginner", "expert").
            
        Returns:
            String explanation of the code.
        """
        return self.code_explainer.explain(code, audience)

    def suggest_improvements(self, code: str) -> List[Dict[str, Any]]:
        """Suggest improvements for the code.
        
        Args:
            code: The code to improve.
            
        Returns:
            List of improvement suggestions.
        """
        return self.code_explainer.suggest_improvements(code)

    def calculate_metrics(self, code: str) -> Dict[str, Any]:
        """Calculate basic code metrics (public interface).
        
        Args:
            code: The code to analyze.
            
        Returns:
            Dict containing code metrics.
        """
        return self._calculate_metrics(code)

    def _calculate_metrics(self, code: str) -> Dict[str, Any]:
        """Calculate basic code metrics.
        
        Args:
            code: The code to analyze.
            
        Returns:
            Dict containing code metrics.
        """
        metrics = {
            "lines": count_lines(code),
            "functions": count_functions(code),
            "classes": count_classes(code),
            "cyclomatic_complexity": calculate_cyclomatic_complexity(code),
        }
        
        # Add complexity score
        metrics["complexity"] = calculate_complexity_score(metrics)  # type: ignore
        
        return metrics

    def _analyze_ast(self, code: str) -> Dict[str, Any]:
        """Analyze code using AST.
        
        Args:
            code: The code to analyze.
            
        Returns:
            Dict containing AST analysis results.
        """
        try:
            tree = ast.parse(code)
            
            # Extract imports
            imports = self._extract_imports(tree)
            
            # Extract function and class names
            functions, classes = self._extract_definitions(tree)
            
            return {
                "imports": imports,
                "function_names": functions,
                "class_names": classes,
                "parse_error": None,
            }
        except SyntaxError as e:
            logger.warning(f"Syntax error in code: {e}")
            return {
                "imports": [],
                "function_names": [],
                "class_names": [],
                "parse_error": str(e),
            }

    def _extract_imports(self, tree: ast.Module) -> List[str]:
        """Extract import statements from AST.
        
        Args:
            tree: AST tree.
            
        Returns:
            List of import statements.
        """
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    imports.append(f"{module}.{name.name}")
        return imports

    def _extract_definitions(self, tree: ast.Module) -> Tuple[List[str], List[str]]:
        """Extract function and class definitions from AST.
        
        Args:
            tree: AST tree.
            
        Returns:
            Tuple of (function_names, class_names).
        """
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
                
        return functions, classes