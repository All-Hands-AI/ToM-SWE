"""Tests for the code analyzer module."""

import unittest
from unittest.mock import patch, MagicMock

from tom_module.analyzer import CodeAnalyzer


class TestCodeAnalyzer(unittest.TestCase):
    """Test case for the CodeAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = CodeAnalyzer()
        self.test_code = '''
def hello(name="World"):
    """Print a greeting message."""
    print(f"Hello, {name}!")

hello()
hello("Alice")
'''

    @patch('tom_module.intent_predictor.IntentPredictor.predict_intent')
    def test_analyze(self, mock_predict_intent):
        """Test the analyze method."""
        # Mock the intent prediction
        mock_predict_intent.return_value = {
            "inferred_intent": "Print a greeting message",
            "confidence": 0.9
        }
        
        # Call the analyze method
        result = self.analyzer.analyze(self.test_code)
        
        # Check that the result contains the expected keys
        self.assertIn("lines", result)
        self.assertIn("functions", result)
        self.assertIn("classes", result)
        self.assertIn("cyclomatic_complexity", result)
        self.assertIn("imports", result)
        self.assertIn("function_names", result)
        self.assertIn("class_names", result)
        self.assertIn("intent_analysis", result)
        
        # Check specific values
        self.assertEqual(result["functions"], 1)
        self.assertEqual(result["classes"], 0)
        self.assertEqual(result["function_names"], ["hello"])
        self.assertEqual(result["intent_analysis"]["inferred_intent"], "Print a greeting message")
        
        # Verify that the mock was called
        mock_predict_intent.assert_called_once_with(self.test_code)

    @patch('tom_module.intent_predictor.IntentPredictor.predict_intent')
    def test_understand_intent(self, mock_predict_intent):
        """Test the understand_intent method."""
        # Mock the intent prediction
        mock_predict_intent.return_value = {
            "inferred_intent": "Print a greeting message",
            "confidence": 0.9
        }
        
        # Call the understand_intent method
        result = self.analyzer.understand_intent(self.test_code)
        
        # Check the result
        self.assertEqual(result["inferred_intent"], "Print a greeting message")
        self.assertEqual(result["confidence"], 0.9)
        
        # Verify that the mock was called
        mock_predict_intent.assert_called_once_with(self.test_code)

    @patch('tom_module.code_explainer.CodeExplainer.explain')
    def test_explain_code(self, mock_explain):
        """Test the explain_code method."""
        # Mock the explanation
        mock_explain.return_value = "This code defines a function that prints a greeting message."
        
        # Call the explain_code method
        result = self.analyzer.explain_code(self.test_code, audience="beginner")
        
        # Check the result
        self.assertEqual(result, "This code defines a function that prints a greeting message.")
        
        # Verify that the mock was called with the correct arguments
        mock_explain.assert_called_once_with(self.test_code, "beginner")

    @patch('tom_module.code_explainer.CodeExplainer.suggest_improvements')
    def test_suggest_improvements(self, mock_suggest_improvements):
        """Test the suggest_improvements method."""
        # Mock the suggestions
        mock_suggest_improvements.return_value = [
            {
                "type": "documentation",
                "description": "Add more detailed docstring",
                "priority": "medium"
            }
        ]
        
        # Call the suggest_improvements method
        result = self.analyzer.suggest_improvements(self.test_code)
        
        # Check the result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "documentation")
        self.assertEqual(result[0]["description"], "Add more detailed docstring")
        
        # Verify that the mock was called
        mock_suggest_improvements.assert_called_once_with(self.test_code)

    def test_calculate_metrics(self):
        """Test the _calculate_metrics method."""
        # Call the _calculate_metrics method
        result = self.analyzer._calculate_metrics(self.test_code)
        
        # Check the result
        self.assertIn("lines", result)
        self.assertIn("functions", result)
        self.assertIn("classes", result)
        self.assertIn("cyclomatic_complexity", result)
        
        self.assertEqual(result["functions"], 1)
        self.assertEqual(result["classes"], 0)

    def test_analyze_ast(self):
        """Test the _analyze_ast method."""
        # Call the _analyze_ast method
        result = self.analyzer._analyze_ast(self.test_code)
        
        # Check the result
        self.assertIn("imports", result)
        self.assertIn("function_names", result)
        self.assertIn("class_names", result)
        self.assertIn("parse_error", result)
        
        self.assertEqual(result["function_names"], ["hello"])
        self.assertEqual(result["class_names"], [])
        self.assertIsNone(result["parse_error"])
        
        # Test with invalid code
        invalid_code = "def hello() print('hello')"
        result = self.analyzer._analyze_ast(invalid_code)
        
        self.assertIsNotNone(result["parse_error"])
        self.assertEqual(result["function_names"], [])


if __name__ == "__main__":
    unittest.main()