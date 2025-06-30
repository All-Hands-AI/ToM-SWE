"""Tests for the ToM module."""

import pytest
from unittest.mock import MagicMock, patch

from tom_module.tom_module import ToMModule, LLMConfig


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    with patch('utils.llm_client.LLMClient') as mock_client:
        # Configure the mock to return a specific response
        instance = mock_client.return_value
        instance.generate.return_value = '{"summary": "Test summary", "complexity": "low"}'
        instance.analyze_code.return_value = {
            "summary": "Test summary",
            "complexity": "low",
            "potential_issues": [],
            "suggestions": [],
            "code_quality": "high"
        }
        yield mock_client


def test_tom_module_initialization():
    """Test ToMModule initialization."""
    # Test with default config
    tom = ToMModule()
    assert tom.llm_config is not None
    assert tom.llm_client is not None
    
    # Test with custom config
    config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    tom = ToMModule(llm_config=config)
    assert tom.llm_config == config


def test_analyze_code(mock_llm_client):
    """Test code analysis functionality."""
    tom = ToMModule()
    
    code = """
    def hello_world():
        print("Hello, world!")
    """
    
    result = tom.analyze_code(code)
    
    # Check that the result contains expected keys
    assert "lines" in result
    assert "functions" in result
    assert "classes" in result
    assert "complexity" in result
    assert "llm_analysis" in result
    
    # Check specific values
    assert result["functions"] == 1
    assert result["classes"] == 0
    assert result["llm_analysis"]["complexity"] == "low"


def test_understand_intent(mock_llm_client):
    """Test intent understanding functionality."""
    tom = ToMModule()
    
    code = """
    def hello_world():
        print("Hello, world!")
    """
    
    # Configure mock to return a specific response for this test
    tom.llm_client.generate = MagicMock(return_value="""
    {
        "inferred_intent": "Print a greeting message",
        "goals": ["Display text to console"],
        "assumptions": ["Standard output is available"],
        "mental_model": "Simple procedural programming",
        "confidence": 0.9
    }
    """)
    
    result = tom.understand_intent(code)
    
    # Check that the result contains expected keys
    assert "inferred_intent" in result
    assert "goals" in result
    assert "assumptions" in result
    assert "mental_model" in result
    assert "confidence" in result
    
    # Check specific values
    assert result["inferred_intent"] == "Print a greeting message"
    assert "Display text to console" in result["goals"]
    assert result["confidence"] == 0.9


def test_predict_changes(mock_llm_client):
    """Test change prediction functionality."""
    tom = ToMModule()
    
    code = """
    def hello_world():
        print("Hello, world!")
    """
    
    intent = "Add a parameter to customize the greeting"
    
    # Configure mock to return a specific response for this test
    tom.llm_client.generate = MagicMock(return_value="""
    {
        "predicted_changes": [
            {
                "type": "modify",
                "location": "function signature",
                "description": "Add a name parameter with default value"
            },
            {
                "type": "modify",
                "location": "print statement",
                "description": "Use the name parameter in the greeting"
            }
        ],
        "reasoning": "The intent suggests personalizing the greeting, which requires a parameter",
        "confidence": 0.8
    }
    """)
    
    result = tom.predict_changes(code, intent)
    
    # Check that the result contains expected keys
    assert "predicted_changes" in result
    assert "reasoning" in result
    assert "confidence" in result
    
    # Check specific values
    assert len(result["predicted_changes"]) == 2
    assert result["predicted_changes"][0]["type"] == "modify"
    assert "function signature" in result["predicted_changes"][0]["location"]
    assert result["confidence"] == 0.8


def test_explain_code(mock_llm_client):
    """Test code explanation functionality."""
    tom = ToMModule()
    
    code = """
    def hello_world():
        print("Hello, world!")
    """
    
    # Configure mock to return a specific response for this test
    expected_explanation = "This code defines a function called hello_world that prints 'Hello, world!' to the console."
    tom.llm_client.generate = MagicMock(return_value=expected_explanation)
    
    result = tom.explain_code(code, audience="beginner")
    
    # Check the result
    assert result == expected_explanation
    
    # Verify that the audience was included in the prompt
    call_args = tom.llm_client.generate.call_args[0][0]
    assert "beginner" in call_args


def test_suggest_improvements(mock_llm_client):
    """Test improvement suggestion functionality."""
    tom = ToMModule()
    
    code = """
    def hello_world():
        print("Hello, world!")
    """
    
    # Configure mock to return a specific response for this test
    tom.llm_client.generate = MagicMock(return_value="""
    {
        "suggestions": [
            {
                "type": "documentation",
                "description": "Add a docstring to explain the function",
                "code_example": "def hello_world():\\n    \\"\\"\\"Print a greeting message.\\"\\"\\"\\"\\n    print(\\"Hello, world!\\")",
                "priority": "medium"
            }
        ]
    }
    """)
    
    result = tom.suggest_improvements(code)
    
    # Check the result
    assert len(result) == 1
    assert result[0]["type"] == "documentation"
    assert "Add a docstring" in result[0]["description"]
    assert result[0]["priority"] == "medium"