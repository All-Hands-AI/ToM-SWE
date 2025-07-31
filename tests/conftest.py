"""
Test fixtures and mock data for tom_module tests.
"""

import json
import os
import tempfile
from typing import Any, Dict, Generator, List

import pytest

from tom_swe.tom_module import UserMentalStateAnalyzer, UserMessageAnalysis


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_user_data() -> Dict[str, Any]:
    """Sample user data with multiple sessions."""
    return {
        "session_001": {
            "convo_start": "2024-01-15T10:00:00",
            "convo_end": "2024-01-15T10:30:00",
            "convo_events": [
                {
                    "id": "event_1",
                    "source": "user",
                    "content": "I need help debugging this Python function that keeps throwing errors",
                },
                {
                    "id": "event_2",
                    "source": "assistant",
                    "content": "I'd be happy to help you debug that function.",
                },
                {
                    "id": "event_3",
                    "source": "user",
                    "content": "The error says 'list index out of range' but I can't figure out why",
                },
            ],
        },
        "session_002": {
            "convo_start": "2024-01-16T14:00:00",
            "convo_end": "2024-01-16T14:15:00",
            "convo_events": [
                {
                    "id": "event_4",
                    "source": "user",
                    "content": "Can you help me optimize this machine learning model?",
                },
                {
                    "id": "event_5",
                    "source": "assistant",
                    "content": "Sure! What kind of optimization are you looking for?",
                },
            ],
        },
    }


@pytest.fixture
def empty_user_data() -> Dict[str, Any]:
    """User data with no sessions."""
    return {}


@pytest.fixture
def sample_message_analyses() -> List[UserMessageAnalysis]:
    """Sample UserMessageAnalysis objects for testing."""
    return [
        UserMessageAnalysis(
            intent="debugging",
            emotions=["frustrated", "confused"],
            preference=["detailed_explanations"],
            user_modeling="struggling with Python debugging",
            should_ask_clarification=True,
        ),
        UserMessageAnalysis(
            intent="debugging",
            emotions=["confused"],
            preference=["step_by_step_help"],
            user_modeling="needs guidance on error handling",
            should_ask_clarification=False,
        ),
        UserMessageAnalysis(
            intent="optimization",
            emotions=["focused", "exploratory"],
            preference=["performance_tips"],
            user_modeling="interested in ML optimization",
            should_ask_clarification=False,
        ),
    ]


@pytest.fixture
def mock_llm_response() -> str:
    """Mock LLM response for testing."""
    return """
    {
        "intent": "debugging",
        "emotions": ["frustrated"],
        "preference": ["detailed_help"],
        "user_modeling": "user needs debugging help",
        "should_ask_clarification": true
    }
    """


@pytest.fixture
def analyzer_with_temp_dir(temp_dir: str) -> tuple[UserMentalStateAnalyzer, str]:
    """UserMentalStateAnalyzer with temporary directories."""
    processed_data_dir = os.path.join(temp_dir, "processed_data")
    os.makedirs(processed_data_dir, exist_ok=True)

    analyzer = UserMentalStateAnalyzer(processed_data_dir=processed_data_dir)
    return analyzer, temp_dir


@pytest.fixture
def user_file_setup(
    analyzer_with_temp_dir: tuple[UserMentalStateAnalyzer, str],
    sample_user_data: Dict[str, Any],
) -> tuple[UserMentalStateAnalyzer, str, str]:
    """Set up a user file with sample data."""
    analyzer, temp_dir = analyzer_with_temp_dir
    user_id = "test_user_123"

    # Create user file
    user_file = os.path.join(analyzer.processed_data_dir, f"{user_id}.json")
    with open(user_file, "w", encoding="utf-8") as f:
        json.dump(sample_user_data, f, indent=2)

    return analyzer, temp_dir, user_id


@pytest.fixture
def mock_analyze_user_mental_state() -> Any:
    """Mock the analyze_all_session_messages method."""

    async def _mock_method(
        user_id: str, session_id: str
    ) -> tuple[List[UserMessageAnalysis], List[str]]:
        # Return different analyses based on session_id for testing
        if session_id == "session_001":
            analyses = [
                UserMessageAnalysis(
                    intent="debugging",
                    emotions=["frustrated"],
                    preference=["detailed_help"],
                    user_modeling="needs debugging assistance",
                    should_ask_clarification=True,
                ),
                UserMessageAnalysis(
                    intent="debugging",
                    emotions=["confused"],
                    preference=["step_by_step"],
                    user_modeling="struggling with errors",
                    should_ask_clarification=False,
                ),
            ]
            messages = ["Mock message 1", "Mock message 2"]
            return (analyses, messages)
        elif session_id == "session_002":
            analyses = [
                UserMessageAnalysis(
                    intent="optimization",
                    emotions=["focused"],
                    preference=["performance_tips"],
                    user_modeling="ML optimization focused",
                    should_ask_clarification=False,
                )
            ]
            messages = ["Mock message 3"]
            return (analyses, messages)
        return ([], [])

    return _mock_method
