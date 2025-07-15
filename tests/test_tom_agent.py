"""
Test ToM Agent functions for instruction improvement and next action prediction.
"""

import json
import os
import tempfile
from datetime import datetime
from typing import Iterator
from unittest.mock import AsyncMock, patch

import pytest

from tom_swe.database import SessionSummary, UserProfile
from tom_swe.tom_agent import (
    InstructionRecommendation,
    NextActionSuggestion,
    ToMAgent,
    ToMAgentConfig,
    UserContext,
)


@pytest.fixture
def temp_dir() -> Iterator[str]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_user_context() -> UserContext:
    """Sample user context for testing."""
    user_profile = UserProfile(
        user_id="test_user_123",
        total_sessions=5,
        overall_description="A focused developer who prefers detailed guidance and systematic approaches to problem-solving.",
        intent_distribution={"debugging": 15, "optimization": 10, "learning": 5},
        emotion_distribution={"focused": 12, "frustrated": 8, "curious": 10},
        preference_summary=["detailed_explanations", "step_by_step_help"],
    )

    session_summary = SessionSummary(
        session_id="session_001",
        timestamp=datetime.now(),
        message_count=10,
        intent_distribution={"debugging": 6, "optimization": 4},
        emotion_distribution={"focused": 5, "frustrated": 3, "curious": 2},
        key_preferences=["detailed_help", "code_examples"],
        user_modeling_summary="User worked on debugging Python functions",
        clarification_requests=1,
        session_start="2024-01-15T10:00:00",
        session_end="2024-01-15T10:30:00",
    )

    return UserContext(
        user_id="test_user_123",
        user_profile=user_profile,
        recent_sessions=[session_summary],
        current_query="Working on debugging a Python application",
        preferences=["detailed_explanations", "step_by_step_help"],
        mental_state_summary="A focused developer who prefers detailed guidance and systematic approaches to problem-solving.",
    )


@pytest.fixture
def tom_agent(temp_dir: str) -> ToMAgent:
    """Create a ToM Agent for testing."""
    processed_data_dir = os.path.join(temp_dir, "processed_data")
    user_model_dir = os.path.join(temp_dir, "user_model")
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(user_model_dir, exist_ok=True)

    config = ToMAgentConfig(
        processed_data_dir=processed_data_dir,
        user_model_dir=user_model_dir,
        llm_model="test_model",
    )
    agent = ToMAgent(config=config)
    return agent


@pytest.fixture
def mock_rag_agent() -> AsyncMock:
    """Mock RAG agent for testing."""
    mock_agent = AsyncMock()
    mock_agent.query.return_value = {
        "response": "User prefers detailed step-by-step instructions and code examples. Shows systematic approach to debugging."
    }
    return mock_agent


@pytest.fixture
def mock_llm_instruction_response() -> str:
    """Mock LLM response for instruction improvement."""
    return json.dumps(
        {
            "improved_instruction": "Debug the Python function by first examining the error traceback, then systematically checking each variable and loop condition step-by-step, with detailed logging at each stage.",
            "reasoning": "Based on the user's preference for detailed explanations and systematic approaches, the improved instruction provides a structured debugging methodology with explicit steps.",
            "confidence_score": 0.85,
            "personalization_factors": [
                "detailed_explanations",
                "step_by_step_help",
                "systematic_approach",
            ],
        }
    )


@pytest.fixture
def mock_llm_actions_response() -> str:
    """Mock LLM response for next action suggestions."""
    return json.dumps(
        {
            "suggestions": [
                {
                    "action_description": "Add comprehensive logging statements to identify the exact point of failure",
                    "priority": "high",
                    "reasoning": "Based on the user's systematic approach and preference for detailed information, adding logging will provide the visibility needed for effective debugging.",
                    "expected_outcome": "Clear identification of where the error occurs in the code execution flow",
                    "user_preference_alignment": 0.9,
                },
                {
                    "action_description": "Create a minimal test case that reproduces the issue",
                    "priority": "medium",
                    "reasoning": "The user values systematic approaches and would benefit from isolating the problem in a controlled environment.",
                    "expected_outcome": "Simplified reproduction of the bug for easier analysis",
                    "user_preference_alignment": 0.8,
                },
            ]
        }
    )


class TestProposeInstructions:
    """Test the propose_instructions function."""

    @pytest.mark.asyncio
    async def test_propose_instructions_success(
        self,
        tom_agent: ToMAgent,
        sample_user_context: UserContext,
        mock_rag_agent: AsyncMock,
        mock_llm_instruction_response: str,
    ) -> None:
        """Test successful instruction improvement."""
        with patch.object(tom_agent, "_get_rag_agent", return_value=mock_rag_agent):
            with patch.object(tom_agent, "_call_llm", return_value=mock_llm_instruction_response):
                original_instruction = "Debug the function that's causing errors"
                recommendations = await tom_agent.propose_instructions(
                    user_context=sample_user_context,
                    original_instruction=original_instruction,
                    user_msg_context="Python development",
                )

                assert len(recommendations) == 1
                rec = recommendations[0]
                assert isinstance(rec, InstructionRecommendation)
                assert rec.original_instruction == original_instruction
                assert "step-by-step" in rec.improved_instruction
                assert "systematic" in rec.reasoning
                assert rec.confidence_score == 0.85
                # personalization_factors field was removed, checking clarity_score instead
                assert hasattr(rec, "clarity_score")

                # Verify RAG agent was called with appropriate query
                mock_rag_agent.query.assert_called_once()
                call_args = mock_rag_agent.query.call_args[0]
                assert "Debug the function that's causing errors" in call_args[0]
                assert "Python development" in call_args[0]

    @pytest.mark.asyncio
    async def test_propose_instructions_llm_failure(
        self, tom_agent: ToMAgent, sample_user_context: UserContext, mock_rag_agent: AsyncMock
    ) -> None:
        """Test instruction improvement when LLM fails."""
        with patch.object(tom_agent, "_get_rag_agent", return_value=mock_rag_agent):
            with patch.object(tom_agent, "_call_llm", return_value=None):
                recommendations = await tom_agent.propose_instructions(
                    user_context=sample_user_context,
                    original_instruction="Debug the function",
                    user_msg_context="Python development",
                )

                assert recommendations == []

    @pytest.mark.asyncio
    async def test_propose_instructions_invalid_json(
        self, tom_agent: ToMAgent, sample_user_context: UserContext, mock_rag_agent: AsyncMock
    ) -> None:
        """Test instruction improvement with invalid JSON response."""
        with patch.object(tom_agent, "_get_rag_agent", return_value=mock_rag_agent):
            with patch.object(tom_agent, "_call_llm", return_value="invalid json"):
                recommendations = await tom_agent.propose_instructions(
                    user_context=sample_user_context,
                    original_instruction="Debug the function",
                    user_msg_context="Python development",
                )

                assert recommendations == []

    @pytest.mark.asyncio
    async def test_propose_instructions_no_user_msg_context(
        self,
        tom_agent: ToMAgent,
        sample_user_context: UserContext,
        mock_rag_agent: AsyncMock,
        mock_llm_instruction_response: str,
    ) -> None:
        """Test instruction improvement without user message context."""
        with patch.object(tom_agent, "_get_rag_agent", return_value=mock_rag_agent):
            with patch.object(tom_agent, "_call_llm", return_value=mock_llm_instruction_response):
                recommendations = await tom_agent.propose_instructions(
                    user_context=sample_user_context,
                    original_instruction="Debug the function",
                    user_msg_context=None,
                )

                assert len(recommendations) == 1
                # Verify RAG query doesn't include domain context
                call_args = mock_rag_agent.query.call_args[0]
                assert "Debug the function" in call_args[0]


class TestSuggestNextActions:
    """Test the suggest_next_actions function."""

    @pytest.mark.asyncio
    async def test_suggest_next_actions_success(
        self,
        tom_agent: ToMAgent,
        sample_user_context: UserContext,
        mock_rag_agent: AsyncMock,
        mock_llm_actions_response: str,
    ) -> None:
        """Test successful next action suggestions."""
        with patch.object(tom_agent, "_get_rag_agent", return_value=mock_rag_agent):
            with patch.object(tom_agent, "_call_llm", return_value=mock_llm_actions_response):
                suggestions = await tom_agent.suggest_next_actions(
                    user_context=sample_user_context,
                    current_task_context="Debugging Python application",
                )

                assert len(suggestions) == 2

                # Test first suggestion
                suggestion1 = suggestions[0]
                assert isinstance(suggestion1, NextActionSuggestion)
                assert "logging" in suggestion1.action_description
                assert suggestion1.priority == "high"
                assert "systematic" in suggestion1.reasoning
                assert suggestion1.user_preference_alignment == 0.9

                # Test second suggestion
                suggestion2 = suggestions[1]
                assert "test case" in suggestion2.action_description
                assert suggestion2.priority == "medium"
                assert suggestion2.user_preference_alignment == 0.8

                # Verify RAG agent was called
                mock_rag_agent.query.assert_called_once()
                call_args = mock_rag_agent.query.call_args[0]
                assert "workflow patterns" in call_args[0]
                assert "Debugging Python application" in call_args[0]

    @pytest.mark.asyncio
    async def test_suggest_next_actions_llm_failure(
        self, tom_agent: ToMAgent, sample_user_context: UserContext, mock_rag_agent: AsyncMock
    ) -> None:
        """Test next action suggestions when LLM fails."""
        with patch.object(tom_agent, "_get_rag_agent", return_value=mock_rag_agent):
            with patch.object(tom_agent, "_call_llm", return_value=None):
                suggestions = await tom_agent.suggest_next_actions(
                    user_context=sample_user_context,
                    current_task_context="Debugging Python application",
                )

                assert suggestions == []

    @pytest.mark.asyncio
    async def test_suggest_next_actions_invalid_json(
        self, tom_agent: ToMAgent, sample_user_context: UserContext, mock_rag_agent: AsyncMock
    ) -> None:
        """Test next action suggestions with invalid JSON response."""
        with patch.object(tom_agent, "_get_rag_agent", return_value=mock_rag_agent):
            with patch.object(tom_agent, "_call_llm", return_value="invalid json"):
                suggestions = await tom_agent.suggest_next_actions(
                    user_context=sample_user_context,
                    current_task_context="Debugging Python application",
                )

                assert suggestions == []

    @pytest.mark.asyncio
    async def test_suggest_next_actions_no_task_context(
        self,
        tom_agent: ToMAgent,
        sample_user_context: UserContext,
        mock_rag_agent: AsyncMock,
        mock_llm_actions_response: str,
    ) -> None:
        """Test next action suggestions without task context."""
        with patch.object(tom_agent, "_get_rag_agent", return_value=mock_rag_agent):
            with patch.object(tom_agent, "_call_llm", return_value=mock_llm_actions_response):
                suggestions = await tom_agent.suggest_next_actions(
                    user_context=sample_user_context, current_task_context=None
                )

                assert len(suggestions) == 2
                # Verify RAG query works without task context
                call_args = mock_rag_agent.query.call_args[0]
                assert "workflow patterns" in call_args[0]

    @pytest.mark.asyncio
    async def test_suggest_next_actions_empty_user_context(
        self, tom_agent: ToMAgent, mock_rag_agent: AsyncMock, mock_llm_actions_response: str
    ) -> None:
        """Test next action suggestions with minimal user context."""
        minimal_context = UserContext(
            user_id="test_user_minimal",
            user_profile=None,
            recent_sessions=None,
            current_query=None,
            preferences=None,
            mental_state_summary=None,
        )

        with patch.object(tom_agent, "_get_rag_agent", return_value=mock_rag_agent):
            with patch.object(tom_agent, "_call_llm", return_value=mock_llm_actions_response):
                suggestions = await tom_agent.suggest_next_actions(
                    user_context=minimal_context, current_task_context="General development"
                )

                assert len(suggestions) == 2
                # Should still work with minimal context
                assert all(isinstance(s, NextActionSuggestion) for s in suggestions)

    @pytest.mark.asyncio
    async def test_suggest_next_actions_recent_sessions_analysis(
        self,
        tom_agent: ToMAgent,
        sample_user_context: UserContext,
        mock_rag_agent: AsyncMock,
        mock_llm_actions_response: str,
    ) -> None:
        """Test that recent sessions are properly analyzed for intents and emotions."""
        # Add more sessions to test the analysis
        additional_session = SessionSummary(
            session_id="session_002",
            timestamp=datetime.now(),
            message_count=8,
            intent_distribution={"optimization": 7, "learning": 3},
            emotion_distribution={"curious": 6, "focused": 4},
            key_preferences=["performance_tips"],
            user_modeling_summary="User worked on optimization",
            clarification_requests=0,
            session_start="2024-01-16T10:00:00",
            session_end="2024-01-16T10:30:00",
        )

        # Create a new UserContext with the additional session
        if sample_user_context.recent_sessions is not None:
            updated_sessions = [*sample_user_context.recent_sessions, additional_session]
            sample_user_context = sample_user_context.model_copy(
                update={"recent_sessions": updated_sessions}
            )

        with patch.object(tom_agent, "_get_rag_agent", return_value=mock_rag_agent):
            with patch.object(
                tom_agent, "_call_llm", return_value=mock_llm_actions_response
            ) as mock_llm:
                suggestions = await tom_agent.suggest_next_actions(
                    user_context=sample_user_context, current_task_context="Development work"
                )

                assert len(suggestions) == 2

                # Check that the LLM prompt included recent session analysis
                mock_llm.assert_called_once()
                prompt = mock_llm.call_args[0][0]
                assert "debugging" in prompt  # From first session
                assert "optimization" in prompt  # From second session
                assert "focused" in prompt  # From emotions
                assert "curious" in prompt  # From emotions
