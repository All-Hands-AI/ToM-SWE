#!/usr/bin/env python3
"""
Comprehensive tests for instruction improvement functionality after cleanup.

Tests the core ToM Agent functionality focusing only on instruction improvement,
verifying that next action functionality has been properly removed.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tom_swe.database import (
    InstructionImprovementResponse,
    InstructionRecommendation,
    SessionSummary,
    UserContext,
    UserProfile,
)
from tom_swe.tom_agent import ToMAgent, ToMAgentConfig, create_tom_agent


class TestToMAgentCreation:
    """Test ToM Agent creation and configuration."""

    @pytest.mark.asyncio
    async def test_create_tom_agent_default_config(self) -> None:
        """Test creating ToM agent with default configuration."""
        with patch("tom_swe.tom_agent.UserMentalStateAnalyzer"):
            agent = await create_tom_agent()
            assert isinstance(agent, ToMAgent)
            assert agent.llm_model is not None
            assert agent.enable_rag is True

    @pytest.mark.asyncio
    async def test_create_tom_agent_custom_config(self) -> None:
        """Test creating ToM agent with custom configuration."""
        with patch("tom_swe.tom_agent.UserMentalStateAnalyzer"):
            agent = await create_tom_agent(
                processed_data_dir="./custom_data",
                user_model_dir="./custom_model",
                enable_rag=False,
            )
            assert isinstance(agent, ToMAgent)
            assert agent.processed_data_dir == "./custom_data"
            assert agent.user_model_dir == "./custom_model"
            assert agent.enable_rag is False

    def test_tom_agent_config_creation(self) -> None:
        """Test ToMAgentConfig creation."""
        config = ToMAgentConfig(
            processed_data_dir="./test_data",
            user_model_dir="./test_model",
            llm_model="test-model",
            enable_rag=False,
        )
        assert config.processed_data_dir == "./test_data"
        assert config.user_model_dir == "./test_model"
        assert config.llm_model == "test-model"
        assert config.enable_rag is False


class TestToMAgentMethodsExist:
    """Test that required methods exist and removed methods are gone."""

    @pytest.fixture
    def mock_agent(self) -> ToMAgent:
        """Create a mock ToM agent for testing."""
        with patch("tom_swe.tom_agent.UserMentalStateAnalyzer"):
            config = ToMAgentConfig()
            return ToMAgent(config)

    def test_required_methods_exist(self, mock_agent: ToMAgent) -> None:
        """Test that instruction improvement methods exist."""
        assert hasattr(mock_agent, "analyze_user_context")
        assert hasattr(mock_agent, "propose_instructions")
        assert callable(mock_agent.analyze_user_context)
        assert callable(mock_agent.propose_instructions)

    def test_removed_methods_are_gone(self, mock_agent: ToMAgent) -> None:
        """Test that next action methods have been removed."""
        # These methods should no longer exist
        assert not hasattr(mock_agent, "suggest_next_actions")
        assert not hasattr(mock_agent, "get_personalized_guidance")
        assert not hasattr(mock_agent, "_generate_overall_guidance")


class TestAnalyzeUserContext:
    """Test the analyze_user_context method."""

    @pytest.fixture
    def mock_agent(self) -> ToMAgent:
        """Create a mock ToM agent with mocked dependencies."""
        with patch("tom_swe.tom_agent.UserMentalStateAnalyzer") as mock_analyzer:
            config = ToMAgentConfig()
            agent = ToMAgent(config)

            # Mock the analyzer's load_existing_user_profile method
            mock_profile = MagicMock()
            mock_profile.user_profile = UserProfile(
                user_id="test_user",
                total_sessions=5,
                overall_description="Test user profile",
                intent_distribution={"debugging": 10, "code_generation": 5},
                emotion_distribution={"focused": 8, "frustrated": 2},
                preference_summary=["Prefers detailed explanations", "Likes step-by-step guidance"],
            )
            mock_profile.session_summaries = [
                SessionSummary(
                    session_id="session_1",
                    timestamp=datetime(2024, 1, 1, 10, 0, 0),
                    message_count=5,
                    intent_distribution={"debugging": 3, "code_generation": 2},
                    emotion_distribution={"focused": 4, "frustrated": 1},
                    key_preferences=["detailed_help", "step_by_step"],
                    user_modeling_summary="User worked on debugging Python code",
                    clarification_requests=1,
                    session_start="2024-01-01T10:00:00",
                    session_end="2024-01-01T10:30:00",
                )
            ]

            mock_analyzer.return_value.load_existing_user_profile = AsyncMock(
                return_value=mock_profile
            )
            agent.tom_analyzer = mock_analyzer.return_value

            return agent

    @pytest.mark.asyncio
    async def test_analyze_user_context_success(self, mock_agent: ToMAgent) -> None:
        """Test successful user context analysis."""
        user_context = await mock_agent.analyze_user_context("test_user", "Help me debug this code")

        assert isinstance(user_context, UserContext)
        assert user_context.user_id == "test_user"
        assert user_context.current_query == "Help me debug this code"
        assert user_context.user_profile is not None
        assert user_context.preferences == [
            "Prefers detailed explanations",
            "Likes step-by-step guidance",
        ]
        assert user_context.recent_sessions is not None
        assert len(user_context.recent_sessions) == 1

    @pytest.mark.asyncio
    async def test_analyze_user_context_no_profile(self, mock_agent: ToMAgent) -> None:
        """Test user context analysis when no profile exists."""
        # Mock no profile found
        mock_agent.tom_analyzer.load_existing_user_profile = AsyncMock(return_value=None)  # type: ignore[method-assign]

        user_context = await mock_agent.analyze_user_context("unknown_user", "Help me")

        assert isinstance(user_context, UserContext)
        assert user_context.user_id == "unknown_user"
        assert user_context.user_profile is None
        assert user_context.preferences == []
        assert user_context.recent_sessions == []
        assert user_context.mental_state_summary == ""


class TestProposeInstructions:
    """Test the propose_instructions method."""

    @pytest.fixture
    def mock_agent(self) -> ToMAgent:
        """Create a mock ToM agent with mocked LLM calls."""
        with patch("tom_swe.tom_agent.UserMentalStateAnalyzer"):
            config = ToMAgentConfig()
            agent = ToMAgent(config)
            return agent

    @pytest.fixture
    def sample_user_context(self) -> UserContext:
        """Create a sample user context for testing."""
        return UserContext(
            user_id="test_user",
            user_profile=UserProfile(
                user_id="test_user",
                total_sessions=3,
                overall_description="Experienced developer who prefers detailed explanations",
                intent_distribution={"debugging": 15, "code_generation": 8, "optimization": 2},
                emotion_distribution={"focused": 12, "confident": 8, "curious": 5},
                preference_summary=["Likes step-by-step guidance", "Prefers code examples"],
            ),
            current_query="Help me fix this bug",
            preferences=["Likes step-by-step guidance", "Prefers code examples"],
            mental_state_summary="Focused and methodical developer",
        )

    @pytest.mark.asyncio
    async def test_propose_instructions_success(
        self, mock_agent: ToMAgent, sample_user_context: UserContext
    ) -> None:
        """Test successful instruction proposal."""
        # Mock RAG behavior
        with patch.object(
            mock_agent, "_get_relevant_behavior", return_value="Relevant user behavior context"
        ):
            # Mock LLM call
            mock_llm_response = InstructionImprovementResponse(
                improved_instruction="Here's a detailed step-by-step approach to debug your code:\n1. Check error logs\n2. Examine stack trace\n3. Add debugging prints",
                reasoning="User prefers detailed explanations and step-by-step guidance",
                confidence_score=0.85,
                clarity_score=0.75,
            )

            with patch.object(mock_agent, "_call_llm_structured", return_value=mock_llm_response):
                recommendations = await mock_agent.propose_instructions(
                    sample_user_context, "Fix this bug", "Working on a Python application"
                )

                assert len(recommendations) == 1
                rec = recommendations[0]
                assert isinstance(rec, InstructionRecommendation)
                assert rec.original_instruction == "Fix this bug"
                assert "step-by-step" in rec.improved_instruction
                assert rec.confidence_score == 0.85
                assert rec.clarity_score == 0.75

    @pytest.mark.asyncio
    async def test_propose_instructions_llm_failure(
        self, mock_agent: ToMAgent, sample_user_context: UserContext
    ) -> None:
        """Test instruction proposal when LLM call fails."""
        with patch.object(mock_agent, "_get_relevant_behavior", return_value="Context"):
            with patch.object(mock_agent, "_call_llm_structured", return_value=None):
                recommendations = await mock_agent.propose_instructions(
                    sample_user_context, "Fix this bug", "Working on a Python application"
                )

                assert recommendations == []

    @pytest.mark.asyncio
    async def test_propose_instructions_with_rag_disabled(
        self, mock_agent: ToMAgent, sample_user_context: UserContext
    ) -> None:
        """Test instruction proposal with RAG disabled."""
        mock_agent.enable_rag = False

        mock_llm_response = InstructionImprovementResponse(
            improved_instruction="Simple fix approach",
            reasoning="Basic reasoning without RAG context",
            confidence_score=0.70,
            clarity_score=0.80,
        )

        with patch.object(mock_agent, "_call_llm_structured", return_value=mock_llm_response):
            recommendations = await mock_agent.propose_instructions(
                sample_user_context, "Fix this bug", "Working on a Python application"
            )

            assert len(recommendations) == 1
            rec = recommendations[0]
            assert rec.confidence_score == 0.70


class TestRAGIntegration:
    """Test RAG integration with instruction improvement."""

    @pytest.fixture
    def mock_agent_with_rag(self) -> ToMAgent:
        """Create a mock ToM agent with mocked RAG."""
        with patch("tom_swe.tom_agent.UserMentalStateAnalyzer"):
            config = ToMAgentConfig(enable_rag=True)
            agent = ToMAgent(config)

            # Mock RAG agent
            mock_rag_agent = MagicMock()
            mock_rag_agent.retrieve.return_value = [
                MagicMock(
                    content="User prefers detailed debugging steps",
                    metadata={"session_title": "Debug Session"},
                ),
                MagicMock(
                    content="User likes code examples with explanations",
                    metadata={"repository_context": "Python Project"},
                ),
            ]

            agent._rag_agent = mock_rag_agent
            return agent

    @pytest.mark.asyncio
    async def test_get_relevant_behavior_with_rag(self, mock_agent_with_rag: ToMAgent) -> None:
        """Test RAG behavior retrieval."""
        with patch.object(
            mock_agent_with_rag, "_get_rag_agent", return_value=mock_agent_with_rag._rag_agent
        ):
            behavior = await mock_agent_with_rag._get_relevant_behavior("Fix this bug")

            assert "User prefers detailed debugging steps" in behavior
            assert "User likes code examples" in behavior
            # Use getattr to safely access the mock attribute
            mock_rag_agent = getattr(mock_agent_with_rag, "_rag_agent", None)
            assert mock_rag_agent is not None
            mock_rag_agent.retrieve.assert_called_once_with("Fix this bug", k=5)  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_get_relevant_behavior_without_rag(self) -> None:
        """Test behavior when RAG is disabled."""
        with patch("tom_swe.tom_agent.UserMentalStateAnalyzer"):
            config = ToMAgentConfig(enable_rag=False)
            agent = ToMAgent(config)

            behavior = await agent._get_relevant_behavior("Fix this bug")
            assert behavior == "RAG disabled - using user context only"


class TestInstructionFormatting:
    """Test instruction formatting and output."""

    @pytest.fixture
    def mock_agent(self) -> ToMAgent:
        """Create a mock ToM agent."""
        with patch("tom_swe.tom_agent.UserMentalStateAnalyzer"):
            config = ToMAgentConfig()
            return ToMAgent(config)

    def test_format_proposed_instruction(self, mock_agent: ToMAgent) -> None:
        """Test instruction formatting with ToM agent analysis."""
        from tom_swe.tom_agent import InstructionScores

        scores = InstructionScores(confidence_score=0.85, clarity_score=0.75)

        formatted = mock_agent._format_proposed_instruction(
            original_instruction="Fix the bug",
            improved_instruction="Here's a detailed debugging approach with steps",
            reasoning="User prefers detailed step-by-step guidance",
            scores=scores,
        )

        assert "Fix the bug" in formatted
        assert "85%" in formatted  # confidence score
        assert "75%" in formatted  # clarity score
        assert "detailed debugging approach" in formatted
        assert "ToM Agent Analysis" in formatted


class TestDatabaseModels:
    """Test database models used in instruction improvement."""

    def test_instruction_recommendation_model(self) -> None:
        """Test InstructionRecommendation model validation."""
        rec = InstructionRecommendation(
            original_instruction="Fix this",
            improved_instruction="Here's how to fix it step by step",
            reasoning="User prefers detailed guidance",
            confidence_score=0.85,
            clarity_score=0.75,
        )

        assert rec.original_instruction == "Fix this"
        assert rec.confidence_score == 0.85
        assert rec.clarity_score == 0.75

    def test_instruction_recommendation_validation_error(self) -> None:
        """Test InstructionRecommendation validation with invalid scores."""
        with pytest.raises(ValueError):
            InstructionRecommendation(
                original_instruction="Fix this",
                improved_instruction="Here's how to fix it",
                reasoning="User prefers detailed guidance",
                confidence_score=1.5,  # Invalid: > 1.0
                clarity_score=0.75,
            )

    def test_user_context_model(self) -> None:
        """Test UserContext model."""
        context = UserContext(
            user_id="test_user",
            current_query="Help me debug",
            preferences=["Detailed explanations", "Code examples"],
            mental_state_summary="Focused developer",
        )

        assert context.user_id == "test_user"
        assert context.preferences is not None
        assert len(context.preferences) == 2
        assert context.mental_state_summary == "Focused developer"


class TestErrorHandling:
    """Test error handling in instruction improvement."""

    @pytest.fixture
    def mock_agent(self) -> ToMAgent:
        """Create a mock ToM agent."""
        with patch("tom_swe.tom_agent.UserMentalStateAnalyzer"):
            config = ToMAgentConfig()
            return ToMAgent(config)

    @pytest.mark.asyncio
    async def test_llm_call_exception_handling(self, mock_agent: ToMAgent) -> None:
        """Test handling of LLM call exceptions."""
        sample_context = UserContext(user_id="test_user")

        with patch.object(mock_agent, "_get_relevant_behavior", return_value="Context"):
            # Mock LLM call to return None (simulating handled exception)
            with patch.object(mock_agent, "_call_llm_structured", return_value=None):
                # Should not raise exception, should return empty list
                recommendations = await mock_agent.propose_instructions(
                    sample_context, "Fix this bug", "Context"
                )

                assert recommendations == []

    @pytest.mark.asyncio
    async def test_rag_initialization_failure(self, mock_agent: ToMAgent) -> None:
        """Test handling of RAG initialization failure."""
        mock_agent.enable_rag = True

        # Mock _get_rag_agent to raise exception
        with patch.object(mock_agent, "_get_rag_agent", side_effect=Exception("RAG Init Error")):
            # Since _get_relevant_behavior doesn't have explicit exception handling,
            # we expect the exception to propagate. This test verifies the exception occurs.
            with pytest.raises(Exception, match="RAG Init Error"):
                await mock_agent._get_relevant_behavior("test query")

    @pytest.mark.asyncio
    async def test_propose_instructions_with_rag_failure(self, mock_agent: ToMAgent) -> None:
        """Test propose_instructions when RAG fails but LLM succeeds."""
        sample_context = UserContext(user_id="test_user")
        mock_agent.enable_rag = True

        # Mock RAG to fail, but provide a fallback behavior
        with patch.object(
            mock_agent, "_get_relevant_behavior", return_value="RAG failed - using fallback"
        ):
            # Mock successful LLM response
            mock_llm_response = InstructionImprovementResponse(
                improved_instruction="Fallback instruction without RAG context",
                reasoning="Used fallback due to RAG failure",
                confidence_score=0.6,  # Lower confidence due to missing RAG
                clarity_score=0.8,
            )

            with patch.object(mock_agent, "_call_llm_structured", return_value=mock_llm_response):
                recommendations = await mock_agent.propose_instructions(
                    sample_context, "Fix this bug", "Context"
                )

                assert len(recommendations) == 1
                assert "fallback" in recommendations[0].reasoning.lower()
                assert recommendations[0].confidence_score == 0.6


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
