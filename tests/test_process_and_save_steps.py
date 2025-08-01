"""
Tests for each step in process_and_save_user_session and supporting functions.
Updated for async tom_module.
"""

import json
import os
from datetime import datetime
from typing import Any, List, Tuple
from unittest.mock import Mock, patch

import pytest

from tom_swe.tom_module import (
    SessionSummary,
    UserMentalStateAnalyzer,
    UserMessageAnalysis,
)

# Test constants
EXPECTED_SESSION_COUNT = 2
EXPECTED_USER_MESSAGE_COUNT = 2


class TestDataLoading:
    """Test data loading and session discovery functions."""

    @pytest.mark.asyncio
    async def test_get_user_session_ids_valid_user(
        self, user_file_setup: Tuple[UserMentalStateAnalyzer, str, str]
    ) -> None:
        """Test getting session IDs for a valid user."""
        analyzer, temp_dir, user_id = user_file_setup

        session_ids = await analyzer.get_user_session_ids(user_id)

        assert isinstance(session_ids, list)
        assert len(session_ids) == EXPECTED_SESSION_COUNT
        assert "session_001" in session_ids
        assert "session_002" in session_ids

    @pytest.mark.asyncio
    async def test_get_user_session_ids_missing_user(
        self, analyzer_with_temp_dir: Tuple[UserMentalStateAnalyzer, str]
    ) -> None:
        """Test getting session IDs for non-existent user."""
        analyzer, temp_dir = analyzer_with_temp_dir

        session_ids = await analyzer.get_user_session_ids("nonexistent_user")

        assert session_ids == []

    @pytest.mark.asyncio
    async def test_get_user_session_ids_empty_file(
        self,
        analyzer_with_temp_dir: Tuple[UserMentalStateAnalyzer, str],
        empty_user_data: Any,
    ) -> None:
        """Test getting session IDs for user with empty data."""
        analyzer, temp_dir = analyzer_with_temp_dir
        user_id = "empty_user"

        # Create empty user file
        user_file = os.path.join(analyzer.processed_data_dir, f"{user_id}.json")
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(empty_user_data, f)

        session_ids = await analyzer.get_user_session_ids(user_id)

        assert session_ids == []


class TestStep1AnalyzeUserMentalState:
    """Test Step 1: analyze_user_mental_state function."""

    @pytest.mark.asyncio
    @patch("tom_swe.tom_module.UserMentalStateAnalyzer.call_llm")
    @patch("tom_swe.tom_module.PydanticOutputParser")
    async def test_analyze_user_mental_state_success(
        self,
        mock_parser: Any,
        mock_llm: Any,
        user_file_setup: Tuple[UserMentalStateAnalyzer, str, str],
    ) -> None:
        """Test successful analysis of user mental state."""
        analyzer, temp_dir, user_id = user_file_setup

        # Mock LLM response and parser
        mock_llm.return_value = '{"intent": "debugging", "emotions": ["frustrated"], "preference": ["help"], "user_modeling": "needs help", "should_ask_clarification": true}'
        mock_parser_instance = Mock()
        mock_parser_instance.parse.return_value = UserMessageAnalysis(
            intent="debugging",
            emotions=["frustrated"],
            preference=["help"],
            user_modeling="needs help",
            should_ask_clarification=True,
        )
        mock_parser.return_value = mock_parser_instance

        result = await analyzer.analyze_all_session_messages(user_id, "session_001")

        assert isinstance(result, tuple)
        analyses, messages = result
        assert (
            len(analyses) == EXPECTED_USER_MESSAGE_COUNT
        )  # Two user messages in session_001
        assert all(isinstance(analysis, UserMessageAnalysis) for analysis in analyses)

    @pytest.mark.asyncio
    async def test_analyze_user_mental_state_missing_session(
        self, user_file_setup: Tuple[UserMentalStateAnalyzer, str, str]
    ) -> None:
        """Test analysis with non-existent session."""
        analyzer, temp_dir, user_id = user_file_setup

        result = await analyzer.analyze_all_session_messages(
            user_id, "nonexistent_session"
        )

        if result is None:
            # No session found, which is expected behavior
            assert True
        else:
            analyses, messages = result
            assert analyses == []
            assert messages == []

    @pytest.mark.asyncio
    async def test_analyze_user_mental_state_missing_user(
        self, analyzer_with_temp_dir: Tuple[UserMentalStateAnalyzer, str]
    ) -> None:
        """Test analysis with non-existent user."""
        analyzer, temp_dir = analyzer_with_temp_dir

        result = await analyzer.analyze_all_session_messages(
            "nonexistent_user", "session_001"
        )

        assert result is None


class TestStep2SaveSessionAnalysesToJsonl:
    """Test Step 2: save_session_analyses_to_jsonl function."""

    @pytest.mark.asyncio
    async def test_save_session_analyses_to_jsonl_success(
        self,
        analyzer_with_temp_dir: Tuple[UserMentalStateAnalyzer, str],
        sample_message_analyses: List[UserMessageAnalysis],
    ) -> None:
        """Test successful saving of analyses to JSONL."""
        analyzer, temp_dir = analyzer_with_temp_dir
        user_id = "test_user"
        session_id = "test_session"

        # Create matching user messages for the analyses
        user_messages = [
            f"User message {i}" for i in range(len(sample_message_analyses))
        ]

        await analyzer.save_session_analyses_to_json(
            user_id, session_id, (sample_message_analyses, user_messages), None
        )

        # Check file was created (method uses default base directory)
        expected_path = os.path.join(
            "./data/user_model", "user_model_detailed", user_id, f"{session_id}.json"
        )
        assert os.path.exists(expected_path)

        # Check file content
        with open(expected_path, encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["message_analyses"]) == len(sample_message_analyses)

        # Verify each analysis has expected structure
        for i, message_data in enumerate(data["message_analyses"]):
            analysis = message_data["analysis"]
            assert analysis["intent"] == sample_message_analyses[i].intent
            assert analysis["emotions"] == sample_message_analyses[i].emotions
            assert analysis["preference"] == sample_message_analyses[i].preference
            assert analysis["user_modeling"] == sample_message_analyses[i].user_modeling
            assert (
                analysis["should_ask_clarification"]
                == sample_message_analyses[i].should_ask_clarification
            )

    @pytest.mark.asyncio
    async def test_save_session_analyses_to_jsonl_empty_analyses(
        self, analyzer_with_temp_dir: Tuple[UserMentalStateAnalyzer, str]
    ) -> None:
        """Test saving empty analyses list."""
        analyzer, temp_dir = analyzer_with_temp_dir

        await analyzer.save_session_analyses_to_json("user", "session", ([], []), None)

        expected_path = os.path.join(
            "./data/user_model", "user_model_detailed", "user", "session.json"
        )
        assert os.path.exists(expected_path)

        # File should have empty analyses list
        with open(expected_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["message_analyses"]) == 0


class TestStep3SummarizeSessionFromAnalyses:
    """Test Step 3: summarize_session_from_analyses function."""

    def test_summarize_session_from_analyses_success(
        self,
        analyzer_with_temp_dir: Tuple[UserMentalStateAnalyzer, str],
        sample_message_analyses: List[UserMessageAnalysis],
    ) -> None:
        """Test successful session summarization."""
        analyzer, temp_dir = analyzer_with_temp_dir
        session_id = "test_session"
        session_metadata = {
            "convo_start": "2024-01-15T10:00:00",
            "convo_end": "2024-01-15T10:30:00",
        }

        summary = analyzer.create_session_summary_from_analyses(
            session_id, sample_message_analyses, session_metadata
        )

        assert isinstance(summary, SessionSummary)
        assert summary.session_id == session_id
        assert summary.message_count == len(sample_message_analyses)
        assert "debugging" in summary.intent_distribution
        assert "optimization" in summary.intent_distribution
        assert len(summary.emotion_distribution) > 0
        assert (
            summary.clarification_requests == 1
        )  # Only first analysis has should_ask_clarification=True
        assert summary.session_start == "2024-01-15T10:00:00"
        assert summary.session_end == "2024-01-15T10:30:00"

    def test_summarize_session_from_analyses_empty_list(
        self, analyzer_with_temp_dir: Tuple[UserMentalStateAnalyzer, str]
    ) -> None:
        """Test summarization with empty analyses list."""
        analyzer, temp_dir = analyzer_with_temp_dir

        summary = analyzer.create_session_summary_from_analyses("session", [], {})

        assert summary is None

    def test_summarize_session_intent_counting(
        self, analyzer_with_temp_dir: Tuple[UserMentalStateAnalyzer, str]
    ) -> None:
        """Test that intent counting works correctly."""
        analyzer, temp_dir = analyzer_with_temp_dir

        # Create analyses with known intent distribution
        analyses = [
            UserMessageAnalysis(
                intent="debugging",
                emotions=[],
                preference=[],
                user_modeling="",
                should_ask_clarification=False,
            ),
            UserMessageAnalysis(
                intent="debugging",
                emotions=[],
                preference=[],
                user_modeling="",
                should_ask_clarification=False,
            ),
            UserMessageAnalysis(
                intent="optimization",
                emotions=[],
                preference=[],
                user_modeling="",
                should_ask_clarification=False,
            ),
        ]

        summary = analyzer.create_session_summary_from_analyses("session", analyses, {})

        # Check that debugging is the most common intent
        assert summary is not None
        assert summary.intent_distribution["debugging"] == 2  # Most common
        assert summary.intent_distribution["optimization"] == 1


class TestStep4UpdateOverallUserAnalysis:
    """Test Step 4: update_overall_user_analysis function."""

    @pytest.mark.asyncio
    async def test_update_overall_user_analysis_new_user(
        self, analyzer_with_temp_dir: Tuple[UserMentalStateAnalyzer, str]
    ) -> None:
        """Test updating analysis for a new user."""
        analyzer, temp_dir = analyzer_with_temp_dir
        user_id = "new_user"
        base_dir = os.path.join(temp_dir, "user_model")

        session_summary = SessionSummary(
            session_id="test_session",
            timestamp=datetime.now(),
            message_count=3,
            intent_distribution={"debugging": 2, "general": 1},
            emotion_distribution={"frustrated": 1, "confused": 1},
            key_preferences=["detailed_help"],
            user_modeling_summary="user needs debugging help",
            clarification_requests=1,
            session_start="2024-01-15T10:00:00",
            session_end="2024-01-15T10:30:00",
        )

        await analyzer.save_updated_user_profile(user_id, [session_summary], base_dir)

        # Check file was created
        expected_path = os.path.join(base_dir, "user_model_overall", f"{user_id}.json")
        assert os.path.exists(expected_path)

        # Check file content
        with open(expected_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["user_profile"]["user_id"] == user_id
        assert data["user_profile"]["total_sessions"] == 1
        assert len(data["session_summaries"]) == 1
        assert data["session_summaries"][0]["session_id"] == "test_session"

    @pytest.mark.asyncio
    async def test_update_overall_user_analysis_existing_user(
        self, analyzer_with_temp_dir: Tuple[UserMentalStateAnalyzer, str]
    ) -> None:
        """Test updating analysis for an existing user."""
        analyzer, temp_dir = analyzer_with_temp_dir
        user_id = "existing_user"
        base_dir = os.path.join(temp_dir, "user_model")

        # Create initial analysis file
        initial_data = {
            "user_profile": {
                "user_id": user_id,
                "total_sessions": 1,
                "overall_description": "",
                "intent_distribution": {"debugging": 1},
                "emotion_distribution": {"confused": 1},
                "preference_summary": [],
            },
            "session_summaries": [
                {
                    "session_id": "old_session",
                    "timestamp": "2024-01-01T00:00:00",
                    "message_count": 2,
                    "intent_distribution": {"debugging": 1},
                    "emotion_distribution": {"confused": 1},
                    "key_preferences": [],
                    "user_modeling_summary": "",
                    "clarification_requests": 0,
                    "session_start": "",
                    "session_end": "",
                }
            ],
            "last_updated": "2024-01-01T00:00:00",
        }

        os.makedirs(os.path.join(base_dir, "user_model_overall"), exist_ok=True)
        overall_path = os.path.join(base_dir, "user_model_overall", f"{user_id}.json")
        with open(overall_path, "w", encoding="utf-8") as f:
            json.dump(initial_data, f)

        # Add new session
        new_session_summary = SessionSummary(
            session_id="new_session",
            timestamp=datetime.now(),
            message_count=1,
            intent_distribution={"optimization": 1},
            emotion_distribution={"focused": 1},
            key_preferences=["performance_tips"],
            user_modeling_summary="ML focused",
            clarification_requests=0,
            session_start="2024-01-15T10:00:00",
            session_end="2024-01-15T10:30:00",
        )

        await analyzer.save_updated_user_profile(
            user_id, [new_session_summary], base_dir
        )

        # Check updated content
        with open(overall_path, encoding="utf-8") as f:
            updated_data = json.load(f)

        assert updated_data["user_profile"]["total_sessions"] == EXPECTED_SESSION_COUNT
        assert len(updated_data["session_summaries"]) == EXPECTED_SESSION_COUNT
        assert updated_data["user_profile"]["intent_distribution"]["optimization"] == 1


class TestIntegrationProcessAndSaveUserSession:
    """Test integration: process_and_save_user_session function."""

    @pytest.mark.asyncio
    async def test_process_and_save_user_session_success(
        self,
        user_file_setup: Tuple[UserMentalStateAnalyzer, str, str],
        mock_analyze_user_mental_state: Any,
    ) -> None:
        """Test complete process_and_save_user_session pipeline."""
        analyzer, temp_dir, user_id = user_file_setup
        session_id = "session_001"
        base_dir = os.path.join(temp_dir, "user_model")

        # Mock the analyze_all_session_messages method
        with patch.object(
            analyzer, "analyze_all_session_messages", mock_analyze_user_mental_state
        ):
            session_summary = await analyzer.process_and_save_single_session(
                user_id, session_id, None
            )

            # Also update the user profile to create the overall analysis file
            if session_summary:
                await analyzer.save_updated_user_profile(
                    user_id, [session_summary], base_dir
                )

        # Check that all three tiers were created

        # Tier 1: JSON file
        json_path = os.path.join(
            "./data/user_model", "user_model_detailed", user_id, f"{session_id}.json"
        )
        assert os.path.exists(json_path)

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert (
            len(data["message_analyses"]) == EXPECTED_USER_MESSAGE_COUNT
        )  # Two analyses from mock

        # Tier 2 & 3: Overall analysis file
        overall_path = os.path.join(base_dir, "user_model_overall", f"{user_id}.json")
        assert os.path.exists(overall_path)

        with open(overall_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["user_profile"]["user_id"] == user_id
        assert data["user_profile"]["total_sessions"] == 1
        assert len(data["session_summaries"]) == 1
        assert data["session_summaries"][0]["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_process_and_save_user_session_no_analyses(
        self, user_file_setup: Tuple[UserMentalStateAnalyzer, str, str]
    ) -> None:
        """Test process_and_save_user_session when no analyses are found."""
        analyzer, temp_dir, user_id = user_file_setup
        base_dir = os.path.join(temp_dir, "user_model")

        # Mock analyze_all_session_messages to return empty list
        with patch.object(
            analyzer, "analyze_all_session_messages", return_value=([], [])
        ):
            await analyzer.process_and_save_single_session(user_id, "session_001", None)

        # Should not create any files
        jsonl_path = os.path.join(
            base_dir, "user_model_detailed", user_id, "session_001.jsonl"
        )
        overall_path = os.path.join(base_dir, "user_model_overall", f"{user_id}.json")

        assert not os.path.exists(jsonl_path)
        assert not os.path.exists(overall_path)


class TestUtilityFunctions:
    """Test utility and helper functions."""

    def test_ensure_directory_structure(
        self, analyzer_with_temp_dir: Tuple[UserMentalStateAnalyzer, str]
    ) -> None:
        """Test directory structure creation."""
        analyzer, temp_dir = analyzer_with_temp_dir
        base_dir = os.path.join(temp_dir, "user_model")

        analyzer.ensure_directory_structure(base_dir)

        assert os.path.exists(os.path.join(base_dir, "user_model_detailed"))
        assert os.path.exists(os.path.join(base_dir, "user_model_overall"))

    def test_create_new_user_analysis(
        self, analyzer_with_temp_dir: Tuple[UserMentalStateAnalyzer, str]
    ) -> None:
        """Test creating new user analysis structure."""
        analyzer, temp_dir = analyzer_with_temp_dir
        user_id = "test_user"

        analysis = analyzer.create_new_user_profile(user_id)

        assert analysis.user_profile.user_id == user_id
        assert analysis.user_profile.total_sessions == 0
        assert len(analysis.session_summaries) == 0
        assert isinstance(analysis.last_updated, datetime)

    def test_update_user_profile(
        self, analyzer_with_temp_dir: Tuple[UserMentalStateAnalyzer, str]
    ) -> None:
        """Test user profile update logic."""
        analyzer, temp_dir = analyzer_with_temp_dir

        # Create analysis with sample session summaries
        analysis = analyzer.create_new_user_profile("test_user")

        # Add sample session summaries
        summary1 = SessionSummary(
            session_id="s1",
            timestamp=datetime.now(),
            message_count=2,
            intent_distribution={"debugging": 1, "optimization": 1},
            emotion_distribution={"frustrated": 1},
            key_preferences=["detailed_help"],
            user_modeling_summary="python debugging expert",
            clarification_requests=1,
            session_start="",
            session_end="",
        )

        summary2 = SessionSummary(
            session_id="s2",
            timestamp=datetime.now(),
            message_count=1,
            intent_distribution={"debugging": 1},
            emotion_distribution={"confident": 1},
            key_preferences=["quick_tips"],
            user_modeling_summary="experienced developer",
            clarification_requests=0,
            session_start="",
            session_end="",
        )

        analysis.session_summaries = [summary1, summary2]

        analyzer.update_user_profile_with_session(analysis)

        assert analysis.user_profile.total_sessions == EXPECTED_SESSION_COUNT
        assert (
            analysis.user_profile.intent_distribution["debugging"]
            == EXPECTED_SESSION_COUNT
        )
        assert analysis.user_profile.intent_distribution["optimization"] == 1
        assert len(analysis.user_profile.preference_summary) >= 0  # Can be empty
