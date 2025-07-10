"""
Tests for the ToM Agent REST API.

This module contains comprehensive tests for all API endpoints,
including success cases, error handling, and edge cases.
"""

import tempfile
from datetime import datetime
from typing import Any, Dict, Iterator, Optional
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from tom_swe.database import (
    InstructionRecommendation,
    NextActionSuggestion,
    PersonalizedGuidance,
    SessionSummary,
    UserContext,
    UserProfile,
)
from tom_swe.tom_agent import ToMAgent


@pytest.fixture
def temp_dir() -> Iterator[str]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_tom_agent() -> AsyncMock:
    """Create a mock ToM agent for testing."""
    mock_agent = AsyncMock(spec=ToMAgent)

    # Mock user context
    user_profile = UserProfile(
        user_id="test_user_123",
        total_sessions=5,
        overall_description="A focused developer who prefers detailed guidance.",
        intent_distribution={"debugging": 15, "optimization": 10},
        emotion_distribution={"focused": 12, "frustrated": 8},
        preference_summary=["detailed_explanations", "step_by_step_help"],
    )

    session_summary = SessionSummary(
        session_id="session_001",
        timestamp=datetime.now(),
        message_count=10,
        intent_distribution={"debugging": 6, "optimization": 4},
        emotion_distribution={"focused": 5, "frustrated": 3},
        key_preferences=["detailed_help"],
        user_modeling_summary="User worked on debugging",
        clarification_requests=1,
        session_start="2024-01-15T10:00:00",
        session_end="2024-01-15T10:30:00",
    )

    user_context = UserContext(
        user_id="test_user_123",
        user_profile=user_profile,
        recent_sessions=[session_summary],
        current_query="Working on debugging",
        preferences=["detailed_explanations"],
        mental_state_summary="A focused developer who prefers detailed guidance.",
    )

    # Mock analyze_user_context
    mock_agent.analyze_user_context.return_value = user_context

    # Mock suggest_next_actions
    next_actions = [
        NextActionSuggestion(
            action_description="Add logging statements to identify the error",
            priority="high",
            reasoning="Based on user's systematic approach",
            expected_outcome="Clear identification of the error location",
            user_preference_alignment=0.9,
        ),
        NextActionSuggestion(
            action_description="Create a minimal test case",
            priority="medium",
            reasoning="User values systematic debugging",
            expected_outcome="Isolated reproduction of the issue",
            user_preference_alignment=0.8,
        ),
    ]
    mock_agent.suggest_next_actions.return_value = next_actions

    # Mock propose_instructions
    instruction_recommendations = [
        InstructionRecommendation(
            original_instruction="Debug the function",
            improved_instruction="Debug the function by systematically adding logging statements and checking each variable step-by-step",
            reasoning="Personalized for user's preference for detailed, systematic approaches",
            confidence_score=0.85,
        )
    ]
    mock_agent.propose_instructions.return_value = instruction_recommendations

    # Mock get_personalized_guidance
    personalized_guidance = PersonalizedGuidance(
        user_context=user_context,
        instruction_recommendations=instruction_recommendations,
        next_action_suggestions=next_actions,
        overall_guidance="Based on your systematic approach, I've provided detailed debugging steps and next actions.",
        confidence_score=0.85,
    )
    mock_agent.get_personalized_guidance.return_value = personalized_guidance

    return mock_agent


def create_test_app(tom_agent: Optional[ToMAgent] = None) -> FastAPI:
    """Create a test FastAPI app without lifespan."""
    from tom_swe.api.main import (
        get_conversation_status,
        global_exception_handler,
        health_check,
        propose_instructions,
        suggest_next_actions,
    )

    test_app = FastAPI(
        title="ToM Agent API (Test)",
        description="Test version of ToM Agent API",
        version="1.0.0",
    )

    # Add CORS middleware
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add exception handler
    test_app.add_exception_handler(Exception, global_exception_handler)

    # Add routes
    test_app.get("/health")(health_check)
    test_app.post("/propose_instructions")(propose_instructions)
    test_app.post("/suggest_next_actions")(suggest_next_actions)
    test_app.get("/conversation_status")(get_conversation_status)

    return test_app


@pytest.fixture
def client_with_mock_agent(mock_tom_agent: AsyncMock) -> Iterator[TestClient]:
    """Create a test client with a mocked ToM agent."""
    test_app = create_test_app()
    test_app.state.tom_agent = mock_tom_agent
    yield TestClient(test_app)


@pytest.fixture
def client_without_agent() -> Iterator[TestClient]:
    """Create a test client without a ToM agent (simulating initialization failure)."""
    test_app = create_test_app()
    test_app.state.tom_agent = None
    yield TestClient(test_app)


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_with_agent(self, client_with_mock_agent: TestClient) -> None:
        """Test health endpoint when ToM agent is available."""
        response = client_with_mock_agent.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["tom_agent_ready"] is True

    def test_health_without_agent(self, client_without_agent: TestClient) -> None:
        """Test health endpoint when ToM agent is not available."""
        response = client_without_agent.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["tom_agent_ready"] is False


class TestSuggestNextActionsEndpoint:
    """Test the suggest_next_actions endpoint."""

    def test_suggest_next_actions_success(
        self, client_with_mock_agent: TestClient, mock_tom_agent: AsyncMock
    ) -> None:
        """Test successful next actions suggestion."""
        request_data = {
            "user_id": "test_user_123",
            "current_task_context": "Debugging Python application",
        }

        response = client_with_mock_agent.post("/suggest_next_actions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "test_user_123"
        assert len(data["suggestions"]) == 2

        # Check first suggestion
        suggestion1 = data["suggestions"][0]
        assert suggestion1["action_description"] == "Add logging statements to identify the error"
        assert suggestion1["priority"] == "high"
        assert suggestion1["user_preference_alignment"] == 0.9

        # Verify mock was called correctly
        mock_tom_agent.analyze_user_context.assert_called_once_with(
            user_id="test_user_123", current_query="Debugging Python application"
        )
        mock_tom_agent.suggest_next_actions.assert_called_once()

    def test_suggest_next_actions_without_task_context(
        self, client_with_mock_agent: TestClient
    ) -> None:
        """Test next actions suggestion without task context."""
        request_data = {"user_id": "test_user_123"}

        response = client_with_mock_agent.post("/suggest_next_actions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "test_user_123"

    def test_suggest_next_actions_without_agent(self, client_without_agent: TestClient) -> None:
        """Test next actions suggestion when ToM agent is not available."""
        request_data = {"user_id": "test_user_123", "current_task_context": "Debugging"}

        response = client_without_agent.post("/suggest_next_actions", json=request_data)
        assert response.status_code == 503
        assert "ToM Agent is not available" in response.json()["detail"]

    def test_suggest_next_actions_invalid_request(self, client_with_mock_agent: TestClient) -> None:
        """Test next actions suggestion with invalid request data."""
        request_data: Dict[str, Any] = {}  # Missing required user_id

        response = client_with_mock_agent.post("/suggest_next_actions", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_suggest_next_actions_agent_error(
        self, client_with_mock_agent: TestClient, mock_tom_agent: AsyncMock
    ) -> None:
        """Test next actions suggestion when ToM agent raises an error."""
        mock_tom_agent.analyze_user_context.side_effect = Exception("Agent error")

        request_data = {"user_id": "test_user_123", "current_task_context": "Debugging"}

        response = client_with_mock_agent.post("/suggest_next_actions", json=request_data)
        assert response.status_code == 500
        assert "Failed to generate next actions" in response.json()["detail"]


class TestProposeInstructionsEndpoint:
    """Test the propose_instructions endpoint."""

    def test_propose_instructions_success(
        self, client_with_mock_agent: TestClient, mock_tom_agent: AsyncMock
    ) -> None:
        """Test successful instruction proposal."""
        request_data = {
            "user_id": "test_user_123",
            "original_instruction": "Debug the function",
            "context": "User message: I need help debugging my Python application. Previous conversation includes discussion about web development and systematic debugging approaches.",
        }

        response = client_with_mock_agent.post("/propose_instructions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "test_user_123"
        assert data["original_instruction"] == "Debug the function"
        assert len(data["recommendations"]) == 1

        # Check recommendation
        recommendation = data["recommendations"][0]
        assert recommendation["original_instruction"] == "Debug the function"
        assert "systematically" in recommendation["improved_instruction"]
        assert recommendation["confidence_score"] == 0.85

        # Verify mock was called correctly
        mock_tom_agent.analyze_user_context.assert_called_once_with(
            user_id="test_user_123", current_query=request_data["context"]
        )
        mock_tom_agent.propose_instructions.assert_called_once()

    def test_propose_instructions_minimal_context(self, client_with_mock_agent: TestClient) -> None:
        """Test instruction proposal with minimal context."""
        request_data = {
            "user_id": "test_user_123",
            "original_instruction": "Debug the function",
            "context": "User is working on debugging.",
        }

        response = client_with_mock_agent.post("/propose_instructions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "test_user_123"

    def test_propose_instructions_without_agent(self, client_without_agent: TestClient) -> None:
        """Test instruction proposal when ToM agent is not available."""
        request_data = {"user_id": "test_user_123", "original_instruction": "Debug the function"}

        response = client_without_agent.post("/propose_instructions", json=request_data)
        assert response.status_code == 503
        assert "ToM Agent is not available" in response.json()["detail"]

    def test_propose_instructions_invalid_request(self, client_with_mock_agent: TestClient) -> None:
        """Test instruction proposal with invalid request data."""
        request_data = {
            "user_id": "test_user_123"
            # Missing required original_instruction and context
        }

        response = client_with_mock_agent.post("/propose_instructions", json=request_data)
        assert response.status_code == 422  # Validation error


class TestConversationStatusEndpoint:
    """Test the conversation_status endpoint."""

    def test_conversation_status_success(self, client_with_mock_agent: TestClient) -> None:
        """Test successful conversation status retrieval."""
        response = client_with_mock_agent.get("/conversation_status?user_id=test_user_123")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "test_user_123"
        assert "message" in data

    def test_conversation_status_without_agent(self, client_without_agent: TestClient) -> None:
        """Test conversation status when ToM agent is not available."""
        response = client_without_agent.get("/conversation_status?user_id=test_user_123")
        assert response.status_code == 200  # This endpoint doesn't depend on ToM agent

        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "test_user_123"


class TestAPIIntegration:
    """Integration tests for the API."""

    def test_api_workflow(self, client_with_mock_agent: TestClient) -> None:
        """Test a complete API workflow."""
        # 1. Check health
        health_response = client_with_mock_agent.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["tom_agent_ready"] is True

        # 2. Check conversation status
        status_response = client_with_mock_agent.get("/conversation_status?user_id=test_user_123")
        assert status_response.status_code == 200

        # 3. Propose instructions
        instructions_response = client_with_mock_agent.post(
            "/propose_instructions",
            json={
                "user_id": "test_user_123",
                "original_instruction": "Debug the function",
                "context": "User is working on debugging a Python web application.",
            },
        )
        assert instructions_response.status_code == 200

        # 4. Suggest next actions
        actions_response = client_with_mock_agent.post(
            "/suggest_next_actions",
            json={
                "user_id": "test_user_123",
                "context": "Agent has identified the issue and provided debugging steps.",
            },
        )
        assert actions_response.status_code == 200

        # All responses should be successful
        assert all(
            response.json()["success"]
            for response in [status_response, instructions_response, actions_response]
        )
