#!/usr/bin/env python3
"""
Comprehensive tests for instruction improvement API functionality after cleanup.

Tests the API endpoints that remain after removing next action functionality,
focusing on instruction improvement, health checks, and conversation status.
"""

from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from tom_swe.database import InstructionRecommendation, UserContext
from tom_swe.tom_agent import ToMAgent


class TestAPIConfiguration:
    """Test API configuration and startup."""

    def test_api_imports(self) -> None:
        """Test that API imports work correctly."""
        from tom_swe.api.main import app
        from tom_swe.api.models import ConversationStatusResponse, ProposeInstructionsRequest

        assert app is not None
        assert ConversationStatusResponse is not None
        assert ProposeInstructionsRequest is not None

    def test_openapi_schema(self) -> None:
        """Test that OpenAPI schema is properly configured."""
        from tom_swe.api.main import app

        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "ToM-SWE Instruction Improvement API"

    def test_docs_endpoint(self) -> None:
        """Test that docs endpoint is accessible."""
        from tom_swe.api.main import app

        client = TestClient(app)
        response = client.get("/docs")
        assert response.status_code == 200

    @pytest.fixture
    def test_client(self) -> Generator[TestClient, None, None]:
        """Create a test client for the API."""
        from tom_swe.api.main import app

        with TestClient(app) as client:
            yield client

    @pytest.fixture
    async def async_client(self) -> AsyncGenerator[AsyncClient, None]:
        """Create an async test client for the API."""
        from tom_swe.api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client


class TestHealthEndpoint:
    """Test the health endpoint."""

    @pytest.fixture
    def test_client(self) -> Generator[TestClient, None, None]:
        """Create a test client for the API."""
        from tom_swe.api.main import app

        with TestClient(app) as client:
            yield client

    def test_health_endpoint_exists(self, test_client: TestClient) -> None:
        """Test that health endpoint exists and returns correct response."""
        response = test_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "tom-swe-instruction-improvement"

    def test_health_endpoint_structure(self, test_client: TestClient) -> None:
        """Test the structure of health endpoint response."""
        response = test_client.get("/health")
        data = response.json()

        # Check required fields
        required_fields = ["status", "timestamp", "service", "version"]
        for field in required_fields:
            assert field in data

        # Check data types
        assert isinstance(data["status"], str)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["service"], str)
        assert isinstance(data["version"], str)


class TestConversationStatusEndpoint:
    """Test the conversation status endpoint."""

    @pytest.fixture
    def test_client(self) -> Generator[TestClient, None, None]:
        """Create a test client for the API."""
        from tom_swe.api.main import app

        with TestClient(app) as client:
            yield client

    def test_conversation_status_endpoint_exists(self, test_client: TestClient) -> None:
        """Test that conversation status endpoint exists."""
        response = test_client.get("/conversation_status")
        assert response.status_code == 200

        data = response.json()
        assert "active_features" in data
        assert "removed_features" in data

    def test_conversation_status_cleanup_verification(self, test_client: TestClient) -> None:
        """Test that conversation status reflects cleanup correctly."""
        response = test_client.get("/conversation_status")
        data = response.json()

        # Check that instruction improvement is active
        assert "instruction_improvement" in data["active_features"]

        # Check that next action features are in removed features
        removed_features = data["removed_features"]
        assert "next_action_suggestions" in removed_features
        assert "personalized_guidance" in removed_features

        # Verify feature descriptions
        active = data["active_features"]["instruction_improvement"]
        assert "improves user instructions" in active.lower()

    def test_conversation_status_response_structure(self, test_client: TestClient) -> None:
        """Test the structure of conversation status response."""
        response = test_client.get("/conversation_status")
        data = response.json()

        # Check required top-level fields
        assert "active_features" in data
        assert "removed_features" in data
        assert "cleanup_info" in data

        # Check that active features have descriptions
        for feature, description in data["active_features"].items():
            assert isinstance(feature, str)
            assert isinstance(description, str)
            assert len(description) > 0


class TestInstructionImprovementEndpoint:
    """Test the /propose_instructions endpoint."""

    @pytest.fixture
    def test_client(self) -> Generator[TestClient, None, None]:
        """Create a test client for the API."""
        from tom_swe.api.main import app

        with TestClient(app) as client:
            yield client

    @pytest.fixture
    async def async_client(self) -> AsyncGenerator[AsyncClient, None]:
        """Create an async test client for the API."""
        from tom_swe.api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client

    @pytest.fixture
    def mock_tom_agent(self) -> Generator[MagicMock, None, None]:
        """Create a mock ToM agent for testing."""
        with patch("tom_swe.api.main.get_tom_agent") as mock_get_agent:
            mock_agent = MagicMock(spec=ToMAgent)

            # Mock analyze_user_context
            mock_context = UserContext(
                user_id="test_user",
                current_query="Help me debug this code",
                preferences=["Detailed explanations", "Step-by-step guidance"],
                mental_state_summary="Focused developer",
            )
            mock_agent.analyze_user_context = AsyncMock(return_value=mock_context)

            # Mock propose_instructions
            mock_recommendation = InstructionRecommendation(
                original_instruction="Fix this bug",
                improved_instruction="Here's a detailed debugging approach:\n1. Check error logs\n2. Examine variables\n3. Add debug prints",
                reasoning="User prefers detailed step-by-step guidance",
                confidence_score=0.85,
                clarity_score=0.90,
            )
            mock_agent.propose_instructions = AsyncMock(return_value=[mock_recommendation])

            mock_get_agent.return_value = mock_agent
            yield mock_agent

    def test_propose_instructions_endpoint_exists(self, test_client: TestClient) -> None:
        """Test that the propose_instructions endpoint exists."""
        # This should return 422 (validation error) since we're not sending valid data
        response = test_client.post("/propose_instructions")
        assert response.status_code == 422  # Validation error, but endpoint exists

    @pytest.mark.asyncio
    async def test_propose_instructions_success(
        self, async_client: AsyncClient, mock_tom_agent: MagicMock
    ) -> None:
        """Test successful instruction proposal."""
        request_data = {
            "user_id": "test_user",
            "original_instruction": "Fix this bug",
            "context": "Working on a Python debugging task",
        }

        response = await async_client.post("/propose_instructions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) == 1

        rec = data["recommendations"][0]
        assert rec["original_instruction"] == "Fix this bug"
        assert "debugging approach" in rec["improved_instruction"]
        assert rec["confidence_score"] == 0.85
        assert rec["clarity_score"] == 0.90

        # Verify mock calls
        mock_tom_agent.analyze_user_context.assert_called_once_with("test_user", "Fix this bug")
        mock_tom_agent.propose_instructions.assert_called_once()

    @pytest.mark.asyncio
    async def test_propose_instructions_validation_error(self, async_client: AsyncClient) -> None:
        """Test validation error for invalid request."""
        # Missing required fields
        request_data = {"user_id": "test_user"}

        response = await async_client.post("/propose_instructions", json=request_data)
        assert response.status_code == 422

        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_propose_instructions_empty_result(
        self, async_client: AsyncClient, mock_tom_agent: MagicMock
    ) -> None:
        """Test when ToM agent returns no recommendations."""
        # Mock empty result
        mock_tom_agent.propose_instructions = AsyncMock(return_value=[])

        request_data = {
            "user_id": "test_user",
            "original_instruction": "Help me",
            "context": "General help request",
        }

        response = await async_client.post("/propose_instructions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) == 0

    @pytest.mark.asyncio
    async def test_propose_instructions_agent_failure(
        self, async_client: AsyncClient, mock_tom_agent: MagicMock
    ) -> None:
        """Test when ToM agent raises an exception."""
        # Mock agent failure
        mock_tom_agent.analyze_user_context = AsyncMock(side_effect=Exception("Agent failure"))

        request_data = {
            "user_id": "test_user",
            "original_instruction": "Fix this bug",
            "context": "Working on debugging",
        }

        response = await async_client.post("/propose_instructions", json=request_data)
        assert response.status_code == 500

        data = response.json()
        assert "detail" in data
        assert "Agent failure" in data["detail"]


class TestRemovedEndpoints:
    """Test that removed endpoints return proper 404 responses."""

    @pytest.fixture
    def test_client(self) -> Generator[TestClient, None, None]:
        """Create a test client for the API."""
        from tom_swe.api.main import app

        with TestClient(app) as client:
            yield client

    def test_removed_suggest_next_actions_endpoint(self, test_client: TestClient) -> None:
        """Test that removed suggest_next_actions endpoint returns 404."""
        response = test_client.post(
            "/suggest_next_actions", json={"user_id": "test", "context": "test"}
        )
        assert response.status_code == 404

    def test_removed_personalized_guidance_endpoint(self, test_client: TestClient) -> None:
        """Test that removed personalized_guidance endpoint returns 404."""
        response = test_client.post(
            "/personalized_guidance", json={"user_id": "test", "context": "test"}
        )
        assert response.status_code == 404


class TestAPIModels:
    """Test API model validation."""

    def test_instruction_request_model(self) -> None:
        """Test ProposeInstructionsRequest model validation."""
        from tom_swe.api.models import ProposeInstructionsRequest

        # Valid request
        request = ProposeInstructionsRequest(
            user_id="test_user",
            original_instruction="Fix this bug",
            context="Working on debugging",
        )
        assert request.user_id == "test_user"
        assert request.original_instruction == "Fix this bug"
        assert request.context == "Working on debugging"

    def test_instruction_request_validation_error(self) -> None:
        """Test ProposeInstructionsRequest validation with missing fields."""
        from pydantic import ValidationError

        from tom_swe.api.models import ProposeInstructionsRequest

        # Missing required fields
        with pytest.raises(ValidationError):
            ProposeInstructionsRequest(user_id="test_user")  # type: ignore[call-arg] # Missing original_instruction and context

    def test_conversation_status_model(self) -> None:
        """Test ConversationStatusResponse model."""
        from tom_swe.api.models import ConversationStatusResponse

        status = ConversationStatusResponse(
            user_id="test_user",
            success=True,
            message="System cleaned up successfully",
        )

        assert status.user_id == "test_user"
        assert status.success is True
        assert status.message == "System cleaned up successfully"


class TestAPIDocumentation:
    """Test API documentation and OpenAPI schema."""

    @pytest.fixture
    def test_client(self) -> Generator[TestClient, None, None]:
        """Create a test client for the API."""
        from tom_swe.api.main import app

        with TestClient(app) as client:
            yield client

    def test_openapi_schema_structure(self, test_client: TestClient) -> None:
        """Test OpenAPI schema structure."""
        response = test_client.get("/openapi.json")
        schema = response.json()

        # Check basic structure
        assert "paths" in schema
        assert "components" in schema

        # Check that our endpoints are documented
        paths = schema["paths"]
        assert "/health" in paths
        assert "/conversation_status" in paths
        assert "/propose_instructions" in paths

        # Check that removed endpoints are not in schema
        assert "/suggest_next_actions" not in paths
        assert "/personalized_guidance" not in paths

    def test_endpoint_documentation(self, test_client: TestClient) -> None:
        """Test that endpoints have proper documentation."""
        response = test_client.get("/openapi.json")
        schema = response.json()

        # Check propose_instructions endpoint documentation
        propose_path = schema["paths"]["/propose_instructions"]["post"]
        assert "summary" in propose_path
        assert "requestBody" in propose_path
        assert "responses" in propose_path

        # Check health endpoint documentation
        health_path = schema["paths"]["/health"]["get"]
        assert "summary" in health_path
        assert "responses" in health_path


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
