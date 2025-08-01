"""
API models for the ToM Agent REST API.

This module defines Pydantic models for API requests and responses.
Redesigned for bidirectional communication with SWE Agent.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from ..database import (
    InstructionRecommendation,
)


class ProposeInstructionsRequest(BaseModel):
    """Request model for propose_instructions endpoint."""

    user_id: str = Field(description="The user ID")
    original_instruction: str = Field(description="The original instruction to improve")
    context: str = Field(
        description="Context including previous conversations and interactions"
    )


class ProposeInstructionsResponse(BaseModel):
    """Response model for propose_instructions endpoint."""

    user_id: str = Field(description="The user ID")
    original_instruction: str = Field(
        description="The original instruction that was improved"
    )
    recommendations: List[InstructionRecommendation] = Field(
        description="List of instruction recommendations"
    )
    success: bool = Field(description="Whether the request was successful")
    message: Optional[str] = Field(default=None, description="Status message")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(description="Health status")
    version: str = Field(description="API version")
    tom_agent_ready: bool = Field(description="Whether the ToM agent is ready")


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = Field(default=False, description="Always false for error responses")
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(
        default=None, description="Detailed error information"
    )


class ConversationStatusResponse(BaseModel):
    """Response model for conversation status endpoint."""

    user_id: str = Field(description="The user ID")
    success: bool = Field(description="Whether the status check was successful")
    message: str = Field(description="Status message")
