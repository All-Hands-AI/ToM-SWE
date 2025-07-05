"""
API models for the ToM Agent REST API.

This module defines Pydantic models for API requests and responses.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from ..database import (
    InstructionRecommendation,
    NextActionSuggestion,
    PersonalizedGuidance,
    UserContext,
)


class SuggestNextActionsRequest(BaseModel):
    """Request model for suggest_next_actions endpoint."""

    user_id: str = Field(description="The user ID to analyze")
    current_task_context: Optional[str] = Field(
        default=None, description="Optional context about the current task"
    )


class SuggestNextActionsResponse(BaseModel):
    """Response model for suggest_next_actions endpoint."""

    user_id: str = Field(description="The user ID that was analyzed")
    suggestions: List[NextActionSuggestion] = Field(
        description="List of next action suggestions"
    )
    success: bool = Field(description="Whether the request was successful")
    message: Optional[str] = Field(default=None, description="Optional status message")


class ProposeInstructionsRequest(BaseModel):
    """Request model for propose_instructions endpoint."""

    user_id: str = Field(description="The user ID to analyze")
    original_instruction: str = Field(description="The original instruction to improve")
    domain_context: Optional[str] = Field(
        default=None, description="Optional domain-specific context"
    )


class ProposeInstructionsResponse(BaseModel):
    """Response model for propose_instructions endpoint."""

    user_id: str = Field(description="The user ID that was analyzed")
    recommendations: List[InstructionRecommendation] = Field(
        description="List of instruction recommendations"
    )
    success: bool = Field(description="Whether the request was successful")
    message: Optional[str] = Field(default=None, description="Optional status message")


class SendMessageRequest(BaseModel):
    """Request model for send_message endpoint."""

    user_id: str = Field(description="The user ID")
    message: str = Field(description="The message to send to the ToM agent")
    instruction: Optional[str] = Field(
        default=None, description="Optional instruction to improve"
    )
    current_task: Optional[str] = Field(
        default=None, description="Optional current task context"
    )
    domain_context: Optional[str] = Field(
        default=None, description="Optional domain-specific context"
    )


class SendMessageResponse(BaseModel):
    """Response model for send_message endpoint."""

    user_id: str = Field(description="The user ID")
    guidance: PersonalizedGuidance = Field(
        description="Complete personalized guidance from the ToM agent"
    )
    success: bool = Field(description="Whether the request was successful")
    message: Optional[str] = Field(default=None, description="Optional status message")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(description="Health status")
    version: str = Field(description="API version")
    tom_agent_ready: bool = Field(description="Whether the ToM agent is ready")


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = Field(default=False, description="Always false for error responses")
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")