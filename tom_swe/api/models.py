"""
API models for the ToM Agent REST API.

This module defines Pydantic models for API requests and responses.
Redesigned for bidirectional communication with SWE Agent.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from ..database import (
    InstructionRecommendation,
    NextActionSuggestion,
)


# Message types for conversation state tracking
class MessageType:
    USER_MESSAGE = "user_message"  # User message to SWE Agent
    AGENT_RESPONSE = "agent_response"  # SWE Agent response to user


class SendMessageRequest(BaseModel):
    """Request model for send_message endpoint.

    This endpoint receives messages from SWE Agent and determines
    whether to calculate improved instructions or next actions.
    """

    user_id: str = Field(description="The user ID")
    message: str = Field(description="The message content")
    message_type: str = Field(description="Type of message: 'user_message' or 'agent_response'")
    original_instruction: Optional[str] = Field(
        default=None, description="Original instruction (for user messages)"
    )
    task_context: Optional[str] = Field(default=None, description="Current task context")
    domain_context: Optional[str] = Field(default=None, description="Domain-specific context")


class SendMessageResponse(BaseModel):
    """Response model for send_message endpoint."""

    user_id: str = Field(description="The user ID")
    message_received: bool = Field(description="Whether the message was processed")
    processing_type: str = Field(
        description="Type of processing triggered: 'instructions' or 'next_actions'"
    )
    success: bool = Field(description="Whether the request was successful")
    message: Optional[str] = Field(default=None, description="Status message")
    timestamp: datetime = Field(description="When the message was processed")


class ProposeInstructionsResponse(BaseModel):
    """Response model for GET propose_instructions endpoint.

    Returns cached instruction improvements generated after receiving user message.
    """

    user_id: str = Field(description="The user ID")
    original_instruction: Optional[str] = Field(
        default=None, description="The original instruction that was improved"
    )
    recommendations: List[InstructionRecommendation] = Field(
        description="List of instruction recommendations"
    )
    calculated_at: datetime = Field(description="When the instructions were calculated")
    success: bool = Field(description="Whether instructions are available")
    message: Optional[str] = Field(default=None, description="Status message")


class SuggestNextActionsResponse(BaseModel):
    """Response model for GET suggest_next_actions endpoint.

    Returns cached next action suggestions generated after receiving agent response.
    """

    user_id: str = Field(description="The user ID")
    suggestions: List[NextActionSuggestion] = Field(description="List of next action suggestions")
    based_on_context: str = Field(
        description="What the suggestions are based on (e.g., recent agent actions)"
    )
    calculated_at: datetime = Field(description="When the suggestions were calculated")
    success: bool = Field(description="Whether suggestions are available")
    message: Optional[str] = Field(default=None, description="Status message")


class ConversationState(BaseModel):
    """Model for tracking conversation state."""

    user_id: str = Field(description="The user ID")
    last_user_message: Optional[str] = Field(default=None, description="Last message from user")
    last_user_instruction: Optional[str] = Field(
        default=None, description="Last instruction from user"
    )
    last_agent_actions: List[str] = Field(
        default_factory=list, description="Recent agent actions/responses"
    )
    pending_instructions: Optional[List[InstructionRecommendation]] = Field(
        default=None, description="Cached instruction improvements"
    )
    pending_next_actions: Optional[List[NextActionSuggestion]] = Field(
        default=None, description="Cached next action suggestions"
    )
    instructions_calculated_at: Optional[datetime] = Field(
        default=None, description="When instructions were last calculated"
    )
    next_actions_calculated_at: Optional[datetime] = Field(
        default=None, description="When next actions were last calculated"
    )
    last_updated: datetime = Field(description="Last update timestamp")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(description="Health status")
    version: str = Field(description="API version")
    tom_agent_ready: bool = Field(description="Whether the ToM agent is ready")
    active_conversations: int = Field(description="Number of active conversation states")


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = Field(default=False, description="Always false for error responses")
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")


class ConversationStatusResponse(BaseModel):
    """Response model for conversation status endpoint."""

    user_id: str = Field(description="The user ID")
    has_pending_instructions: bool = Field(
        description="Whether there are pending instruction improvements"
    )
    has_pending_next_actions: bool = Field(
        description="Whether there are pending next action suggestions"
    )
    last_activity: Optional[datetime] = Field(
        default=None, description="Last conversation activity"
    )
    success: bool = Field(description="Whether the status check was successful")
