#!/usr/bin/env python3
"""
Database models for the Theory of Mind (ToM) Module

This module contains all Pydantic BaseModel classes used for data validation
and serialization in the ToM module.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class UserMessageAnalysis(BaseModel):
    intent: str = Field(
        description=(
            "The primary intent of the user message. Choose from: debugging, code_generation, "
            "code_explanation, optimization, learning, configuration, testing, file_management, general."
        )
    )
    emotions: List[str] = Field(
        description=(
            "List of emotional states detected in the message. Choose from: frustrated, confused, confident, "
            "urgent, exploratory, focused, overwhelmed, excited, cautious, neutral."
        )
    )
    preference: List[str] = Field(
        description=(
            "A list of preferences that the user has. Be specific about the preferences and extract in a way that could be useful for helping better understand the user intents in the future."
        )
    )
    user_modeling: str = Field(
        description=(
            "A very short description of the user's modeling (under 10 words). Basically tries to describe "
            "user highlevely based on the message. (E.g., the user seems to be proficient in PyTorch, "
            "or the user has limited programming experience might need more guidance when writing code.)"
        )
    )
    should_ask_clarification: bool = Field(
        description="Whether the coding agents should ask for clarification on the user's request. If the user's request is hard to understand, ask for clarification."
    )


class SessionSummary(BaseModel):
    session_id: str
    timestamp: datetime
    message_count: int
    intent_distribution: Dict[str, int]
    emotion_distribution: Dict[str, int]
    key_preferences: List[str]
    user_modeling_summary: str
    clarification_requests: int
    session_start: str
    session_end: str


class UserProfile(BaseModel):
    user_id: str
    total_sessions: int
    overall_description: str = Field(
        description="Overall description of the user's communication patterns, personality, expertise, and behavior"
    )
    intent_distribution: Dict[str, int]
    emotion_distribution: Dict[str, int]
    preference_summary: List[str] = Field(
        description="Summarized list of user preferences extracted from all sessions"
    )


class UserDescriptionAndPreferences(BaseModel):
    description: str = Field(
        description="2-3 sentence professional description of the user's communication patterns, personality, expertise level, and coding behavior when interacting with coding assistants"
    )
    preferences: List[str] = Field(
        description="List of up to 5 concise preference statements about how the user likes to work with coding assistants"
    )


class OverallUserAnalysis(BaseModel):
    user_profile: UserProfile
    session_summaries: List[SessionSummary]
    last_updated: datetime


class InstructionImprovementResponse(BaseModel):
    """Pydantic model for LLM response to instruction improvement requests."""

    improved_instruction: str = Field(
        description="The improved instruction personalized to the user's preferences and mental state"
    )
    reasoning: str = Field(
        description="Clear reasoning for the improvements made, explaining why they are suited to this user"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score (0-1) for the personalization quality"
    )
    personalization_factors: List[str] = Field(
        description="Key personalization factors applied from the user's profile and preferences"
    )


class NextActionSuggestionLLM(BaseModel):
    """Pydantic model for LLM response - a single next action suggestion."""

    action_description: str = Field(description="Clear description of the suggested action")
    priority: str = Field(description="Priority level of the action", pattern="^(high|medium|low)$")
    reasoning: str = Field(
        description="Reasoning for why this action is suggested based on user context"
    )
    expected_outcome: str = Field(description="Expected outcome or benefit of taking this action")
    user_preference_alignment: float = Field(
        ge=0.0,
        le=1.0,
        description="Score (0-1) indicating how well this action aligns with user preferences",
    )


class NextActionsResponse(BaseModel):
    """Pydantic model for LLM response to next action suggestions requests."""

    suggestions: List[NextActionSuggestionLLM] = Field(
        description="List of 1-5 next action suggestions ranked by relevance and priority",
    )

    @validator("suggestions")
    def validate_suggestions_length(
        self, v: List[NextActionSuggestionLLM]
    ) -> List[NextActionSuggestionLLM]:
        if len(v) < 1 or len(v) > 5:
            raise ValueError("suggestions must contain 1-5 items")
        return v


class InstructionRecommendation(BaseModel):
    """Pydantic model for an instruction recommendation."""

    original_instruction: str = Field(description="The original instruction that was improved")
    improved_instruction: str = Field(
        description="The improved instruction personalized to the user"
    )
    reasoning: str = Field(description="Reasoning for the improvements made")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score for the personalization quality"
    )
    personalization_factors: List[str] = Field(description="Key personalization factors applied")


class NextActionSuggestion(BaseModel):
    """Pydantic model for a next action suggestion."""

    action_description: str = Field(description="Description of the suggested action")
    priority: str = Field(description="Priority level: high, medium, or low")
    reasoning: str = Field(description="Reasoning for the suggestion")
    expected_outcome: str = Field(description="Expected outcome of taking this action")
    user_preference_alignment: float = Field(
        ge=0.0, le=1.0, description="Alignment score with user preferences"
    )


class UserContext(BaseModel):
    """Pydantic model for user context."""

    model_config = {"validate_assignment": True}

    user_id: str = Field(description="The user's unique identifier")
    user_profile: Optional[UserProfile] = None
    recent_sessions: Optional[List[SessionSummary]] = None
    current_query: Optional[str] = None
    preferences: Optional[List[str]] = None
    mental_state_summary: Optional[str] = None


class PersonalizedGuidance(BaseModel):
    """Pydantic model for complete personalized guidance."""

    user_context: UserContext
    instruction_recommendations: List[InstructionRecommendation]
    next_action_suggestions: List[NextActionSuggestion]
    overall_guidance: str
    confidence_score: float
