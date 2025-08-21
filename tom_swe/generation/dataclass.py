#!/usr/bin/env python3
"""
Database models for the Theory of Mind (ToM) Module

This module contains all Pydantic BaseModel classes used for data validation
and serialization in the ToM module.
"""

from typing import Any, List, Union
from enum import Enum

from pydantic import BaseModel, Field


class ActionType(Enum):
    """Available actions for the agent workflow controller."""

    # Core File Operations
    READ_FILE = "read_file"
    SEARCH_FILE = "search_file"
    UPDATE_JSON_FIELD = "update_json_field"

    # Session Analysis (Tier 2 - Per-Session Models)
    ANALYZE_SESSION = "analyze_session"

    # Overall User Model (Tier 3 - Aggregated Profile)
    INITIALIZE_USER_PROFILE = "initialize_user_profile"

    # RAG Operations
    RAG_SEARCH = "rag_search"


# Type-safe parameter models for each action
class ReadFileParams(BaseModel):
    """Parameters for READ_FILE action."""

    file_path: str = Field(description="Path to the file to read")


class SearchFileParams(BaseModel):
    """Parameters for SEARCH_FILE action - searches user memory and past interactions."""

    query: str = Field(
        description="Search query to find in past user sessions, messages, or interactions"
    )
    search_scope: str = Field(
        default="session_analyses",
        description="Scope of search: 'cleaned_sessions' (raw user interactions), 'session_analyses' (analyzed sessions), 'user_profiles' (overall user models), or 'all'",
    )
    search_method: str = Field(
        default="bm25",
        description="Search method: 'bm25' (semantic ranking) or 'string_match' (exact substring)",
    )
    max_results: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Maximum number of matching files/sessions to return",
    )
    latest_first: bool = Field(
        default=True,
        description="Return most recent interactions first based on update_time in JSON data",
    )


class UpdateJsonFieldParams(BaseModel):
    """Parameters for UPDATE_JSON_FIELD action."""

    field_path: str = Field(
        description="Dot notation path to the field (e.g., 'user.preferences.theme')"
    )
    new_value: Any = Field(description="New value to set for the field")
    list_operation: str = Field(
        default="append",
        description="List operation: 'append' or 'remove' (by value or index)",
    )
    create_if_missing: bool = Field(
        default=False, description="Create parent fields/file if they don't exist"
    )
    backup: bool = Field(default=True, description="Create backup before modifying")


class AnalyzeSessionParams(BaseModel):
    """Parameters for ANALYZE_SESSION action."""

    user_id: str = Field(description="User ID for session analysis")
    session_batch: List[str] = Field(description="List of session IDs to analyze")


class InitializeUserProfileParams(BaseModel):
    """Parameters for INITIALIZE_USER_PROFILE action."""

    user_id: str = Field(description="User ID for profile initialization")


class RagSearchParams(BaseModel):
    """Parameters for RAG_SEARCH action."""

    query: str = Field(description="Query for RAG search")
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top results to return",
    )


class CompleteTaskParams(BaseModel):
    """Parameters for COMPLETE_TASK action."""

    result: str = Field(description="Result message for the completed task")


# Union type for all action parameters
ActionParams = Union[
    ReadFileParams,
    SearchFileParams,
    UpdateJsonFieldParams,
    AnalyzeSessionParams,
    InitializeUserProfileParams,
    RagSearchParams,
]


class ActionResponse(BaseModel):
    """Type-safe structured response model for agent actions."""

    reasoning: str = Field(
        description="Providing a concise reasoning for the action you are going to take."
    )
    action: ActionType
    parameters: ActionParams
    is_complete: bool = False


class SleepTimeResponse(BaseModel):
    """Summary of what changed during session processing workflow."""

    summarization: str


class UserMessageAnalysis(BaseModel):
    message_content: str = Field(
        description="The content of the user message (could be a summarization if the original message is too long)"
    )
    emotions: str = Field(
        description=(
            "A description of the emotional states detected in the message. Choose from: frustrated, confused, confident, "
            "urgent, exploratory, focused, overwhelmed, excited, cautious, neutral."
        )
    )
    preference: str = Field(
        description=(
            "A description of the preferences that the user has. Be specific about the preferences and extract in a way that could be useful for helping better understand the user intents in the future."
        )
    )


class SessionAnalysis(BaseModel):
    session_id: str
    user_modeling_summary: str
    intent: str
    per_message_analysis: List[UserMessageAnalysis]
    session_start: str = ""
    session_end: str = ""
    last_updated: str


class SessionAnalysisForLLM(BaseModel):
    user_modeling_summary: str = Field(
        description="User modeling summary describing session goals and behavioral characteristics"
    )
    intent: str = Field(
        description=(
            "The primary intent of the session. Choose from: debugging, code_generation, "
            "code_explanation, optimization, learning, configuration, testing, file_management, general."
        )
    )
    per_message_analysis: List[UserMessageAnalysis] = Field(default_factory=list)


class SessionSummary(BaseModel):
    session_id: str
    session_tldr: str = Field(
        description="A short summary of the session, 1-2 sentences"
    )


class UserProfile(BaseModel):
    user_id: str
    overall_description: List[str] = Field(
        description="Overall description of the user's communication patterns, personality, expertise, and behavior"
    )
    preference_summary: List[str] = Field(
        description="Summarized list of user preferences extracted from all sessions"
    )


class UserAnalysisForLLM(BaseModel):
    user_profile: UserProfile
    session_summaries: List[SessionSummary]


class UserAnalysis(BaseModel):
    user_profile: UserProfile
    session_summaries: List[SessionSummary]
    last_updated: str


class InstructionImprovementLLM(BaseModel):
    """Pydantic model for LLM response to instruction improvement requests."""

    reasoning: str = Field(
        description="Clear reasoning for whether the user's original instruction is clear, focus on what could be missing and what could be inferred from the user's profile and recent sessions."
    )
    clarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Clarity score (0-1) indicating how clear the original user instruction is",
    )
    improved_instruction: str = Field(
        description="Personalized suggestions for the SWE agent on how to better understand and help the user"
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) for the suggestion quality",
    )


class ClarityAssessment(BaseModel):
    """Simple model for assessing instruction clarity."""

    reasoning: str = Field(
        description="Brief reasoning for why the instruction is clear or unclear"
    )
    is_clear: bool = Field(
        description="True if the instruction is clear enough to proceed without additional context, False if it needs clarification"
    )


class InstructionImprovement(BaseModel):
    """Pydantic model for an instruction recommendation."""

    original_instruction: str = Field(
        description="The original instruction that was improved"
    )
    improved_instruction: str = Field(
        description="The improved instruction personalized to the user in markdown format"
    )
    reasoning: str = Field(description="Reasoning for the improvements made")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score for the personalization quality"
    )
    clarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Clarity score (0-1) indicating how clear the original instruction was",
    )
