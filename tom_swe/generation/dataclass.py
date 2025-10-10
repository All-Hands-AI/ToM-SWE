#!/usr/bin/env python3
"""
Database models for the Theory of Mind (ToM) Module

This module contains all Pydantic BaseModel classes used for data validation
and serialization in the ToM module.
"""

from typing import List, Union, Literal
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

    # Final Response Actions (contain structured response data in parameters)
    GENERATE_SUGGESTIONS = "generate_suggestions"
    GENERATE_SLEEP_SUMMARY = "generate_sleep_summary"


# Type-safe parameter models for each action
class ReadFileParams(BaseModel):
    """Parameters for READ_FILE action."""

    file_path: str = Field(
        description="Path to the file to read. You are only allowed to read user model related files. "
    )
    character_start: int = Field(
        default=5000,
        description="Character start to read from the file. Default starts from 5000 since search results would usually give the first 5000 characters of the file.",
    )
    character_end: int = Field(
        default=10000,
        description="Character end to read from the file. Default ends at 10000.",
    )


class SearchFileParams(BaseModel):
    """Parameters for SEARCH_FILE action - searches user memory and past interactions."""

    query: str = Field(
        description="Search query to find in past user sessions, messages, or interactions"
    )
    search_scope: str = Field(
        default="session_analyses",
        description="Scope of search: 'cleaned_sessions' (raw user interactions), 'session_analyses' (analyzed sessions), 'user_profiles' (overall user models)",
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
    chunk_size: int = Field(
        default=5000,
        ge=1000,
        le=10000,
        description="Chunk size to read from the file. The chunk size is the number of characters to read from the file. Default is 5000 characters.",
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
    new_value: str = Field(description="New value to set for the field")
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


class GenerateSuggestionsParams(BaseModel):
    """Parameters for GENERATE_SUGGESTIONS action - contains suggestions to help SWE agent make better decisions through user modeling."""

    suggestions: str = Field(
        description="Personalized suggestions for the SWE agent on how to better understand and help the user based on user modeling"
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) for the suggestion quality",
    )


class GenerateSleepSummaryParams(BaseModel):
    """Parameters for GENERATE_SLEEP_SUMMARY action - contains the final sleep time summary response."""

    summarization: str = Field(
        description="Summary of what changed during the session processing workflow"
    )


# Union type for all action parameters
ActionParams = Union[
    ReadFileParams,
    SearchFileParams,
    UpdateJsonFieldParams,
    AnalyzeSessionParams,
    InitializeUserProfileParams,
    RagSearchParams,
    GenerateSuggestionsParams,
    GenerateSleepSummaryParams,
]


class ActionResponse(BaseModel):
    """Type-safe structured response model for agent actions."""

    reasoning: str = Field(
        description="Providing a concise reasoning for the action you are going to take."
    )
    action: ActionType
    parameters: ActionParams
    is_complete: bool = False


class UserMessageAnalysis(BaseModel):
    message_content: str = Field(
        description="The content of the user message (Try to keep the original message as much as possible unless it's too long or contains too much user copy-pasted content; in a nutshell, try to preserve what the user actually said and include details if possible)"
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
    session_tldr: str = ""
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
    session_tldr: str = Field(
        description="A short summary of the session, 1-2 sentences. It's recommended to include some specific details to make it more useful for the SWE agent."
    )
    per_message_analysis: List[UserMessageAnalysis] = Field(default_factory=list)


class SessionSummary(BaseModel):
    session_id: str
    session_tldr: str = Field(
        description="A short summary of the session, 1-2 sentences. It's recommended to include some specific details to make it more useful for the SWE agent."
    )


class UserProfile(BaseModel):
    user_id: str
    overall_description: List[str] = Field(
        description="Overall description of the user's communication patterns, personality, expertise, and behavior"
    )
    preference_summary: List[str] = Field(
        description="Summarized list of user preferences extracted from all sessions. It's recommended to include some specific details to make it more useful for the SWE agent."
    )


class UserAnalysis(BaseModel):
    user_profile: UserProfile
    session_summaries: List[SessionSummary]
    last_updated: str


class ClarityAssessment(BaseModel):
    """Simple model for assessing instruction clarity."""

    reasoning: str = Field(
        description="Brief reasoning for why the instruction is clear or unclear"
    )
    is_clear: bool = Field(
        description="True if the instruction is clear enough to proceed without additional context, False if it needs clarification"
    )


class SWEAgentSuggestion(BaseModel):
    """Pydantic model for suggestions to help SWE agent make better decisions through user modeling."""

    original_query: str = Field(
        description="The original query or instruction that was analyzed"
    )
    suggestions: str = Field(
        description="The suggestions for the SWE agent based on user modeling and context analysis"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score for the suggestion quality"
    )


class QueryClassification(BaseModel):
    """LLM-based classification of SWE agent consultation queries."""

    category: Literal[
        "code_understanding", "development", "troubleshooting", "other"
    ] = Field(
        description="Primary category for the query based on software development workflow"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the classification (0.0 = very uncertain, 1.0 = very confident)",
    )
    reasoning: str = Field(
        description="Brief explanation of why this query fits the chosen category"
    )
