#!/usr/bin/env python3
"""
Database models for the Theory of Mind (ToM) Module

This module contains all Pydantic BaseModel classes used for data validation
and serialization in the ToM module.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


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

    reasoning: str = Field(
        description="Clear reasoning for whether the user's original instruction is clear, focus on what could be missing and what could be inferred from the user's profile and recent sessions."
    )
    clarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Clarity score (0-1) indicating how clear the original user instruction is (the swe agent is considering the instruction as unclear, but you should give your own judgement combining with the past interactions with the specific user, which the swe agent has no access to), 0 means the original user instruction could be ambiguous or missing important details.",
    )
    improved_instruction: str = Field(
        description="Suggestions to the swe agent on how to better understand and help the user (e.g., 'you could respond the user with: `Based on your previous projects on ..., you might want to ...`'). If the user's instruction is unclear (low clarity score), should suggest the swe agent to ask for clarification (Be very strong about asking the agent to not do anything concretely without figuring out the user's intent first, e.g., 'IMPORTANT: Don't DO anything concretely, FIRST ask for clarification!!'). Also suggest some emojis to make the conversation with the user more engaging if users would prefer that based on your understanding of the user (e.g., 'you could use 🤔 to show that you are thinking about the user's request, or 🤯 to show that you are surprised by the user's request, or 🤗 to show that you are happy to help the user, etc.). Lastly, give swe agent suggestions on how to be mindful about the user's preferences and mental state (e.g., Avoid asking too many questions all at once, making the questions easier to answer for the user, making the output easier to understand for the user, etc.)."
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) for how confident you are about your suggestions to the swe agent. 0 means not confident at all, 1 means very confident.",
    )


class InstructionRecommendation(BaseModel):
    """Pydantic model for an instruction recommendation."""

    original_instruction: str = Field(
        description="The original instruction that was improved"
    )
    improved_instruction: str = Field(
        description="The improved instruction personalized to the user, think hard about what users really want to achieve and output markdown bullet points format with question marks in the points that you are not sure about. (style: blue text, italic)"
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


class UserContext(BaseModel):
    """Pydantic model for user context."""

    model_config = {"validate_assignment": True}

    user_id: str = Field(description="The user's unique identifier")
    user_profile: Optional[UserProfile] = None
    recent_sessions: Optional[List[SessionSummary]] = None
    current_query: Optional[str] = None
    preferences: Optional[List[str]] = None
    mental_state_summary: Optional[str] = None
