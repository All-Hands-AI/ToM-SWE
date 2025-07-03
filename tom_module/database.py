#!/usr/bin/env python3
"""
Database models for the Theory of Mind (ToM) Module

This module contains all Pydantic BaseModel classes used for data validation
and serialization in the ToM module.
"""

from datetime import datetime
from typing import Dict, List

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

    @property
    def dominant_intents(self) -> List[str]:
        """Get the most common intents in order of frequency."""
        if not self.intent_distribution:
            return []
        return sorted(self.intent_distribution.keys(), key=lambda x: self.intent_distribution[x], reverse=True)

    @property
    def emotional_progression(self) -> List[str]:
        """Get the most common emotions in order of frequency."""
        if not self.emotion_distribution:
            return []
        return sorted(self.emotion_distribution.keys(), key=lambda x: self.emotion_distribution[x], reverse=True)


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
    expertise_indicators: List[str] = Field(
        default_factory=list,
        description="List of indicators showing user's expertise level and areas of knowledge"
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
