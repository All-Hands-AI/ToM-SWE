"""
Theory of Mind (ToM) Module for User Behavior Analysis

This package analyzes user interaction data to understand their mental states,
predict their intentions, and anticipate their next actions based on their
typed messages and interaction patterns using Large Language Models.
"""

from tom_swe.database import (
    InstructionImprovementResponse,
    InstructionRecommendation,
    NextActionsResponse,
    NextActionSuggestion,
    OverallUserAnalysis,
    PersonalizedGuidance,
    SessionSummary,
    UserContext,
    UserMessageAnalysis,
    UserProfile,
)
from tom_swe.rag_module import (
    ChunkingConfig,
    Document,
    RAGAgent,
    RetrievalResult,
    VectorDB,
    create_rag_agent,
    load_processed_data,
)
from tom_swe.tom_agent import (
    ToMAgent,
    create_tom_agent,
)
from tom_swe.tom_module import (
    UserMentalStateAnalyzer,
)

__version__ = "1.0.0"
__author__ = "Research Team"
__description__ = "LLM-powered Theory of Mind analysis for user behavior prediction"

__all__ = [
    "Document",
    "InstructionRecommendation",
    "NextActionSuggestion",
    "OverallUserAnalysis",
    "PersonalizedGuidance",
    "RAGAgent",
    "RetrievalResult",
    "SessionSummary",
    "ToMAgent",
    "UserContext",
    "UserMentalStateAnalyzer",
    "UserMessageAnalysis",
    "UserProfile",
    "VectorDB",
    "create_rag_agent",
    "create_tom_agent",
    "load_processed_data",
    "load_user_model_data",
    "ChunkingConfig",
    "InstructionImprovementResponse",
    "InstructionRecommendation",
    "NextActionsResponse",
    "NextActionSuggestion",
    "PersonalizedGuidance",
    "UserContext",
]
