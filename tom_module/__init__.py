"""
Theory of Mind (ToM) Module for User Behavior Analysis

This package analyzes user interaction data to understand their mental states,
predict their intentions, and anticipate their next actions based on their
typed messages and interaction patterns using Large Language Models.
"""

from tom_module.database import (
    OverallUserAnalysis,
    SessionSummary,
    UserMessageAnalysis,
    UserProfile,
)
from tom_module.rag_agent import (
    ContextualVectorDB,
    Document,
    RAGAgent,
    RetrievalResult,
    VectorDB,
    create_rag_agent,
    load_processed_data,
    load_user_model_data,
)
from tom_module.tom_module import (
    UserMentalStateAnalyzer,
)

__version__ = "1.0.0"
__author__ = "Research Team"
__description__ = "LLM-powered Theory of Mind analysis for user behavior prediction"

__all__ = [
    "ContextualVectorDB",
    "Document",
    "OverallUserAnalysis",
    "RAGAgent",
    "RetrievalResult",
    "SessionSummary",
    "UserMentalStateAnalyzer",
    "UserMessageAnalysis",
    "UserProfile",
    "VectorDB",
    "create_rag_agent",
    "load_processed_data",
    "load_user_model_data",
]
