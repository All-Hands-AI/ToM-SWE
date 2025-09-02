"""
Theory of Mind (ToM) Module for User Behavior Analysis

This package analyzes user interaction data to understand their mental states,
predict their intentions, and anticipate their next actions based on their
typed messages and interaction patterns using Large Language Models.
"""

# Auto-configure logging for integration with parent applications
try:
    from tom_swe.logging_config import auto_configure_logging

    auto_configure_logging()
except ImportError:
    # Logging configuration not available, continue without it
    pass

from tom_swe.generation.dataclass import (
    SWEAgentSuggestion,
    UserAnalysis,
    SessionAnalysis,
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
    ToMAnalyzer,
)
from tom_swe.memory.conversation_processor import (
    CleanSessionStore,
)

__version__ = "1.0.0"
__author__ = "Research Team"
__description__ = "LLM-powered Theory of Mind analysis for user behavior prediction"

__all__ = [
    "ChunkingConfig",
    "Document",
    "SWEAgentSuggestion",
    "UserAnalysis",
    "RAGAgent",
    "RetrievalResult",
    "SessionAnalysis",
    "ToMAgent",
    "ToMAnalyzer",
    "UserMessageAnalysis",
    "UserProfile",
    "VectorDB",
    "create_rag_agent",
    "create_tom_agent",
    "load_processed_data",
    "load_user_model_data",
    "CleanSessionStore",
]
