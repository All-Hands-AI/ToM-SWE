"""
Generation utilities for robust LLM calling with Pydantic models.

This module provides improved LLM calling mechanisms with:
- Schema generation and inclusion in prompts
- Fallback parsing for malformed JSON
- Better error handling and retry logic
"""

from .generate import (
    LLMClient,
    LLMConfig,
    call_llm_simple,
    call_llm_structured,
    create_llm_client,
)
from .output_parsers import PydanticOutputParser

__all__ = [
    "PydanticOutputParser",
    "LLMClient",
    "LLMConfig",
    "create_llm_client",
    "call_llm_structured",
    "call_llm_simple",
]
