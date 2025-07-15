"""
Generation utilities for robust LLM calling with Pydantic models.

This module provides improved LLM calling mechanisms with:
- Schema generation and inclusion in prompts
- Fallback parsing for malformed JSON
- Better error handling and retry logic
"""

from .generate import call_llm_simple, call_llm_structured, format_bad_output, generate_with_schema
from .output_parsers import PydanticOutputParser

__all__ = [
    "PydanticOutputParser",
    "generate_with_schema",
    "call_llm_structured",
    "call_llm_simple",
    "format_bad_output",
]
