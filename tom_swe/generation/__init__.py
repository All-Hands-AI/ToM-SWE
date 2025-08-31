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
)
from .output_parsers import PydanticOutputParser

# Note: Prompts now use Jinja2 templates in tom_swe.prompts.templates
from .action import ActionExecutor
from .dataclass import ActionType, ActionResponse

__all__ = [
    "PydanticOutputParser",
    "LLMClient",
    "LLMConfig",
    "ActionType",
    "ActionResponse",
    "ActionExecutor",
]
