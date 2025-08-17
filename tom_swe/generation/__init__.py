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
    create_llm_client,
)
from .output_parsers import PydanticOutputParser
from .prompts import SLEEP_TIME_COMPUTATION_PROMPT, PROPOSE_INSTRUCTIONS_PROMPT
from .action import ActionExecutor
from .dataclass import ActionType, ActionResponse, SleepTimeResponse

__all__ = [
    "PydanticOutputParser",
    "LLMClient",
    "LLMConfig",
    "create_llm_client",
    "SLEEP_TIME_COMPUTATION_PROMPT",
    "PROPOSE_INSTRUCTIONS_PROMPT",
    "ActionType",
    "ActionResponse",
    "SleepTimeResponse",
    "ActionExecutor",
]
