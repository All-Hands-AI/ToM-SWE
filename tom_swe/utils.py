import json
import logging
from typing import Generic, Type, TypeVar, Optional

import json_repair
from pydantic import BaseModel

try:
    from tom_swe.logging_config import get_tom_swe_logger

    logger = get_tom_swe_logger(__name__)
except ImportError:
    # Fallback for standalone use
    logger = logging.getLogger(__name__)

OutputType = TypeVar("OutputType", bound=object)
T = TypeVar("T", bound=BaseModel)


class OutputParser(BaseModel, Generic[OutputType]):
    def parse(self, result: str) -> OutputType:
        raise NotImplementedError

    def get_format_instructions(self) -> str:
        raise NotImplementedError


class PydanticOutputParser(OutputParser[T], Generic[T]):
    pydantic_object: Type[T]

    def parse(self, result: str) -> T:
        json_result = json_repair.loads(result)
        assert isinstance(json_result, dict)
        if "properties" in json_result:
            validated_result: T = self.pydantic_object.model_validate_json(
                json.dumps(json_result["properties"])
            )
            return validated_result
        else:
            parsed_result: T = self.pydantic_object.model_validate_json(result)
            return parsed_result

    def get_format_instructions(self) -> str:
        return json.dumps(self.pydantic_object.model_json_schema())


def split_text_for_embedding(text: str, max_tokens: int = 8191) -> list[str]:
    """
    Split text into chunks that stay within token limits for embedding models.
    Uses tiktoken for accurate token counting when available.
    """
    safe_max_tokens = min(max_tokens - 300, 7800)

    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text, disallowed_special=())

        if len(tokens) <= safe_max_tokens:
            return [text]

        # Split into chunks
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + safe_max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end

        return chunks

    except ImportError:
        # Fallback to character-based estimation
        max_chars = int(safe_max_tokens * 2.8)
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            start = end

        return chunks


def debug_large_prompt(
    prompt: str, user_context: Optional[object] = None, relevant_behavior: str = ""
) -> None:
    """Debug large prompts by logging details and saving to file."""
    prompt_length = len(prompt)

    if prompt_length > 50000:  # If prompt is suspiciously large
        logger.warning(f"⚠️  LARGE PROMPT DETECTED: {prompt_length:,} characters")

        if user_context and hasattr(user_context, "mental_state_summary"):
            logger.info(
                f"  - Mental state summary length: {len(user_context.mental_state_summary or ''):,}"
            )
        if user_context and hasattr(user_context, "preferences"):
            logger.info(f"  - Preferences count: {len(user_context.preferences or [])}")

        logger.info(f"  - RAG behavior snippet: {relevant_behavior[:200]}...")

        # Save full prompt to file for inspection
        with open("/tmp/large_prompt_debug.txt", "w") as f:
            f.write(prompt)
        logger.info("  - Full prompt saved to /tmp/large_prompt_debug.txt")


def format_proposed_suggestions(
    query: str,
    suggestions: str,
    confidence_score: float,
) -> str:
    final_instruction = f"""
Based on the conversation context and user patterns, here's a suggestion to help swe agent better understand and help the user:
*****************ToM Agent Analysis Start Here*****************
💡 Suggestion for SWE Agent (IMPORTANT!)
{suggestions}

💡 Confidence in the guidance
The ToM agent is {confidence_score*100:.0f}% confident in this suggestion for SWE agent.
*****************ToM Agent Analysis End Here*****************
"""

    return final_instruction
