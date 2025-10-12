"""LLM Client with robust structured output support.

This module provides an LLMClient class that encapsulates configuration
and provides both async and sync methods for structured LLM calls.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Type, TypeVar, List, Dict, Any

from litellm import acompletion, completion
from pydantic import BaseModel

from .output_parsers import PydanticOutputParser

try:
    from tom_swe.logging_config import get_tom_swe_logger

    logger = get_tom_swe_logger(__name__)
except ImportError:
    # Fallback for standalone use
    logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Default LLM configuration
DEFAULT_MODEL = "litellm_proxy/claude-sonnet-4-20250514"

# Fallback model for fixing bad outputs
DEFAULT_BAD_OUTPUT_PROCESS_MODEL = "gpt-5-nano"


@dataclass
class LLMConfig:
    """Configuration for LLM calls."""

    model: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    fallback_model: str = DEFAULT_BAD_OUTPUT_PROCESS_MODEL


class LLMClient:
    """
    LLM client that encapsulates configuration and provides both async and sync methods.

    This class stores the LLM configuration as instance attributes and provides
    a clean interface for structured LLM operations with built-in fallback mechanisms.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client with configuration.

        Args:
            config: LLM configuration to use for all calls
        """
        self.config = config
        logger.info(f"LLMClient initialized with model: {config.model}")

    async def format_bad_output(
        self,
        ill_formed_output: str,
        format_instructions: str,
        output_type: Type[T],
    ) -> str:
        """
        Reformat ill-formed output to valid JSON using a fallback model.

        Args:
            ill_formed_output: The malformed output from the original LLM
            format_instructions: The format instructions that should be followed

        Returns:
            Reformatted JSON string
        """
        template = """
        Given the string that can not be parsed by json parser, reformat it to a string that can be parsed by json parser.
        Original string: {ill_formed_output}

        Format instructions: {format_instructions}

        Please only generate the JSON:
        """

        input_values = {
            "ill_formed_output": ill_formed_output,
            "format_instructions": format_instructions,
        }
        content = template.format(**input_values)

        completion_args = {
            "model": self.config.fallback_model,
            "response_format": output_type,
            "messages": [{"role": "user", "content": content}],
        }

        if self.config.api_key:
            completion_args["api_key"] = str(self.config.api_key)
        if self.config.api_base:
            completion_args["api_base"] = str(self.config.api_base)

        response = await acompletion(**completion_args)
        reformatted_output = response.choices[0].message.content
        assert isinstance(reformatted_output, str)
        logger.info(f"Reformatted output: {reformatted_output}")
        return reformatted_output

    async def call_structured_async(
        self,
        prompt: str,
        output_type: Type[T],
        max_tokens: Optional[int] = None,
    ) -> T:
        """
        Call LLM with structured output (async).

        Args:
            prompt: The main prompt for the LLM
            output_type: Pydantic model class for the expected output
            max_tokens: Override default max_tokens for this call

        Returns:
            Parsed and validated Pydantic model instance
        """
        # Create output parser
        output_parser = PydanticOutputParser(output_type)

        # Construct the full prompt with schema instructions
        format_instructions = output_parser.get_format_instructions()
        full_prompt = f"{prompt}\n\n{format_instructions}"

        logger.info(f"Full prompt {len(full_prompt)} characters")

        # Prepare completion arguments
        completion_args = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": full_prompt}],
            "response_format": output_type,
        }

        # Add optional parameters
        if self.config.api_key:
            completion_args["api_key"] = str(self.config.api_key)
        if self.config.api_base:
            completion_args["api_base"] = str(self.config.api_base)

        # Call LLM
        response = await acompletion(**completion_args)
        content = response.choices[0].message.content

        if not content:
            raise ValueError("Empty response from LLM")

        logger.debug(f"Raw LLM response (first 200 chars): {content[:200]}...")

        # Try to parse the response
        try:
            result = output_parser.parse(content)
            logger.debug("Successfully parsed output on first attempt")
            return result
        except ValueError as parse_error:
            logger.warning(f"Parse failed on first attempt: {parse_error}")
            logger.info("Attempting to reformat bad output with fallback model")

            # Use fallback model to fix the output
            try:
                reformatted_output = await self.format_bad_output(
                    ill_formed_output=content,
                    format_instructions=format_instructions,
                    output_type=output_type,
                )

                # Try to parse the reformatted output
                result = output_parser.parse(reformatted_output)
                logger.info("Successfully parsed reformatted output")
                return result

            except Exception as fallback_error:
                logger.error(f"Fallback reformatting failed: {fallback_error}")
                raise ValueError(
                    f"Failed to parse output even after reformatting. Original error: {parse_error}, Fallback error: {fallback_error}"
                ) from parse_error

    def call_structured_messages(
        self,
        messages: List[Dict[str, Any]],
        output_type: Type[T],
        max_tokens: Optional[int] = None,
    ) -> T:
        """
        Call LLM with structured output (sync).

        Args:
            messages: List of message dicts (supports cache_control if present)
            output_type: Pydantic model class for the expected output
            max_tokens: Override default max_tokens for this call

        Returns:
            Parsed and validated Pydantic model instance
        """
        # Create output parser
        output_parser = PydanticOutputParser(output_type)

        # Construct the full messages list with schema instructions
        format_instructions = output_parser.get_format_instructions()
        full_messages = messages + [{"role": "user", "content": format_instructions}]

        logger.info(f"Full prompt {len(full_messages)} messages")

        # Prepare completion arguments
        completion_args = {
            "model": self.config.model,
            "messages": full_messages,
            "response_format": output_type,
        }

        # Add optional parameters
        if self.config.api_key:
            completion_args["api_key"] = str(self.config.api_key)
        if self.config.api_base:
            completion_args["api_base"] = str(self.config.api_base)

        response = completion(**completion_args)
        content = response.choices[0].message.content

        if not content:
            raise ValueError("Empty response from LLM")

        logger.debug(f"Raw LLM response (first 200 chars): {content[:200]}...")

        # Try to parse the response
        try:
            result = output_parser.parse(content)
            logger.debug("Successfully parsed output on first attempt")
            return result
        except ValueError as parse_error:
            logger.warning(f"Parse failed on first attempt: {parse_error}")
            logger.info("Attempting to reformat bad output with fallback model")

            # Use fallback model to fix the output (sync version)
            try:
                # Format bad output synchronously
                template = """
        Given the string that can not be parsed by json parser, reformat it to a string that can be parsed by json parser.
        Original string: {ill_formed_output}

        Format instructions: {format_instructions}

        Please only generate the JSON:
        """
                input_values = {
                    "ill_formed_output": content,
                    "format_instructions": format_instructions,
                }
                fallback_content = template.format(**input_values)

                fallback_completion_args = {
                    "model": self.config.fallback_model,
                    "response_format": output_type,
                    "messages": [{"role": "user", "content": fallback_content}],
                }

                if self.config.api_key:
                    fallback_completion_args["api_key"] = self.config.api_key
                if self.config.api_base:
                    fallback_completion_args["api_base"] = self.config.api_base

                # Call fallback model synchronously using completion (not acompletion)
                fallback_response = completion(**fallback_completion_args)

                reformatted_output = fallback_response.choices[0].message.content
                assert isinstance(reformatted_output, str)
                logger.info(f"Reformatted output: {reformatted_output}")

                # Try to parse the reformatted output
                result = output_parser.parse(reformatted_output)
                logger.info("Successfully parsed reformatted output")
                return result

            except Exception as fallback_error:
                logger.error(f"Fallback reformatting failed: {fallback_error}")
                raise ValueError(
                    f"Failed to parse output even after reformatting. Original error: {parse_error}, Fallback error: {fallback_error}"
                ) from parse_error
