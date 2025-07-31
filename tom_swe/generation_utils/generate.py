"""LLM Client with robust structured output support.

This module provides an LLMClient class that encapsulates configuration
and provides both async and sync methods for LLM calls.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional, Type, TypeVar

from litellm import acompletion
from pydantic import BaseModel

from .output_parsers import PydanticOutputParser

logger = logging.getLogger(__name__)

# Set logger level based on environment variable
log_level = os.getenv("LOG_LEVEL", "info").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

T = TypeVar("T", bound=BaseModel)

# Default LLM configuration
DEFAULT_MODEL = "litellm_proxy/claude-sonnet-4-20250514"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1024

# Fallback model for fixing bad outputs
DEFAULT_BAD_OUTPUT_PROCESS_MODEL = "gpt-4o-mini"


@dataclass
class LLMConfig:
    """Configuration for LLM calls."""

    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    fallback_model: str = DEFAULT_BAD_OUTPUT_PROCESS_MODEL


class LLMClient:
    """
    LLM client that encapsulates configuration and provides both async and sync methods.

    This class stores the LLM configuration as instance attributes and provides
    a clean interface for all LLM operations with built-in fallback mechanisms.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client with configuration.

        Args:
            config: LLM configuration to use for all calls
        """
        self.config = config
        logger.info(f"LLMClient initialized with model: {config.model}")

    async def _format_bad_output(
        self,
        ill_formed_output: str,
        format_instructions: str,
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
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}],
        }

        if self.config.api_key:
            completion_args["api_key"] = self.config.api_key
        if self.config.api_base:
            completion_args["api_base"] = self.config.api_base

        response = await acompletion(**completion_args)
        reformatted_output = response.choices[0].message.content
        assert isinstance(reformatted_output, str)
        logger.info(f"Reformatted output: {reformatted_output}")
        return reformatted_output

    async def _generate_with_schema(
        self,
        prompt: str,
        output_parser: PydanticOutputParser[T],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> T:
        """
        Generate structured output using schema-in-prompt approach with fallback.

        Args:
            prompt: The main prompt for the LLM
            output_parser: Parser for the expected output type
            temperature: Override default temperature for this call
            max_tokens: Override default max_tokens for this call

        Returns:
            Parsed and validated Pydantic model instance

        Raises:
            ValueError: If generation and parsing fail after fallback attempt
        """
        # Construct the full prompt with schema instructions
        format_instructions = output_parser.get_format_instructions()
        full_prompt = f"{prompt}\n\n{format_instructions}"

        logger.info(f"Full prompt {len(full_prompt)} characters")

        # Prepare completion arguments
        completion_args = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        # Add optional parameters
        if self.config.api_key:
            completion_args["api_key"] = self.config.api_key
        if self.config.api_base:
            completion_args["api_base"] = self.config.api_base

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
                reformatted_output = await self._format_bad_output(
                    ill_formed_output=content,
                    format_instructions=format_instructions,
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

    async def call_structured(
        self,
        prompt: str,
        output_type: Type[T],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> T:
        """
        Call LLM with structured output (async).

        Args:
            prompt: The main prompt for the LLM
            output_type: Pydantic model class for the expected output
            temperature: Override default temperature for this call
            max_tokens: Override default max_tokens for this call

        Returns:
            Parsed and validated Pydantic model instance
        """
        output_parser = PydanticOutputParser(output_type)
        return await self._generate_with_schema(
            prompt=prompt,
            output_parser=output_parser,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def call_structured_sync(
        self,
        prompt: str,
        output_type: Type[T],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> T:
        """
        Call LLM with structured output (sync).

        Args:
            prompt: The main prompt for the LLM
            output_type: Pydantic model class for the expected output
            temperature: Override default temperature for this call
            max_tokens: Override default max_tokens for this call

        Returns:
            Parsed and validated Pydantic model instance
        """
        try:
            loop = asyncio.get_running_loop()
            return asyncio.run_coroutine_threadsafe(
                self.call_structured(prompt, output_type, temperature, max_tokens), loop
            ).result()
        except RuntimeError:
            # No event loop running
            return asyncio.run(
                self.call_structured(prompt, output_type, temperature, max_tokens)
            )

    async def call_simple(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call LLM for simple text generation (async).

        Args:
            prompt: The prompt for the LLM
            temperature: Override default temperature for this call
            max_tokens: Override default max_tokens for this call

        Returns:
            Raw text response from the LLM
        """
        completion_args = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        if self.config.api_key:
            completion_args["api_key"] = self.config.api_key
        if self.config.api_base:
            completion_args["api_base"] = self.config.api_base

        response = await acompletion(**completion_args)
        content = response.choices[0].message.content

        if not content:
            raise ValueError("Empty response from LLM")

        # Ensure content is a string for type checking
        assert isinstance(content, str), f"Expected string content, got {type(content)}"
        return content.strip()

    def call_simple_sync(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call LLM for simple text generation (sync).

        Args:
            prompt: The prompt for the LLM
            temperature: Override default temperature for this call
            max_tokens: Override default max_tokens for this call

        Returns:
            Raw text response from the LLM
        """
        try:
            loop = asyncio.get_running_loop()
            return asyncio.run_coroutine_threadsafe(
                self.call_simple(prompt, temperature, max_tokens), loop
            ).result()
        except RuntimeError:
            # No event loop running
            return asyncio.run(self.call_simple(prompt, temperature, max_tokens))


# Backward compatibility and convenience functions
def create_llm_client(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    fallback_model: str = DEFAULT_BAD_OUTPUT_PROCESS_MODEL,
) -> LLMClient:
    """
    Create an LLM client with the given configuration.

    Args:
        model: LLM model to use
        temperature: Default temperature for generation
        max_tokens: Default max tokens for generation
        api_key: API key for LLM service
        api_base: Base URL for LLM service
        fallback_model: Model to use for reformatting bad outputs

    Returns:
        Configured LLMClient instance
    """
    config = LLMConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        api_base=api_base,
        fallback_model=fallback_model,
    )
    return LLMClient(config)


# Legacy function names for backward compatibility
async def call_llm_structured(
    prompt: str,
    output_type: Type[T],
    config: LLMConfig,
) -> T:
    """Legacy function - use LLMClient.call_structured instead."""
    client = LLMClient(config)
    return await client.call_structured(prompt, output_type)


async def call_llm_simple(
    prompt: str,
    config: LLMConfig,
) -> str:
    """Legacy function - use LLMClient.call_simple instead."""
    client = LLMClient(config)
    return await client.call_simple(prompt)
