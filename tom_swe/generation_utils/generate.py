"""
Enhanced LLM generation with robust structured output support.

This module provides improved LLM calling mechanisms that include schema
in prompts and handle parsing with fallback mechanisms.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Type, TypeVar

from litellm import acompletion
from pydantic import BaseModel, validate_call

from .output_parsers import PydanticOutputParser

logger = logging.getLogger(__name__)

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


@validate_call
async def format_bad_output(
    ill_formed_output: str,
    format_instructions: str,
    model_name: str,
    use_fixed_model_version: bool = True,
) -> str:
    """
    Reformat ill-formed output to valid JSON using a fallback model.

    Args:
        ill_formed_output: The malformed output from the original LLM
        format_instructions: The format instructions that should be followed
        model_name: The model to use for reformatting
        use_fixed_model_version: Whether to use a fixed model version

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
    response = await acompletion(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": content}],
    )
    reformatted_output = response.choices[0].message.content
    assert isinstance(reformatted_output, str)
    logger.info(f"Reformatted output: {reformatted_output}")
    return reformatted_output


async def generate_with_schema(
    prompt: str,
    output_parser: PydanticOutputParser[T],
    config: Optional[LLMConfig] = None,
) -> T:
    """
    Generate structured output using schema-in-prompt approach with fallback.

    Args:
        prompt: The main prompt for the LLM
        output_parser: Parser for the expected output type
        config: LLM configuration parameters

    Returns:
        Parsed and validated Pydantic model instance

    Raises:
        ValueError: If generation and parsing fail after fallback attempt
    """
    if config is None:
        config = LLMConfig()

    # Construct the full prompt with schema instructions
    format_instructions = output_parser.get_format_instructions()
    full_prompt = f"{prompt}\n\n{format_instructions}"

    logger.debug(f"Full prompt length: {len(full_prompt)} characters")

    # Prepare completion arguments
    completion_args = {
        "model": config.model,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    # Add optional parameters
    if config.api_key:
        completion_args["api_key"] = config.api_key
    if config.api_base:
        completion_args["api_base"] = config.api_base

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
            reformatted_output = await format_bad_output(
                ill_formed_output=content,
                format_instructions=format_instructions,
                model_name=config.fallback_model,
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


async def call_llm_structured(
    prompt: str,
    output_type: Type[T],
    config: Optional[LLMConfig] = None,
) -> T:
    """
    Convenience function for structured LLM calls.

    This is a simplified interface that creates the output parser automatically.

    Args:
        prompt: The main prompt for the LLM
        output_type: Pydantic model class for the expected output
        config: LLM configuration parameters

    Returns:
        Parsed and validated Pydantic model instance
    """
    if config is None:
        config = LLMConfig()

    output_parser = PydanticOutputParser(output_type)
    return await generate_with_schema(
        prompt=prompt,
        output_parser=output_parser,
        config=config,
    )


async def call_llm_simple(
    prompt: str,
    config: Optional[LLMConfig] = None,
) -> str:
    """
    Simple LLM call for unstructured text generation.

    Args:
        prompt: The prompt for the LLM
        config: LLM configuration parameters

    Returns:
        Raw text response from the LLM
    """
    if config is None:
        config = LLMConfig()

    completion_args = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    if config.api_key:
        completion_args["api_key"] = config.api_key
    if config.api_base:
        completion_args["api_base"] = config.api_base

    response = await acompletion(**completion_args)
    content = response.choices[0].message.content

    if not content:
        raise ValueError("Empty response from LLM")

    # Ensure content is a string for type checking
    assert isinstance(content, str), f"Expected string content, got {type(content)}"
    return content.strip()


def create_output_parser(pydantic_class: Type[T]) -> PydanticOutputParser[T]:
    """
    Create an output parser for a Pydantic class.

    Args:
        pydantic_class: The Pydantic model class

    Returns:
        Output parser instance
    """
    return PydanticOutputParser(pydantic_class)
