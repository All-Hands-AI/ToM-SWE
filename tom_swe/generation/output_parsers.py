"""
Output parsers for structured LLM responses.

Based on Sotopia's implementation with improvements for robust JSON parsing.
"""

import json
import logging
import re
from typing import Generic, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

try:
    import json_repair  # type: ignore

    JSON_REPAIR_AVAILABLE = True
except ImportError:
    json_repair = None  # type: ignore
    JSON_REPAIR_AVAILABLE = False

try:
    from tom_swe.logging_config import get_tom_swe_logger

    logger = get_tom_swe_logger(__name__)
except ImportError:
    # Fallback for standalone use
    logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class PydanticOutputParser(Generic[T]):
    """
    Parser that converts LLM output to Pydantic models with fallback mechanisms.

    Features:
    - Generates JSON schema instructions for prompts
    - Handles malformed JSON with repair attempts
    - Provides detailed error logging
    - Supports flexible parsing strategies
    """

    def __init__(self, pydantic_object: Type[T]):
        self.pydantic_object = pydantic_object

    def get_format_instructions(self) -> str:
        """
        Generate format instructions including JSON schema for the Pydantic model.

        Returns:
            String with clear formatting instructions and JSON schema
        """
        schema = self.pydantic_object.model_json_schema()

        # Generate a clean, readable schema description
        instructions = f"""Please respond with a valid JSON object that matches this exact schema:

{json.dumps(schema, indent=2)}

Important formatting requirements:
1. Response must be valid JSON
2. Include all required fields: {list(schema.get('required', []))}
3. Follow the exact field names and types specified
4. Do not include any text before or after the JSON object
5. Use double quotes for all strings in JSON
"""
        return instructions

    def parse(self, text: str) -> T:
        """
        Parse LLM output text into a Pydantic model instance.

        Args:
            text: Raw text output from LLM

        Returns:
            Validated Pydantic model instance

        Raises:
            ValueError: If parsing fails after all fallback attempts
        """
        # Clean the text
        text = text.strip()

        # Try to extract JSON from the text
        json_str = self._extract_json(text)

        # Attempt multiple parsing strategies
        parsed_data = None
        last_error: Optional[Exception] = None

        # Strategy 1: Direct JSON parsing
        try:
            parsed_data = json.loads(json_str)
            logger.debug("Successfully parsed JSON on first attempt")
        except json.JSONDecodeError as e:
            last_error = e
            logger.debug(f"Direct JSON parsing failed: {e}")

        # Strategy 2: Try json_repair if available
        if parsed_data is None and JSON_REPAIR_AVAILABLE and json_repair is not None:
            try:
                parsed_data = json_repair.loads(json_str)
                logger.debug("Successfully repaired and parsed JSON")
            except Exception as e:
                last_error = e
                logger.debug(f"JSON repair failed: {e}")
        elif parsed_data is None and not JSON_REPAIR_AVAILABLE:
            logger.debug("json_repair not available, skipping repair attempt")

        # Strategy 3: Try to fix common JSON issues manually
        if parsed_data is None:
            try:
                fixed_json = self._fix_common_json_issues(json_str)
                parsed_data = json.loads(fixed_json)
                logger.debug("Successfully parsed JSON after manual fixes")
            except Exception as e:
                last_error = e
                logger.debug(f"Manual JSON fix failed: {e}")

        if parsed_data is None:
            raise ValueError(
                f"Failed to parse JSON from LLM output: {last_error}"
            ) from None

        # Handle nested "properties" structure if present
        if isinstance(parsed_data, dict) and "properties" in parsed_data:
            # Sometimes LLMs return the schema structure instead of data
            if all(
                key in parsed_data["properties"]
                for key in self.pydantic_object.model_fields
            ):
                parsed_data = parsed_data["properties"]

        # Validate with Pydantic
        try:
            validated_result: T = self.pydantic_object.model_validate(parsed_data)
            return validated_result
        except ValidationError as e:
            logger.error(f"Pydantic validation failed: {e}")
            logger.error(f"Parsed data: {parsed_data}")
            raise ValueError(f"Failed to validate parsed data: {e}") from None

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain additional content."""
        # Look for JSON object boundaries
        start_idx = text.find("{")
        if start_idx == -1:
            return text

        # Find the matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        return text[start_idx:end_idx]

    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove trailing commas
        json_str = json_str.replace(",}", "}").replace(",]", "]")

        # Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')

        # Fix unquoted keys (basic attempt)
        json_str = re.sub(r"(\w+):", r'"\1":', json_str)

        return json_str

    def output_type(self) -> Type[T]:
        """Get the output type for type hints."""
        return self.pydantic_object
