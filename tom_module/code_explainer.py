"""Code explanation module for ToM-SWE."""

import json
import logging
from typing import Dict, Any, Optional, List

from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CodeExplainer:
    """Explains code using Theory of Mind principles."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the code explainer.
        
        Args:
            llm_config: Configuration for the language model.
        """
        self.llm_client = LLMClient(llm_config)
        logger.info("CodeExplainer initialized")

    def explain(self, code: str, audience: str = "general") -> str:
        """Explain code for a specific audience.
        
        Args:
            code: The code to explain.
            audience: Target audience (e.g., "beginner", "expert").
            
        Returns:
            String explanation of the code.
        """
        logger.info(f"Explaining code for audience: {audience}")
        
        prompt = self._create_explanation_prompt(code, audience)
        explanation = self.llm_client.generate(prompt)
        
        logger.info("Code explanation generated")
        return explanation

    def suggest_improvements(self, code: str) -> List[Dict[str, Any]]:
        """Suggest improvements for the code.
        
        Args:
            code: The code to improve.
            
        Returns:
            List of improvement suggestions.
        """
        logger.info("Suggesting improvements for code")
        
        prompt = self._create_improvements_prompt(code)
        response = self.llm_client.generate(prompt)
        
        try:
            suggestions = json.loads(response)
            return suggestions.get("suggestions", [])
        except json.JSONDecodeError:
            logger.warning("Failed to parse improvement suggestions as JSON")
            return [{
                "type": "general",
                "description": response,
                "priority": "medium"
            }]

    def _create_explanation_prompt(self, code: str, audience: str) -> str:
        """Create a prompt for code explanation.
        
        Args:
            code: The code to explain.
            audience: Target audience.
            
        Returns:
            Prompt string.
        """
        audience_guidance = {
            "beginner": "Explain in simple terms, avoiding jargon. Focus on what the code does rather than how.",
            "intermediate": "Provide a balanced explanation of what the code does and how it works.",
            "expert": "Focus on the implementation details, design patterns, and potential edge cases.",
            "general": "Provide a clear explanation suitable for a general programming audience."
        }.get(audience.lower(), "Provide a clear explanation suitable for a general programming audience.")
        
        return f"""
        You are an expert software engineer with a talent for explaining code.
        Explain the following code for a {audience} audience.
        
        {audience_guidance}
        
        ```
        {code}
        ```
        
        Provide your explanation in clear, concise language.
        """

    def _create_improvements_prompt(self, code: str) -> str:
        """Create a prompt for suggesting improvements.
        
        Args:
            code: The code to improve.
            
        Returns:
            Prompt string.
        """
        return f"""
        You are an expert software engineer with a deep understanding of code quality.
        Analyze the following code and suggest improvements.
        
        ```
        {code}
        ```
        
        Provide your suggestions in JSON format with the following structure:
        {{
            "suggestions": [
                {{
                    "type": "The type of suggestion (e.g., performance, readability, security)",
                    "description": "A clear description of the improvement",
                    "code_example": "An example of the improved code (if applicable)",
                    "priority": "The priority of this suggestion (high, medium, low)"
                }}
            ]
        }}
        
        Focus on the most important improvements first.
        """