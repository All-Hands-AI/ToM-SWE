"""LLM client for ToM-SWE."""

import os
import json
import logging
from typing import Dict, Any, Optional, List

import openai

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with language models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the LLM client.
        
        Args:
            config: Configuration for the language model.
        """
        self.config = config or {}
        self.provider = self.config.get("provider", "openai")
        self.model = self.config.get("model", "gpt-4")
        self.temperature = self.config.get("temperature", 0.2)
        self.max_tokens = self.config.get("max_tokens", 1000)
        
        # Set up API keys
        if self.provider == "openai":
            openai.api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not openai.api_key:
                logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        logger.info(f"LLMClient initialized with provider: {self.provider}, model: {self.model}")

    def generate(self, prompt: str) -> str:
        """Generate text using the language model.
        
        Args:
            prompt: The prompt to send to the model.
            
        Returns:
            Generated text.
        """
        logger.info(f"Generating text with {self.provider} model: {self.model}")
        
        if self.provider == "openai":
            return self._generate_openai(prompt)
        else:
            logger.warning(f"Unsupported provider: {self.provider}, falling back to mock response")
            return self._generate_mock(prompt)

    def analyze_code(self, code: str, task: str) -> Dict[str, Any]:
        """Analyze code using the language model.
        
        Args:
            code: The code to analyze.
            task: The analysis task (e.g., "complexity", "intent").
            
        Returns:
            Analysis results.
        """
        prompt = self._create_analysis_prompt(code, task)
        response = self.generate(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse analysis response as JSON")
            return {"error": "Failed to parse response", "raw_response": response}

    def _generate_openai(self, prompt: str) -> str:
        """Generate text using OpenAI's API.
        
        Args:
            prompt: The prompt to send to the model.
            
        Returns:
            Generated text.
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            return f"Error: {str(e)}"

    def _generate_mock(self, prompt: str) -> str:
        """Generate mock text for testing.
        
        Args:
            prompt: The prompt to send to the model.
            
        Returns:
            Generated text.
        """
        if "intent" in prompt.lower():
            return json.dumps({
                "inferred_intent": "This code appears to be a simple function that prints a greeting.",
                "goals": ["Display text to the user"],
                "assumptions": ["Standard output is available"],
                "mental_model": "Simple procedural programming",
                "confidence": 0.9
            })
        elif "explain" in prompt.lower():
            return "This code defines a function that prints 'Hello, world!' to the console."
        elif "improve" in prompt.lower():
            return json.dumps({
                "suggestions": [
                    {
                        "type": "documentation",
                        "description": "Add a docstring to explain the function's purpose",
                        "code_example": "def hello_world():\n    \"\"\"Print a greeting message.\"\"\"\n    print(\"Hello, world!\")",
                        "priority": "medium"
                    }
                ]
            })
        else:
            return "Mock response for: " + prompt[:50] + "..."

    def _create_analysis_prompt(self, code: str, task: str) -> str:
        """Create a prompt for code analysis.
        
        Args:
            code: The code to analyze.
            task: The analysis task.
            
        Returns:
            Prompt string.
        """
        return f"""
        You are an expert software engineer analyzing code.
        
        Code to analyze:
        ```
        {code}
        ```
        
        Task: {task}
        
        Provide your analysis in JSON format.
        """