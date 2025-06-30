"""Utilities for configuring and using LLMs."""

import os
import json
from typing import Dict, List, Any, Optional


class LLMConfig:
    """Configuration for LLM integration."""
    
    def __init__(self, 
                 provider: str = "openai", 
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Initialize LLM configuration.

        Args:
            provider: The LLM provider (e.g., "openai", "anthropic").
            model: The model to use.
            api_key: API key for the provider.
            temperature: Temperature parameter for generation.
            max_tokens: Maximum tokens to generate.
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LLMConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration values.

        Returns:
            LLMConfig instance.
        """
        return cls(
            provider=config_dict.get("provider", "openai"),
            model=config_dict.get("model", "gpt-4"),
            api_key=config_dict.get("api_key"),
            temperature=config_dict.get("temperature", 0.7),
            max_tokens=config_dict.get("max_tokens", 1000)
        )
    
    def save(self, file_path: str) -> None:
        """
        Save configuration to file.

        Args:
            file_path: Path to save the configuration.
        """
        config_dict = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> 'LLMConfig':
        """
        Load configuration from file.

        Args:
            file_path: Path to load the configuration from.

        Returns:
            LLMConfig instance.
        """
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class LLMClient:
    """Client for interacting with LLMs."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client.

        Args:
            config: LLM configuration.
        """
        self.config = config
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the appropriate client based on the provider."""
        if self.config.provider == "openai":
            try:
                import openai
                openai.api_key = self.config.api_key
                self._client = openai
            except ImportError:
                print("OpenAI package not installed. Please install it with: pip install openai")
        elif self.config.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                print("Anthropic package not installed. Please install it with: pip install anthropic")
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using the configured LLM.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            The generated text.
        """
        if not self._client:
            raise ValueError("LLM client not initialized")
        
        if self.config.provider == "openai":
            response = self._client.ChatCompletion.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        
        elif self.config.provider == "anthropic":
            response = self._client.completions.create(
                model=self.config.model,
                prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.completion
        
        raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze code using the LLM.

        Args:
            code: The code to analyze.

        Returns:
            Analysis results.
        """
        prompt = f"""
        Please analyze the following code and provide insights:
        
        ```
        {code}
        ```
        
        Provide your analysis in JSON format with the following structure:
        {{
            "summary": "Brief summary of what the code does",
            "complexity": "low/medium/high",
            "potential_issues": ["issue1", "issue2", ...],
            "suggestions": ["suggestion1", "suggestion2", ...],
            "code_quality": "low/medium/high"
        }}
        """
        
        response = self.generate(prompt)
        
        # Extract JSON from response
        try:
            # Find JSON content (between curly braces)
            import re
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                # Fallback if no JSON found
                return {
                    "summary": "Failed to extract structured analysis",
                    "raw_response": response
                }
        except Exception as e:
            return {
                "error": f"Failed to parse LLM response: {str(e)}",
                "raw_response": response
            }