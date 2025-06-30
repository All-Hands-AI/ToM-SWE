"""Theory of Mind module for software engineering."""

import os
import json
from typing import Dict, List, Any, Optional

from .utils import calculate_complexity
from .utils.configure_llm import LLMConfig, LLMClient


class ToMModule:
    """Theory of Mind module for software engineering."""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """
        Initialize the ToM module.

        Args:
            llm_config: Configuration for the LLM.
        """
        self.llm_config = llm_config or LLMConfig()
        self.llm_client = LLMClient(self.llm_config)
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze code using Theory of Mind principles.

        Args:
            code: The code to analyze.

        Returns:
            Analysis results.
        """
        # Calculate basic complexity metrics
        complexity_metrics = calculate_complexity(code)
        
        # Get LLM analysis
        llm_analysis = self.llm_client.analyze_code(code)
        
        # Combine analyses
        combined_analysis = {
            **complexity_metrics,
            "llm_analysis": llm_analysis
        }
        
        return combined_analysis
    
    def understand_intent(self, code: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Understand the intent behind the code.

        Args:
            code: The code to analyze.
            context: Additional context about the code.

        Returns:
            Intent analysis results.
        """
        prompt = f"""
        Please analyze the following code and infer the developer's intent:
        
        ```
        {code}
        ```
        
        {f"Additional context: {context}" if context else ""}
        
        Provide your analysis in JSON format with the following structure:
        {{
            "inferred_intent": "What the developer was trying to accomplish",
            "goals": ["goal1", "goal2", ...],
            "assumptions": ["assumption1", "assumption2", ...],
            "mental_model": "Description of the developer's mental model",
            "confidence": 0.0-1.0
        }}
        """
        
        response = self.llm_client.generate(prompt)
        
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
                    "inferred_intent": "Failed to extract structured analysis",
                    "raw_response": response
                }
        except Exception as e:
            return {
                "error": f"Failed to parse LLM response: {str(e)}",
                "raw_response": response
            }
    
    def predict_changes(self, code: str, intent: str) -> Dict[str, Any]:
        """
        Predict changes that might be made to the code based on intent.

        Args:
            code: The current code.
            intent: The stated intent for changes.

        Returns:
            Predicted changes.
        """
        prompt = f"""
        Given the following code and the developer's stated intent, predict what changes they might make:
        
        CODE:
        ```
        {code}
        ```
        
        INTENT: {intent}
        
        Provide your prediction in JSON format with the following structure:
        {{
            "predicted_changes": [
                {{
                    "type": "add/modify/delete",
                    "location": "description of where in the code",
                    "description": "what will be changed"
                }},
                ...
            ],
            "reasoning": "explanation for these predictions",
            "confidence": 0.0-1.0
        }}
        """
        
        response = self.llm_client.generate(prompt)
        
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
                    "predicted_changes": [],
                    "reasoning": "Failed to extract structured analysis",
                    "raw_response": response
                }
        except Exception as e:
            return {
                "error": f"Failed to parse LLM response: {str(e)}",
                "raw_response": response
            }
    
    def explain_code(self, code: str, audience: str = "developer") -> str:
        """
        Generate an explanation of the code for a specific audience.

        Args:
            code: The code to explain.
            audience: The target audience (e.g., "developer", "manager", "beginner").

        Returns:
            Explanation of the code.
        """
        prompt = f"""
        Please explain the following code for a {audience}:
        
        ```
        {code}
        ```
        
        Tailor your explanation to be appropriate for a {audience}'s level of technical understanding.
        """
        
        return self.llm_client.generate(prompt)
    
    def suggest_improvements(self, code: str) -> List[Dict[str, Any]]:
        """
        Suggest improvements to the code.

        Args:
            code: The code to improve.

        Returns:
            List of suggested improvements.
        """
        prompt = f"""
        Please suggest improvements for the following code:
        
        ```
        {code}
        ```
        
        Provide your suggestions in JSON format with the following structure:
        {{
            "suggestions": [
                {{
                    "type": "performance/readability/security/etc.",
                    "description": "description of the suggestion",
                    "code_example": "example code implementing the suggestion",
                    "priority": "high/medium/low"
                }},
                ...
            ]
        }}
        """
        
        response = self.llm_client.generate(prompt)
        
        # Extract JSON from response
        try:
            # Find JSON content (between curly braces)
            import re
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                return data.get("suggestions", [])
            else:
                # Fallback if no JSON found
                return [{
                    "type": "general",
                    "description": "Failed to extract structured suggestions",
                    "raw_response": response,
                    "priority": "medium"
                }]
        except Exception as e:
            return [{
                "type": "error",
                "description": f"Failed to parse LLM response: {str(e)}",
                "raw_response": response,
                "priority": "low"
            }]