"""Intent prediction module for ToM-SWE."""

import json
import logging
from typing import Dict, Any, Optional, List

from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class IntentPredictor:
    """Predicts developer intent from code using LLMs."""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the intent predictor.
        
        Args:
            llm_config: Configuration for the language model.
        """
        self.llm_client = LLMClient(llm_config)
        logger.info("IntentPredictor initialized")

    def predict_intent(self, code: str) -> Dict[str, Any]:
        """Predict the intent behind the code.
        
        Args:
            code: The code to analyze.
            
        Returns:
            Dict containing intent analysis.
        """
        logger.info("Predicting intent for code snippet")
        
        prompt = self._create_intent_prompt(code)
        response = self.llm_client.generate(prompt)
        
        try:
            intent_analysis = json.loads(response)
            logger.info("Intent prediction successful")
            return intent_analysis
        except json.JSONDecodeError:
            logger.warning("Failed to parse intent prediction response as JSON")
            return {
                "inferred_intent": response,
                "confidence": 0.5,
                "parse_error": "Failed to parse response as JSON"
            }

    def predict_changes(self, code: str, intent: str) -> Dict[str, Any]:
        """Predict changes needed to implement an intent.
        
        Args:
            code: The original code.
            intent: The developer's intent.
            
        Returns:
            Dict containing predicted changes.
        """
        logger.info(f"Predicting changes for intent: {intent}")
        
        prompt = self._create_changes_prompt(code, intent)
        response = self.llm_client.generate(prompt)
        
        try:
            changes = json.loads(response)
            logger.info("Change prediction successful")
            return changes
        except json.JSONDecodeError:
            logger.warning("Failed to parse change prediction response as JSON")
            return {
                "predicted_changes": [],
                "reasoning": response,
                "confidence": 0.5,
                "parse_error": "Failed to parse response as JSON"
            }

    def _create_intent_prompt(self, code: str) -> str:
        """Create a prompt for intent prediction.
        
        Args:
            code: The code to analyze.
            
        Returns:
            Prompt string.
        """
        return f"""
        You are an expert software engineer with a deep understanding of developer intent.
        Analyze the following code and infer the developer's intent.
        
        ```
        {code}
        ```
        
        Provide your analysis in JSON format with the following fields:
        - inferred_intent: A concise description of what the code is trying to accomplish
        - goals: A list of specific goals the developer is trying to achieve
        - assumptions: A list of assumptions the developer is making
        - mental_model: The developer's mental model of the problem
        - confidence: A number between 0 and 1 indicating your confidence in this analysis
        """

    def _create_changes_prompt(self, code: str, intent: str) -> str:
        """Create a prompt for predicting changes.
        
        Args:
            code: The original code.
            intent: The developer's intent.
            
        Returns:
            Prompt string.
        """
        return f"""
        You are an expert software engineer with a deep understanding of code modification.
        Given the following code and a new intent, predict what changes would be needed.
        
        Original code:
        ```
        {code}
        ```
        
        New intent: {intent}
        
        Provide your analysis in JSON format with the following fields:
        - predicted_changes: A list of objects, each with:
          - type: The type of change (add, modify, delete)
          - location: Where in the code the change should occur
          - description: A description of the change
        - reasoning: Your reasoning for these changes
        - confidence: A number between 0 and 1 indicating your confidence
        """