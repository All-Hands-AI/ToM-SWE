"""Main ToM module that combines all components."""

from typing import Dict, Any, List, Optional
import json

from .analyzer import CodeAnalyzer
from .intent_predictor import IntentPredictor
from .code_explainer import CodeExplainer


class LLMConfig:
    """Configuration for LLM."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-3.5-turbo"):
        self.provider = provider
        self.model = model


class ToMModule:
    """Main Theory of Mind module for software engineering."""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """Initialize the ToM module.
        
        Args:
            llm_config: Configuration for the language model.
        """
        self.llm_config = llm_config or LLMConfig()
        self.analyzer = CodeAnalyzer()
        self.intent_predictor = IntentPredictor()
        self.code_explainer = CodeExplainer()
        self.llm_client = self.analyzer.intent_predictor.llm_client
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code and return comprehensive analysis.
        
        Args:
            code: Code to analyze.
            
        Returns:
            Analysis results.
        """
        # Get basic metrics
        metrics = self.analyzer.calculate_metrics(code)
        
        # Get LLM analysis
        llm_analysis = self.analyzer.analyze(code)
        
        return {
            **metrics,
            "llm_analysis": llm_analysis
        }
    
    def understand_intent(self, code: str) -> Dict[str, Any]:
        """Understand the intent behind the code.
        
        Args:
            code: Code to analyze.
            
        Returns:
            Intent analysis results.
        """
        return self.intent_predictor.predict_intent(code)
    
    def predict_changes(self, code: str, intent: str) -> Dict[str, Any]:
        """Predict changes needed to achieve the given intent.
        
        Args:
            code: Current code.
            intent: Desired intent.
            
        Returns:
            Predicted changes.
        """
        return self.intent_predictor.predict_changes(code, intent)
    
    def explain_code(self, code: str, audience: str = "general") -> str:
        """Explain code for a specific audience.
        
        Args:
            code: Code to explain.
            audience: Target audience.
            
        Returns:
            Code explanation.
        """
        explanations = self.code_explainer.explain_code(code, audience)
        if explanations:
            return explanations[0].get("explanation", "")
        return ""
    
    def suggest_improvements(self, code: str) -> List[Dict[str, Any]]:
        """Suggest improvements for the code.
        
        Args:
            code: Code to improve.
            
        Returns:
            List of improvement suggestions.
        """
        suggestions = self.analyzer.suggest_improvements(code)
        return suggestions.get("suggestions", [])