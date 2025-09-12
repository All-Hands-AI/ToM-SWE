#!/usr/bin/env python3
"""
Clarity Classification Evaluation for ToM Agent

This script evaluates the ToM agent's ability to classify whether problem statements
are clear or unclear by comparing:
- problem_statement (unclear/vague version)
- original_issue (clear/detailed version)

The evaluation checks if the ToM agent correctly identifies which statements need
clarification and suggests asking questions for unclear statements.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datasets import load_from_disk, Dataset

from tom_swe.tom_agent import ToMAgent, ToMAgentConfig
from tom_swe.memory.local import LocalFileStore


@dataclass
class EvalResult:
    """Result of a single evaluation."""

    instance_id: str
    statement_type: str  # "clear" or "unclear"
    problem_statement: str
    original_issue: str
    tom_suggestion: str
    suggested_questions: bool  # True if ToM suggested asking questions
    confidence_score: float
    correct_classification: bool


class ClarityEvaluator:
    """Evaluates ToM agent's clarity classification accuracy."""

    def __init__(self, dataset_path: str, num_samples: int = 50):
        """
        Initialize the evaluator.

        Args:
            dataset_path: Path to the stateful SWE dataset
            num_samples: Number of samples to evaluate (default: 50)
        """
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.agent: Optional[ToMAgent] = None
        self.results: List[EvalResult] = []

    def setup_agent(self) -> None:
        """Initialize the ToM agent."""
        print("🤖 Initializing ToM Agent...")

        # Check if API key is configured
        if not os.getenv("LITELLM_API_KEY"):
            raise ValueError(
                "❌ Please run 'uv run tom-config' to set up your LLM API credentials"
            )

        config = ToMAgentConfig(
            file_store=LocalFileStore(root="~/.openhands"),
            llm_model="litellm_proxy/claude-sonnet-4-20250514",
        )
        self.agent = ToMAgent(config)
        print("✅ Agent initialized successfully")

    def _detect_question_suggestion(self, suggestion: str) -> bool:
        """
        Sophisticated detection of whether the ToM suggestion recommends asking questions.

        Uses semantic analysis to distinguish between:
        1. Actually suggesting to ask questions (TRUE positive)
        2. Mentioning question-related words in other contexts (FALSE positive)

        Args:
            suggestion: The ToM agent's suggestion text

        Returns:
            bool: True if the suggestion recommends asking questions
        """
        suggestion_lower = suggestion.lower()

        # Patterns that strongly indicate asking questions is recommended
        question_asking_patterns = [
            # Direct instructions to ask
            r"ask\s+(?:for|about|the\s+user|user\s+to)",
            r"should\s+ask",
            r"need\s+to\s+ask",
            r"request\s+(?:more|additional|specific)",
            r"inquire\s+about",
            # Suggestions for clarification
            r"ask\s+for\s+clarification",
            r"seek\s+clarification",
            r"request\s+clarification",
            r"needs?\s+clarification",
            r"requires?\s+clarification",
            # Information gathering suggestions
            r"ask\s+for\s+(?:more|additional)\s+(?:information|details)",
            r"request\s+(?:more|additional)\s+(?:information|details)",
            r"need\s+(?:more|additional)\s+(?:information|details)",
            r"gather\s+(?:more|additional)\s+(?:information|details)",
            # Specific question suggestions
            r"ask\s+(?:what|how|when|where|why)",
            r"question\s+(?:about|regarding)",
            r"find\s+out\s+(?:what|how|when|where|why)",
            # Problem identification requiring questions
            r"(?:unclear|vague|ambiguous|insufficient).*(?:ask|question|clarify)",
            r"(?:missing|lacks?).*(?:ask|question|request)",
            r"not\s+enough.*(?:ask|question|request)",
        ]

        # Check for question-asking patterns
        import re

        for pattern in question_asking_patterns:
            if re.search(pattern, suggestion_lower):
                return True

        # Patterns that indicate the issue is clear/actionable (negative indicators)
        clear_indicators = [
            r"(?:clear|actionable|well-defined|specific|detailed|comprehensive)",
            r"user\s+has\s+provided",
            r"issue\s+(?:provides|includes|contains)",
            r"(?:concrete|specific)\s+(?:examples?|code|steps)",
            r"reproduction\s+steps",
            r"error\s+messages?",
            r"no\s+(?:need|reason)\s+(?:to\s+ask|for.*clarification)",
        ]

        # If suggestion clearly indicates the issue is actionable, it's likely not asking for questions
        clear_indicator_count = sum(
            1 for pattern in clear_indicators if re.search(pattern, suggestion_lower)
        )

        # Simple keyword fallback (but with context awareness)
        question_keywords = [
            "ask for",
            "ask about",
            "ask the",
            "ask what",
            "ask how",
            "ask when",
            "ask where",
            "ask why",
            "request more",
            "request additional",
            "request specific",
            "needs clarification",
            "requires clarification",
            "seek clarification",
            "more information needed",
            "additional information",
            "insufficient information",
            "unclear about",
            "vague about",
            "ambiguous about",
            "elaborate on",
            "specify what",
            "specify how",
        ]

        keyword_matches = sum(
            1 for keyword in question_keywords if keyword in suggestion_lower
        )

        # Decision logic:
        # - If we found strong question-asking patterns, return True
        # - If we found many clear indicators but no question patterns, return False
        # - Use keyword matches as a tiebreaker, but weight against clear indicators

        if keyword_matches > 0:
            # If there are more clear indicators than keyword matches, likely false positive
            if clear_indicator_count > keyword_matches:
                return False
            return True

        return False

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and sample the stateful SWE dataset."""
        print(f"📂 Loading dataset from {self.dataset_path}...")

        # Check if we have a parquet file
        parquet_file = Path(self.dataset_path) / "test-00000-of-00001.parquet"
        if parquet_file.exists():
            print(f"📄 Loading from parquet file: {parquet_file}")
            dataset = Dataset.from_parquet(str(parquet_file))
        else:
            # Try loading from disk as before
            dataset = load_from_disk(self.dataset_path)

        # Convert to list and sample
        all_data = list(dataset)
        sampled_data = random.sample(all_data, min(self.num_samples, len(all_data)))

        print(f"✅ Loaded {len(sampled_data)} samples from dataset")
        return sampled_data

    def evaluate_statement(
        self, statement: str, statement_type: str, original_issue: str, instance_id: str
    ) -> EvalResult:
        """
        Evaluate a single statement for clarity.

        Args:
            statement: The problem statement to evaluate
            statement_type: "clear" or "unclear"
            instance_id: Instance identifier

        Returns:
            EvalResult with classification results
        """
        print(f"🔍 Evaluating {statement_type} statement for {instance_id}...")

        # Create simple user message format
        formatted_messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """<uploaded_files>/workspace/
</uploaded_files>

Relevant python code files are in the directory /workspace/.
DON'T modify the testing logic or any of the tests in any way!
Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages."""
                        + statement,
                    }
                ],
            }
        ]

        # Create a query simulating SWE agent asking for guidance
        query = f"I am an SWE agent. I need to understand the user's intent and expectations for this issue. I need to consult ToM agent about the user's message: {statement}"

        try:
            # Get ToM agent suggestion
            if self.agent is None:
                raise ValueError("Agent not initialized. Call setup_agent() first.")

            recommendation = self.agent.give_suggestions(
                user_id="",
                query=query,
                formatted_messages=formatted_messages,
            )

            if recommendation:
                suggestion = recommendation.suggestions
                confidence = recommendation.confidence_score
            else:
                suggestion = "No recommendation provided"
                confidence = 0.0

            # Check if ToM suggested asking questions (sophisticated semantic analysis)
            suggested_questions = self._detect_question_suggestion(suggestion)

            # Determine if classification is correct
            # For unclear statements, ToM should suggest asking questions
            # For clear statements, ToM should not suggest asking questions
            if statement_type == "unclear":
                correct_classification = suggested_questions
            else:  # clear
                correct_classification = not suggested_questions

            return EvalResult(
                instance_id=instance_id,
                statement_type=statement_type,
                problem_statement=statement,
                original_issue=original_issue,
                tom_suggestion=suggestion,
                suggested_questions=suggested_questions,
                confidence_score=confidence,
                correct_classification=correct_classification,
            )

        except Exception as e:
            print(f"❌ Error evaluating {instance_id}: {e}")
            return EvalResult(
                instance_id=instance_id,
                statement_type=statement_type,
                problem_statement=statement,
                original_issue=statement,
                tom_suggestion=f"Error: {e}",
                suggested_questions=False,
                confidence_score=0.0,
                correct_classification=False,
            )

    def run_evaluation(self) -> Tuple[float, Dict[str, Any]]:
        """
        Run the full evaluation.

        Returns:
            Tuple of (accuracy, detailed_results)
        """
        print("🚀 Starting Clarity Classification Evaluation")
        print("=" * 60)

        # Setup
        self.setup_agent()
        data = self.load_dataset()

        self.results = []

        # Evaluate each sample twice: once with unclear statement, once with clear
        for i, item in enumerate(data):
            instance_id = item["instance_id"]
            problem_statement = item["problem_statement"]  # unclear version
            original_issue = item["original_issue"]  # clear version

            print(f"\n📊 Progress: {i+1}/{len(data)} - {instance_id}")

            # Evaluate unclear statement
            unclear_result = self.evaluate_statement(
                problem_statement, "unclear", original_issue, f"{instance_id}_unclear"
            )
            self.results.append(unclear_result)

            # Evaluate clear statement
            clear_result = self.evaluate_statement(
                original_issue, "clear", original_issue, f"{instance_id}_clear"
            )
            self.results.append(clear_result)

        # Calculate accuracy
        correct_count = sum(
            1 for result in self.results if result.correct_classification
        )
        total_count = len(self.results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        # Generate detailed results
        unclear_results = [r for r in self.results if r.statement_type == "unclear"]
        clear_results = [r for r in self.results if r.statement_type == "clear"]

        unclear_accuracy = sum(
            1 for r in unclear_results if r.correct_classification
        ) / len(unclear_results)
        clear_accuracy = sum(
            1 for r in clear_results if r.correct_classification
        ) / len(clear_results)

        detailed_results = {
            "overall_accuracy": accuracy,
            "unclear_accuracy": unclear_accuracy,
            "clear_accuracy": clear_accuracy,
            "total_samples": total_count,
            "unclear_samples": len(unclear_results),
            "clear_samples": len(clear_results),
            "correct_classifications": correct_count,
            "results": [
                {
                    "instance_id": r.instance_id,
                    "statement_type": r.statement_type,
                    "problem_statement": r.problem_statement,
                    "original_issue": r.original_issue,
                    "suggested_questions": r.suggested_questions,
                    "confidence_score": r.confidence_score,
                    "correct_classification": r.correct_classification,
                    "tom_suggestion": r.tom_suggestion,
                }
                for r in self.results
            ],
        }

        return accuracy, detailed_results

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_file}")


def main():
    """Run the clarity classification evaluation."""

    # Configuration
    dataset_path = (
        "/Users/xuhuizhou/Projects/ToM-SWE/data/stateful_swe/dataset_for_upload"
    )
    output_path = (
        "/Users/xuhuizhou/Projects/ToM-SWE/stateful_swe/clarity_eval_results.json"
    )
    num_samples = 50  # Sample 5 data points for testing

    # Set random seed for reproducibility
    random.seed(42)

    try:
        # Run evaluation
        evaluator = ClarityEvaluator(dataset_path, num_samples)
        accuracy, detailed_results = evaluator.run_evaluation()

        # Print summary
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)
        print(f"Overall Accuracy: {accuracy:.2%}")
        print(f"Unclear Statement Accuracy: {detailed_results['unclear_accuracy']:.2%}")
        print(f"Clear Statement Accuracy: {detailed_results['clear_accuracy']:.2%}")
        print(f"Total Samples: {detailed_results['total_samples']}")
        print(f"Correct Classifications: {detailed_results['correct_classifications']}")

        # Save results
        evaluator.save_results(detailed_results, output_path)

        print("\n✅ Evaluation completed successfully!")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
