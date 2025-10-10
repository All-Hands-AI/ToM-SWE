#!/usr/bin/env python3
"""
Multi-Model Clarity Classification Evaluation for ToM Agent

This script evaluates multiple LLM models' ability to classify whether problem statements
are clear or unclear by comparing:
- problem_statement (unclear/vague version)
- original_issue (clear/detailed version)

Tests models: GPT-5, GPT-5-Mini, GPT-5-Nano, Claude 3.5 Sonnet, Claude 4
Includes comprehensive cost tracking and performance analysis.
"""

import os
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datasets import load_from_disk, Dataset

from tom_swe.tom_agent import ToMAgent, ToMAgentConfig
from tom_swe.memory.local import LocalFileStore


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    model_id: str
    input_cost_per_1m: float  # Cost per 1M input tokens
    output_cost_per_1m: float  # Cost per 1M output tokens
    description: str


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
    model_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    evaluation_time_seconds: float


@dataclass
class ModelResults:
    """Aggregated results for a specific model."""

    model_name: str
    total_samples: int
    correct_classifications: int
    accuracy: float
    unclear_accuracy: float
    clear_accuracy: float
    total_cost: float
    avg_cost_per_consultation: float
    total_input_tokens: int
    total_output_tokens: int
    avg_evaluation_time: float
    results: List[EvalResult]


class MultiModelClarityEvaluator:
    """Evaluates multiple models on clarity classification task."""

    def __init__(self, dataset_path: str, num_samples: int = 100):
        """
        Initialize the evaluator.

        Args:
            dataset_path: Path to the stateful SWE dataset
            num_samples: Number of samples to evaluate per model
        """
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.results_by_model: Dict[str, ModelResults] = {}

        # Model configurations with updated pricing
        self.models = [
            ModelConfig(
                name="GPT-5-Nano",
                model_id="gpt-5-nano-2025-08-07",
                input_cost_per_1m=0.05,
                output_cost_per_1m=0.40,
                description="Ultra cost-effective GPT-5 variant",
            ),
            ModelConfig(
                name="GPT-5-Mini",
                model_id="gpt-5-mini-2025-08-07",
                input_cost_per_1m=0.25,
                output_cost_per_1m=2.00,
                description="Balanced cost/performance GPT-5",
            ),
            ModelConfig(
                name="GPT-5",
                model_id="gpt-5-2025-08-07",
                input_cost_per_1m=1.25,
                output_cost_per_1m=10.00,
                description="Full capability GPT-5",
            ),
            ModelConfig(
                name="Claude-3.5-Sonnet",
                model_id="litellm_proxy/claude-3-7-sonnet-20250219",
                input_cost_per_1m=3.00,
                output_cost_per_1m=15.00,
                description="Current baseline model",
            ),
            ModelConfig(
                name="Claude-4",
                model_id="litellm_proxy/claude-sonnet-4-20250514",
                input_cost_per_1m=15.00,
                output_cost_per_1m=75.00,
                description="Premium Claude model",
            ),
        ]

    def setup_agent(self, model_config: ModelConfig) -> ToMAgent:
        """Initialize the ToM agent with specific model."""
        print(f"🤖 Initializing ToM Agent with {model_config.name}...")

        # Check if API key is configured
        if not os.getenv("LITELLM_API_KEY"):
            raise ValueError(
                "❌ Please run 'uv run tom-config' to set up your LLM API credentials"
            )

        config = ToMAgentConfig(
            file_store=LocalFileStore(root="~/.openhands"),
            llm_model=model_config.model_id,
        )
        agent = ToMAgent(config)
        print(f"✅ Agent initialized successfully with {model_config.name}")
        return agent

    def _detect_question_suggestion(self, suggestion: str) -> bool:
        """
        Sophisticated detection of whether the ToM suggestion recommends asking questions.
        """
        suggestion_lower = suggestion.lower()

        # Patterns that strongly indicate asking questions is recommended
        question_asking_patterns = [
            r"ask\s+(?:for|about|the\s+user|user\s+to)",
            r"should\s+ask",
            r"need\s+to\s+ask",
            r"request\s+(?:more|additional|specific)",
            r"inquire\s+about",
            r"ask\s+for\s+clarification",
            r"seek\s+clarification",
            r"request\s+clarification",
            r"needs?\s+clarification",
            r"requires?\s+clarification",
            r"ask\s+for\s+(?:more|additional)\s+(?:information|details)",
            r"request\s+(?:more|additional)\s+(?:information|details)",
            r"need\s+(?:more|additional)\s+(?:information|details)",
            r"gather\s+(?:more|additional)\s+(?:information|details)",
            r"ask\s+(?:what|how|when|where|why)",
            r"question\s+(?:about|regarding)",
            r"find\s+out\s+(?:what|how|when|where|why)",
            r"(?:unclear|vague|ambiguous|insufficient).*(?:ask|question|clarify)",
            r"(?:missing|lacks?).*(?:ask|question|request)",
            r"not\s+enough.*(?:ask|question|request)",
        ]

        import re

        for pattern in question_asking_patterns:
            if re.search(pattern, suggestion_lower):
                return True

        # Simple keyword fallback
        question_keywords = [
            "ask for",
            "ask about",
            "ask the",
            "ask what",
            "ask how",
            "request more",
            "request additional",
            "needs clarification",
            "requires clarification",
            "seek clarification",
            "more information needed",
            "insufficient information",
            "unclear about",
            "vague about",
            "elaborate on",
        ]

        keyword_matches = sum(
            1 for keyword in question_keywords if keyword in suggestion_lower
        )
        return keyword_matches > 0

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and sample the stateful SWE dataset."""
        print(f"📂 Loading dataset from {self.dataset_path}...")

        # Check if we have a parquet file
        parquet_file = Path(self.dataset_path) / "test-00000-of-00001.parquet"
        if parquet_file.exists():
            print(f"📄 Loading from parquet file: {parquet_file}")
            dataset = Dataset.from_parquet(str(parquet_file))
        else:
            dataset = load_from_disk(self.dataset_path)

        # Convert to list and sample
        all_data = list(dataset)
        sampled_data = random.sample(all_data, min(self.num_samples, len(all_data)))

        print(f"✅ Loaded {len(sampled_data)} samples from dataset")
        return sampled_data

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for most models)."""
        return len(text) // 4

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model_config: ModelConfig
    ) -> float:
        """Calculate cost in USD for given token usage."""
        input_cost = (input_tokens / 1_000_000) * model_config.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * model_config.output_cost_per_1m
        return input_cost + output_cost

    def evaluate_statement(
        self,
        agent: ToMAgent,
        model_config: ModelConfig,
        statement: str,
        statement_type: str,
        original_issue: str,
        instance_id: str,
    ) -> EvalResult:
        """Evaluate a single statement for clarity."""
        print(
            f"🔍 Evaluating {statement_type} statement for {instance_id} with {model_config.name}..."
        )

        start_time = time.time()

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
            recommendation = agent.give_suggestions(
                user_id="",
                query=query,
                formatted_messages=formatted_messages,
            )

            evaluation_time = time.time() - start_time

            if recommendation:
                suggestion = recommendation.suggestions
                confidence = recommendation.confidence_score
            else:
                suggestion = "No recommendation provided"
                confidence = 0.0

            # Estimate token usage
            input_text = query + str(formatted_messages) + "system_prompt_content"
            input_tokens = self.estimate_tokens(input_text)
            output_tokens = self.estimate_tokens(suggestion)

            # Calculate cost
            cost = self.calculate_cost(input_tokens, output_tokens, model_config)

            # Check if ToM suggested asking questions
            suggested_questions = self._detect_question_suggestion(suggestion)

            # Determine if classification is correct
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
                model_name=model_config.name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                evaluation_time_seconds=evaluation_time,
            )

        except Exception as e:
            evaluation_time = time.time() - start_time
            print(f"❌ Error evaluating {instance_id} with {model_config.name}: {e}")
            return EvalResult(
                instance_id=instance_id,
                statement_type=statement_type,
                problem_statement=statement,
                original_issue=original_issue,
                tom_suggestion=f"Error: {e}",
                suggested_questions=False,
                confidence_score=0.0,
                correct_classification=False,
                model_name=model_config.name,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                evaluation_time_seconds=evaluation_time,
            )

    def evaluate_model(
        self, model_config: ModelConfig, data: List[Dict[str, Any]]
    ) -> ModelResults:
        """Evaluate a single model on the dataset."""
        print(f"\n{'='*60}")
        print(f"🚀 Evaluating {model_config.name}")
        print(f"📝 {model_config.description}")
        print(f"{'='*60}")

        agent = self.setup_agent(model_config)
        results = []

        # Evaluate each sample twice: once with unclear statement, once with clear
        for i, item in enumerate(data):
            instance_id = item["instance_id"]
            problem_statement = item["problem_statement"]  # unclear version
            original_issue = item["original_issue"]  # clear version

            print(f"\n📊 Progress: {i+1}/{len(data)} - {instance_id}")

            # Evaluate unclear statement
            unclear_result = self.evaluate_statement(
                agent,
                model_config,
                problem_statement,
                "unclear",
                original_issue,
                f"{instance_id}_unclear",
            )
            results.append(unclear_result)

            # Evaluate clear statement
            clear_result = self.evaluate_statement(
                agent,
                model_config,
                original_issue,
                "clear",
                original_issue,
                f"{instance_id}_clear",
            )
            results.append(clear_result)

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        # Calculate aggregated metrics
        correct_count = sum(1 for r in results if r.correct_classification)
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        unclear_results = [r for r in results if r.statement_type == "unclear"]
        clear_results = [r for r in results if r.statement_type == "clear"]

        unclear_accuracy = sum(
            1 for r in unclear_results if r.correct_classification
        ) / len(unclear_results)
        clear_accuracy = sum(
            1 for r in clear_results if r.correct_classification
        ) / len(clear_results)

        total_cost = sum(r.cost_usd for r in results)
        total_input_tokens = sum(r.input_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)
        avg_evaluation_time = sum(r.evaluation_time_seconds for r in results) / len(
            results
        )

        return ModelResults(
            model_name=model_config.name,
            total_samples=total_count,
            correct_classifications=correct_count,
            accuracy=accuracy,
            unclear_accuracy=unclear_accuracy,
            clear_accuracy=clear_accuracy,
            total_cost=total_cost,
            avg_cost_per_consultation=total_cost / total_count
            if total_count > 0
            else 0.0,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            avg_evaluation_time=avg_evaluation_time,
            results=results,
        )

    def run_full_evaluation(self) -> Dict[str, ModelResults]:
        """Run evaluation across all models."""
        print("🚀 Starting Multi-Model Clarity Classification Evaluation")
        print("=" * 80)

        # Load dataset once
        data = self.load_dataset()

        # Set seed for reproducible sampling across models
        random.seed(42)

        # Evaluate each model
        for model_config in self.models:
            try:
                model_results = self.evaluate_model(model_config, data)
                self.results_by_model[model_config.name] = model_results

                # Print interim results
                print(f"\n✅ {model_config.name} completed:")
                print(f"   Accuracy: {model_results.accuracy:.2%}")
                print(f"   Total Cost: ${model_results.total_cost:.4f}")
                print(
                    f"   Avg Cost/Consultation: ${model_results.avg_cost_per_consultation:.6f}"
                )

            except Exception as e:
                print(f"❌ Failed to evaluate {model_config.name}: {e}")
                continue

        return self.results_by_model

    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        if not self.results_by_model:
            return {}

        # Sort models by accuracy
        sorted_models = sorted(
            self.results_by_model.values(), key=lambda x: x.accuracy, reverse=True
        )

        # Find best cost-effectiveness (accuracy per dollar)
        cost_effectiveness = []
        for model in sorted_models:
            if model.total_cost > 0:
                effectiveness = model.accuracy / model.total_cost
                cost_effectiveness.append((model.model_name, effectiveness))

        cost_effectiveness.sort(key=lambda x: x[1], reverse=True)

        report = {
            "evaluation_summary": {
                "total_models_evaluated": len(self.results_by_model),
                "samples_per_model": self.num_samples,
                "total_consultations": sum(
                    m.total_samples for m in self.results_by_model.values()
                ),
            },
            "accuracy_ranking": [
                {
                    "rank": i + 1,
                    "model": model.model_name,
                    "accuracy": model.accuracy,
                    "unclear_accuracy": model.unclear_accuracy,
                    "clear_accuracy": model.clear_accuracy,
                    "total_cost": model.total_cost,
                    "cost_per_consultation": model.avg_cost_per_consultation,
                }
                for i, model in enumerate(sorted_models)
            ],
            "cost_effectiveness_ranking": [
                {
                    "rank": i + 1,
                    "model": name,
                    "cost_effectiveness_score": effectiveness,
                    "accuracy": next(
                        m.accuracy
                        for m in self.results_by_model.values()
                        if m.model_name == name
                    ),
                    "total_cost": next(
                        m.total_cost
                        for m in self.results_by_model.values()
                        if m.model_name == name
                    ),
                }
                for i, (name, effectiveness) in enumerate(cost_effectiveness)
            ],
            "detailed_results": {
                model_name: asdict(results)
                for model_name, results in self.results_by_model.items()
            },
        }

        return report

    def save_results(self, report: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"💾 Results saved to {output_file}")

    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print a summary of results."""
        print("\n" + "=" * 80)
        print("📊 MULTI-MODEL EVALUATION RESULTS")
        print("=" * 80)

        if not report.get("accuracy_ranking"):
            print("❌ No results to display")
            return

        print(
            f"📋 Models Evaluated: {report['evaluation_summary']['total_models_evaluated']}"
        )
        print(
            f"📋 Samples per Model: {report['evaluation_summary']['samples_per_model']}"
        )
        print(
            f"📋 Total Consultations: {report['evaluation_summary']['total_consultations']}"
        )

        print("\n🏆 ACCURACY RANKING:")
        print("-" * 80)
        for item in report["accuracy_ranking"]:
            print(
                f"{item['rank']}. {item['model']:<20} "
                f"Accuracy: {item['accuracy']:.2%} "
                f"Cost: ${item['total_cost']:.4f} "
                f"(${item['cost_per_consultation']:.6f}/consult)"
            )

        print("\n💰 COST-EFFECTIVENESS RANKING:")
        print("-" * 80)
        for item in report["cost_effectiveness_ranking"]:
            print(
                f"{item['rank']}. {item['model']:<20} "
                f"Score: {item['cost_effectiveness_score']:.2f} "
                f"(Accuracy: {item['accuracy']:.2%}, "
                f"Cost: ${item['total_cost']:.4f})"
            )


def main():
    """Run the multi-model clarity classification evaluation."""

    # Configuration
    dataset_path = (
        "/Users/xuhuizhou/Projects/ToM-SWE/data/stateful_swe/dataset_for_upload"
    )
    output_path = "/Users/xuhuizhou/Projects/ToM-SWE/stateful_swe/multi_model_clarity_results.json"
    num_samples = 100

    # Set random seed for reproducibility
    random.seed(42)

    try:
        # Run evaluation
        evaluator = MultiModelClarityEvaluator(dataset_path, num_samples)
        results_by_model = evaluator.run_full_evaluation()

        if not results_by_model:
            print("❌ No models were successfully evaluated")
            return 1

        # Generate comparison report
        report = evaluator.generate_comparison_report()

        # Print summary
        evaluator.print_summary(report)

        # Save results
        evaluator.save_results(report, output_path)

        print("\n✅ Multi-model evaluation completed successfully!")
        print(f"📁 Detailed results saved to: {output_path}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
