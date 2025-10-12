#!/usr/bin/env python3
"""
Parallel Multi-Model Clarity Classification Evaluation for ToM Agent

This script evaluates multiple LLM models with full parallelization:
- All models run in parallel
- All API calls per model run in parallel
- Significant speedup for large evaluations

Tests models: GPT-5, GPT-5-Mini, GPT-5-Nano, Claude 3.7 Sonnet, Claude 4 Sonnet
"""

import json
import random
import time
import asyncio
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict
from datasets import load_from_disk, Dataset
import threading

from tom_swe.tom_agent import ToMAgent, ToMAgentConfig
from tom_swe.memory.local import LocalFileStore
from multi_model_clarity_eval import ModelConfig, EvalResult, ModelResults


class AsyncToMWrapper:
    """Async wrapper for ToM agent to enable parallel API calls."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._agent = None
        self._lock = threading.Lock()

    def _get_agent(self) -> ToMAgent:
        """Thread-safe agent initialization."""
        if self._agent is None:
            with self._lock:
                if self._agent is None:
                    print(f"🤖 Initializing ToM Agent with {self.model_config.name}...")
                    config = ToMAgentConfig(
                        file_store=LocalFileStore(root="~/.openhands"),
                        llm_model=self.model_config.model_id,
                    )
                    self._agent = ToMAgent(config)
                    print(f"✅ Agent {self.model_config.name} initialized")
        return self._agent

    async def give_suggestions_async(
        self, user_id: str, query: str, formatted_messages: List[Dict[str, Any]]
    ) -> Any:
        """Async wrapper for give_suggestions using thread pool."""
        loop = asyncio.get_event_loop()

        def _sync_call():
            agent = self._get_agent()
            return agent.give_suggestions(
                user_id=user_id, query=query, formatted_messages=formatted_messages
            )

        # Run sync ToM agent call in thread pool to avoid blocking
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_sync_call)
            return await loop.run_in_executor(None, future.result)


class ParallelMultiModelEvaluator:
    """Parallel evaluator for multiple models with async API calls."""

    def __init__(
        self, dataset_path: str, num_samples: int = 100, max_concurrent_calls: int = 10
    ):
        """
        Initialize the parallel evaluator.

        Args:
            dataset_path: Path to the stateful SWE dataset
            num_samples: Number of samples to evaluate per model
            max_concurrent_calls: Max concurrent API calls per model
        """
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.max_concurrent_calls = max_concurrent_calls
        self.results_by_model: Dict[str, ModelResults] = {}

        # Model configurations
        self.models = [
            ModelConfig(
                name="GPT-5-Nano",
                model_id="gpt-5-nano",
                input_cost_per_1m=0.05,
                output_cost_per_1m=0.40,
                description="Ultra cost-effective GPT-5 variant",
            ),
            ModelConfig(
                name="GPT-5-Mini",
                model_id="gpt-5-mini",
                input_cost_per_1m=0.25,
                output_cost_per_1m=2.00,
                description="Balanced cost/performance GPT-5",
            ),
            ModelConfig(
                name="GPT-5",
                model_id="gpt-5",
                input_cost_per_1m=1.25,
                output_cost_per_1m=10.00,
                description="Full capability GPT-5",
            ),
            ModelConfig(
                name="Claude-3.5-Sonnet",
                model_id="litellm_proxy/claude-3-5-sonnet-20241022",
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

    def _detect_question_suggestion(self, suggestion: str) -> bool:
        """Detect if ToM suggestion recommends asking questions."""
        suggestion_lower = suggestion.lower()

        # Key patterns that indicate asking questions
        question_patterns = [
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

        return any(pattern in suggestion_lower for pattern in question_patterns)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and sample the dataset."""
        print(f"📂 Loading dataset from {self.dataset_path}...")

        parquet_file = Path(self.dataset_path) / "test-00000-of-00001.parquet"
        if parquet_file.exists():
            dataset = Dataset.from_parquet(str(parquet_file))
        else:
            dataset = load_from_disk(self.dataset_path)

        all_data = list(dataset)
        sampled_data = random.sample(all_data, min(self.num_samples, len(all_data)))

        print(f"✅ Loaded {len(sampled_data)} samples from dataset")
        return sampled_data

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation."""
        return len(text) // 4

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model_config: ModelConfig
    ) -> float:
        """Calculate cost in USD."""
        input_cost = (input_tokens / 1_000_000) * model_config.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * model_config.output_cost_per_1m
        return input_cost + output_cost

    async def evaluate_statement_async(
        self,
        agent_wrapper: AsyncToMWrapper,
        statement: str,
        statement_type: str,
        original_issue: str,
        instance_id: str,
    ) -> EvalResult:
        """Async evaluation of a single statement."""
        start_time = time.time()

        # Create formatted messages
        formatted_messages = [
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

        query = f"I am an SWE agent. I need to understand the user's intent and expectations for this issue. I need to consult ToM agent about the user's message: {statement}"

        try:
            # Async API call
            recommendation = await agent_wrapper.give_suggestions_async(
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

            # Token and cost estimation
            input_text = query + str(formatted_messages) + "system_prompt_content"
            input_tokens = self.estimate_tokens(input_text)
            output_tokens = self.estimate_tokens(suggestion)
            cost = self.calculate_cost(
                input_tokens, output_tokens, agent_wrapper.model_config
            )

            # Check classification
            suggested_questions = self._detect_question_suggestion(suggestion)

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
                model_name=agent_wrapper.model_config.name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                evaluation_time_seconds=evaluation_time,
            )

        except Exception as e:
            evaluation_time = time.time() - start_time
            print(
                f"❌ Error evaluating {instance_id} with {agent_wrapper.model_config.name}: {e}"
            )

            return EvalResult(
                instance_id=instance_id,
                statement_type=statement_type,
                problem_statement=statement,
                original_issue=original_issue,
                tom_suggestion=f"Error: {e}",
                suggested_questions=False,
                confidence_score=0.0,
                correct_classification=False,
                model_name=agent_wrapper.model_config.name,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                evaluation_time_seconds=evaluation_time,
            )

    async def evaluate_model_async(
        self, model_config: ModelConfig, data: List[Dict[str, Any]]
    ) -> ModelResults:
        """Async evaluation of a single model with parallel API calls."""
        print(f"\n🚀 Starting {model_config.name} evaluation...")

        agent_wrapper = AsyncToMWrapper(model_config)

        # Create all evaluation tasks for this model
        tasks = []
        for item in data:
            instance_id = item["instance_id"]
            problem_statement = item["problem_statement"]  # unclear version
            original_issue = item["original_issue"]  # clear version

            # Task for unclear statement
            tasks.append(
                self.evaluate_statement_async(
                    agent_wrapper,
                    problem_statement,
                    "unclear",
                    original_issue,
                    f"{instance_id}_unclear",
                )
            )

            # Task for clear statement
            tasks.append(
                self.evaluate_statement_async(
                    agent_wrapper,
                    original_issue,
                    "clear",
                    original_issue,
                    f"{instance_id}_clear",
                )
            )

        # Run all API calls for this model in parallel with semaphore limiting
        semaphore = asyncio.Semaphore(self.max_concurrent_calls)

        async def limited_task(task):
            async with semaphore:
                return await task

        print(
            f"🔄 Running {len(tasks)} API calls in parallel (max {self.max_concurrent_calls} concurrent)..."
        )
        start_time = time.time()

        results = await asyncio.gather(
            *[limited_task(task) for task in tasks], return_exceptions=True
        )

        # Filter out exceptions and convert to EvalResult list
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"⚠️  Task failed: {result}")
            else:
                valid_results.append(result)

        evaluation_time = time.time() - start_time
        print(
            f"✅ {model_config.name} completed in {evaluation_time:.1f}s ({len(valid_results)} successful)"
        )

        # Calculate metrics
        correct_count = sum(1 for r in valid_results if r.correct_classification)
        total_count = len(valid_results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        unclear_results = [r for r in valid_results if r.statement_type == "unclear"]
        clear_results = [r for r in valid_results if r.statement_type == "clear"]

        unclear_accuracy = (
            sum(1 for r in unclear_results if r.correct_classification)
            / len(unclear_results)
            if unclear_results
            else 0.0
        )
        clear_accuracy = (
            sum(1 for r in clear_results if r.correct_classification)
            / len(clear_results)
            if clear_results
            else 0.0
        )

        total_cost = sum(r.cost_usd for r in valid_results)
        total_input_tokens = sum(r.input_tokens for r in valid_results)
        total_output_tokens = sum(r.output_tokens for r in valid_results)
        avg_evaluation_time = sum(
            r.evaluation_time_seconds for r in valid_results
        ) / len(valid_results)

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
            results=valid_results,
        )

    async def run_all_models_parallel(
        self, data: List[Dict[str, Any]]
    ) -> Dict[str, ModelResults]:
        """Run all models in parallel."""
        print("🚀 Starting PARALLEL evaluation of all models...")
        print(
            f"📊 Models: {len(self.models)}, Samples: {len(data)}, Total API calls: {len(data) * 2 * len(self.models)}"
        )

        start_time = time.time()

        # Run all models in parallel
        model_tasks = [
            self.evaluate_model_async(model_config, data)
            for model_config in self.models
        ]

        model_results = await asyncio.gather(*model_tasks, return_exceptions=True)

        # Process results
        results_by_model = {}
        for i, result in enumerate(model_results):
            if isinstance(result, Exception):
                print(f"❌ Model {self.models[i].name} failed: {result}")
            else:
                results_by_model[result.model_name] = result
                print(
                    f"✅ {result.model_name}: {result.accuracy:.2%} accuracy, ${result.total_cost:.4f} cost"
                )

        total_time = time.time() - start_time
        print(f"\n🎉 All models completed in {total_time:.1f}s!")

        return results_by_model

    def generate_comparison_report(
        self, results_by_model: Dict[str, ModelResults]
    ) -> Dict[str, Any]:
        """Generate comparison report."""
        if not results_by_model:
            return {}

        # Sort by accuracy
        sorted_models = sorted(
            results_by_model.values(), key=lambda x: x.accuracy, reverse=True
        )

        # Calculate cost effectiveness
        cost_effectiveness = []
        for model in sorted_models:
            if model.total_cost > 0:
                effectiveness = model.accuracy / model.total_cost
                cost_effectiveness.append((model.model_name, effectiveness))

        cost_effectiveness.sort(key=lambda x: x[1], reverse=True)

        return {
            "evaluation_summary": {
                "total_models_evaluated": len(results_by_model),
                "samples_per_model": self.num_samples,
                "total_consultations": sum(
                    m.total_samples for m in results_by_model.values()
                ),
                "max_concurrent_calls": self.max_concurrent_calls,
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
                    "avg_evaluation_time": model.avg_evaluation_time,
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
                        for m in results_by_model.values()
                        if m.model_name == name
                    ),
                    "total_cost": next(
                        m.total_cost
                        for m in results_by_model.values()
                        if m.model_name == name
                    ),
                }
                for i, (name, effectiveness) in enumerate(cost_effectiveness)
            ],
            "detailed_results": {
                model_name: asdict(results)
                for model_name, results in results_by_model.items()
            },
        }

    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("📊 PARALLEL MULTI-MODEL EVALUATION RESULTS")
        print("=" * 80)

        if not report.get("accuracy_ranking"):
            print("❌ No results to display")
            return

        summary = report["evaluation_summary"]
        print(f"📋 Models Evaluated: {summary['total_models_evaluated']}")
        print(f"📋 Samples per Model: {summary['samples_per_model']}")
        print(f"📋 Total Consultations: {summary['total_consultations']}")
        print(f"📋 Max Concurrent API Calls: {summary['max_concurrent_calls']}")

        print("\n🏆 ACCURACY RANKING:")
        print("-" * 80)
        for item in report["accuracy_ranking"]:
            print(
                f"{item['rank']}. {item['model']:<20} "
                f"Accuracy: {item['accuracy']:.2%} "
                f"Cost: ${item['total_cost']:.4f} "
                f"Time: {item['avg_evaluation_time']:.1f}s/call"
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

    def save_results(self, report: Dict[str, Any], output_path: str) -> None:
        """Save results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"💾 Results saved to {output_file}")


async def main():
    """Main async entry point."""
    # Configuration
    dataset_path = (
        "/Users/xuhuizhou/Projects/ToM-SWE/data/stateful_swe/dataset_for_upload"
    )
    output_path = "/Users/xuhuizhou/Projects/ToM-SWE/stateful_swe/parallel_results.json"
    num_samples = 10
    max_concurrent_calls = 5  # Adjust based on API rate limits

    # Set random seed
    random.seed(42)

    try:
        # Initialize evaluator
        evaluator = ParallelMultiModelEvaluator(
            dataset_path, num_samples, max_concurrent_calls
        )

        # Load dataset
        data = evaluator.load_dataset()

        # Run parallel evaluation
        results_by_model = await evaluator.run_all_models_parallel(data)

        if not results_by_model:
            print("❌ No models were successfully evaluated")
            return 1

        # Generate and display report
        report = evaluator.generate_comparison_report(results_by_model)
        evaluator.print_summary(report)
        evaluator.save_results(report, output_path)

        print("\n✅ Parallel evaluation completed!")
        print(f"📁 Results saved to: {output_path}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
