#!/usr/bin/env python3
"""
Quick runner and analysis script for multi-model clarity evaluation.

Usage examples:
  # Run full evaluation on all models
  python run_model_comparison.py --all

  # Run specific models only
  python run_model_comparison.py --models gpt-5-nano gpt-5-mini

  # Analyze existing results
  python run_model_comparison.py --analyze-only

  # Quick test with fewer samples
  python run_model_comparison.py --models gpt-5-nano --samples 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from multi_model_clarity_eval import MultiModelClarityEvaluator


def get_available_models() -> List[str]:
    """Get list of available model names."""
    return ["gpt-5-nano", "gpt-5-mini", "gpt-5", "claude-3.5-sonnet", "claude-4"]


def filter_models_by_name(
    evaluator: MultiModelClarityEvaluator, model_names: List[str]
) -> None:
    """Filter evaluator to only include specified models."""
    all_models = {
        model.name.lower().replace("-", "").replace(".", ""): model
        for model in evaluator.models
    }
    filtered_models = []

    for name in model_names:
        # Normalize name for matching
        normalized_name = name.lower().replace("-", "").replace(".", "")

        # Find matching model
        for model_key, model in all_models.items():
            if normalized_name in model_key or model_key in normalized_name:
                filtered_models.append(model)
                break
        else:
            print(
                f"⚠️  Warning: Model '{name}' not found. Available: {list(all_models.keys())}"
            )

    if filtered_models:
        evaluator.models = filtered_models
        print(
            f"✅ Filtered to {len(filtered_models)} models: {[m.name for m in filtered_models]}"
        )
    else:
        print("❌ No valid models found. Using all models.")


def analyze_existing_results(results_file: Path) -> None:
    """Analyze existing results file and print insights."""
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return

    print("📊 Analyzing existing results...")

    with open(results_file, "r") as f:
        report = json.load(f)

    # Create a dummy evaluator just for printing
    evaluator = MultiModelClarityEvaluator("", 0)
    evaluator.print_summary(report)

    # Additional insights
    print("\n🔍 ADDITIONAL INSIGHTS:")
    print("-" * 80)

    if "accuracy_ranking" in report:
        accuracies = [item["accuracy"] for item in report["accuracy_ranking"]]
        costs = [item["total_cost"] for item in report["accuracy_ranking"]]

        best_accuracy = max(accuracies)
        lowest_cost = min(costs)

        print(f"📈 Best Accuracy: {best_accuracy:.2%}")
        print(f"💰 Lowest Total Cost: ${lowest_cost:.4f}")

        # Find sweet spot (good accuracy at reasonable cost)
        sweet_spot = None
        for item in report["accuracy_ranking"]:
            if (
                item["accuracy"] >= 0.7 and item["total_cost"] <= 0.01
            ):  # 70% accuracy under $0.01
                sweet_spot = item
                break

        if sweet_spot:
            print(
                f"🎯 Sweet Spot: {sweet_spot['model']} ({sweet_spot['accuracy']:.2%} accuracy at ${sweet_spot['total_cost']:.4f})"
            )


def run_evaluation(
    dataset_path: str,
    output_path: str,
    num_samples: int,
    model_names: Optional[List[str]] = None,
) -> None:
    """Run the evaluation with specified parameters."""
    print(f"🚀 Starting evaluation with {num_samples} samples per model")

    evaluator = MultiModelClarityEvaluator(dataset_path, num_samples)

    # Filter models if specified
    if model_names:
        filter_models_by_name(evaluator, model_names)

    # Run evaluation
    results_by_model = evaluator.run_full_evaluation()

    if not results_by_model:
        print("❌ No models were successfully evaluated")
        return

    # Generate and save report
    report = evaluator.generate_comparison_report()
    evaluator.print_summary(report)
    evaluator.save_results(report, output_path)

    print("\n✅ Evaluation completed!")
    print(f"📁 Results saved to: {output_path}")


def estimate_costs(num_samples: int, model_names: Optional[List[str]] = None) -> None:
    """Estimate costs for the evaluation."""
    print("💰 COST ESTIMATION:")
    print("-" * 50)

    # Rough estimates per consultation
    model_costs = {
        "gpt-5-nano": 0.001,
        "gpt-5-mini": 0.004,
        "gpt-5": 0.020,
        "claude-3.5-sonnet": 0.070,
        "claude-4": 0.200,
    }

    # Each sample generates 2 consultations (clear + unclear)
    consultations_per_model = num_samples * 2

    models_to_estimate = model_names if model_names else list(model_costs.keys())

    total_cost = 0
    for model in models_to_estimate:
        if model.replace("-", "").replace(".", "").lower() in [
            k.replace("-", "").replace(".", "").lower() for k in model_costs.keys()
        ]:
            # Find matching cost
            cost_per_consult = None
            for k, v in model_costs.items():
                if (
                    model.replace("-", "").replace(".", "").lower()
                    in k.replace("-", "").replace(".", "").lower()
                ):
                    cost_per_consult = v
                    break

            if cost_per_consult:
                model_total = consultations_per_model * cost_per_consult
                total_cost += model_total
                print(
                    f"{model:<20} ${model_total:.4f} ({consultations_per_model} consultations @ ${cost_per_consult:.6f})"
                )

    print("-" * 50)
    print(f"{'TOTAL ESTIMATED COST':<20} ${total_cost:.4f}")
    print(
        f"Models: {len(models_to_estimate)}, Samples: {num_samples}, Consultations: {consultations_per_model * len(models_to_estimate)}"
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-model clarity evaluation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation (all models, 100 samples each)
  python run_model_comparison.py --all

  # Test specific models with fewer samples
  python run_model_comparison.py --models gpt-5-nano gpt-5-mini --samples 10

  # Just analyze existing results
  python run_model_comparison.py --analyze-only

  # Estimate costs without running
  python run_model_comparison.py --estimate-only --models gpt-5-nano --samples 50
        """,
    )

    # Main actions
    parser.add_argument(
        "--all", action="store_true", help="Run evaluation on all models"
    )
    parser.add_argument(
        "--analyze-only", action="store_true", help="Only analyze existing results"
    )
    parser.add_argument(
        "--estimate-only", action="store_true", help="Only estimate costs, don't run"
    )

    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        choices=get_available_models(),
        help="Specific models to evaluate",
    )

    # Configuration
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples per model (default: 100)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/Users/xuhuizhou/Projects/ToM-SWE/data/stateful_swe/dataset_for_upload",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/Users/xuhuizhou/Projects/ToM-SWE/stateful_swe/multi_model_clarity_results.json",
        help="Output file path",
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.all, args.analyze_only, args.estimate_only, args.models]):
        parser.print_help()
        print(
            "\n❌ Error: Must specify --all, --analyze-only, --estimate-only, or --models"
        )
        sys.exit(1)

    # Set model list
    model_names = None
    if args.models:
        model_names = args.models
    elif args.all:
        model_names = get_available_models()

    # Execute requested action
    try:
        if args.analyze_only:
            analyze_existing_results(Path(args.output))
        elif args.estimate_only:
            estimate_costs(args.samples, model_names)
        else:
            if args.estimate_only or input(
                "📋 Show cost estimate first? (y/N): "
            ).lower().startswith("y"):
                estimate_costs(args.samples, model_names)
                if (
                    not input("\n🚀 Proceed with evaluation? (y/N): ")
                    .lower()
                    .startswith("y")
                ):
                    print("Evaluation cancelled.")
                    return

            run_evaluation(args.dataset, args.output, args.samples, model_names)

    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
