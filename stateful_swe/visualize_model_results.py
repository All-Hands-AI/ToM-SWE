#!/usr/bin/env python3
"""
Visualization and detailed analysis of multi-model clarity evaluation results.

Creates charts and detailed breakdowns of model performance vs cost.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import argparse


def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def create_accuracy_vs_cost_plot(report: Dict[str, Any], output_dir: Path) -> None:
    """Create scatter plot of accuracy vs total cost."""
    if not report.get("accuracy_ranking"):
        return

    data = []
    for item in report["accuracy_ranking"]:
        data.append(
            {
                "Model": item["model"],
                "Accuracy": item["accuracy"],
                "Total_Cost": item["total_cost"],
                "Cost_Per_Consultation": item["cost_per_consultation"],
            }
        )

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 8))

    # Create scatter plot
    plt.scatter(df["Total_Cost"], df["Accuracy"], s=100, alpha=0.7)

    # Add model labels
    for i, row in df.iterrows():
        plt.annotate(
            row["Model"],
            (row["Total_Cost"], row["Accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    plt.xlabel("Total Cost ($)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(
        "Model Performance vs Cost\n(Clarity Classification Task)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_cost.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_cost_effectiveness_chart(report: Dict[str, Any], output_dir: Path) -> None:
    """Create bar chart of cost effectiveness scores."""
    if not report.get("cost_effectiveness_ranking"):
        return

    data = report["cost_effectiveness_ranking"]
    models = [item["model"] for item in data]
    scores = [item["cost_effectiveness_score"] for item in data]

    plt.figure(figsize=(12, 6))

    bars = plt.bar(models, scores, alpha=0.7, color=plt.cm.viridis(range(len(models))))

    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Cost Effectiveness Score\n(Accuracy per Dollar)", fontsize=12)
    plt.title("Model Cost Effectiveness Ranking", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{score:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "cost_effectiveness.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_detailed_accuracy_breakdown(
    report: Dict[str, Any], output_dir: Path
) -> None:
    """Create grouped bar chart showing clear vs unclear accuracy."""
    if not report.get("accuracy_ranking"):
        return

    models = []
    clear_acc = []
    unclear_acc = []
    overall_acc = []

    for item in report["accuracy_ranking"]:
        models.append(item["model"])
        clear_acc.append(item["clear_accuracy"])
        unclear_acc.append(item["unclear_accuracy"])
        overall_acc.append(item["accuracy"])

    x = range(len(models))
    width = 0.25

    plt.figure(figsize=(14, 8))

    bars1 = plt.bar(
        [i - width for i in x], clear_acc, width, label="Clear Statements", alpha=0.8
    )
    bars2 = plt.bar(x, unclear_acc, width, label="Unclear Statements", alpha=0.8)
    bars3 = plt.bar(
        [i + width for i in x], overall_acc, width, label="Overall", alpha=0.8
    )

    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(
        "Detailed Accuracy Breakdown by Statement Type", fontsize=14, fontweight="bold"
    )
    plt.xticks(x, models, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_cost_breakdown_chart(report: Dict[str, Any], output_dir: Path) -> None:
    """Create chart showing cost per consultation for each model."""
    if not report.get("accuracy_ranking"):
        return

    models = [item["model"] for item in report["accuracy_ranking"]]
    costs = [item["cost_per_consultation"] for item in report["accuracy_ranking"]]

    plt.figure(figsize=(12, 6))

    bars = plt.bar(models, costs, alpha=0.7, color=plt.cm.plasma(range(len(models))))

    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Cost per Consultation ($)", fontsize=12)
    plt.title("Cost per Consultation by Model", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"${cost:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
        )

    # Use log scale if costs vary widely
    if max(costs) / min(costs) > 10:
        plt.yscale("log")
        plt.ylabel("Cost per Consultation ($) - Log Scale", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "cost_per_consultation.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_table(report: Dict[str, Any], output_dir: Path) -> None:
    """Generate a summary table as HTML and CSV."""
    if not report.get("accuracy_ranking"):
        return

    # Create DataFrame
    data = []
    for i, item in enumerate(report["accuracy_ranking"], 1):
        # Find cost effectiveness rank
        ce_rank = next(
            (
                ce["rank"]
                for ce in report.get("cost_effectiveness_ranking", [])
                if ce["model"] == item["model"]
            ),
            "-",
        )

        data.append(
            {
                "Accuracy_Rank": i,
                "Model": item["model"],
                "Overall_Accuracy": f"{item['accuracy']:.2%}",
                "Clear_Accuracy": f"{item['clear_accuracy']:.2%}",
                "Unclear_Accuracy": f"{item['unclear_accuracy']:.2%}",
                "Total_Cost": f"${item['total_cost']:.4f}",
                "Cost_Per_Consultation": f"${item['cost_per_consultation']:.6f}",
                "Cost_Effectiveness_Rank": ce_rank,
            }
        )

    df = pd.DataFrame(data)

    # Save as CSV
    df.to_csv(output_dir / "summary_table.csv", index=False)

    # Save as HTML with styling
    html_table = df.to_html(index=False, table_id="results-table", escape=False)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Model Clarity Evaluation Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; text-align: center; }}
            #results-table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            #results-table th, #results-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            #results-table th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            #results-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{
                background-color: #e7f3ff;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Multi-Model Clarity Evaluation Results</h1>

        <div class="summary">
            <h3>Evaluation Summary</h3>
            <p><strong>Models Evaluated:</strong> {report.get('evaluation_summary', {}).get('total_models_evaluated', 'N/A')}</p>
            <p><strong>Samples per Model:</strong> {report.get('evaluation_summary', {}).get('samples_per_model', 'N/A')}</p>
            <p><strong>Total Consultations:</strong> {report.get('evaluation_summary', {}).get('total_consultations', 'N/A')}</p>
        </div>

        {html_table}

        <div class="summary">
            <p><em>Cost Effectiveness Rank: Accuracy per dollar spent (higher is better)</em></p>
            <p><em>Generated by ToM-SWE Multi-Model Clarity Evaluator</em></p>
        </div>
    </body>
    </html>
    """

    with open(output_dir / "summary_table.html", "w") as f:
        f.write(html_content)


def analyze_error_patterns(report: Dict[str, Any], output_dir: Path) -> None:
    """Analyze common error patterns across models."""
    if not report.get("detailed_results"):
        return

    error_analysis = {}

    for model_name, model_data in report["detailed_results"].items():
        results = model_data.get("results", [])

        # Count error types
        false_positives = 0  # Model suggested questions for clear statements
        false_negatives = 0  # Model didn't suggest questions for unclear statements

        for result in results:
            if result["statement_type"] == "clear" and result["suggested_questions"]:
                false_positives += 1
            elif (
                result["statement_type"] == "unclear"
                and not result["suggested_questions"]
            ):
                false_negatives += 1

        total_clear = sum(1 for r in results if r["statement_type"] == "clear")
        total_unclear = sum(1 for r in results if r["statement_type"] == "unclear")

        error_analysis[model_name] = {
            "false_positive_rate": false_positives / total_clear
            if total_clear > 0
            else 0,
            "false_negative_rate": false_negatives / total_unclear
            if total_unclear > 0
            else 0,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    # Create visualization
    models = list(error_analysis.keys())
    fp_rates = [error_analysis[m]["false_positive_rate"] for m in models]
    fn_rates = [error_analysis[m]["false_negative_rate"] for m in models]

    x = range(len(models))
    width = 0.35

    plt.figure(figsize=(12, 6))

    bars1 = plt.bar(
        [i - width / 2 for i in x],
        fp_rates,
        width,
        label="False Positive Rate\n(Wrong questions for clear)",
        alpha=0.8,
    )
    bars2 = plt.bar(
        [i + width / 2 for i in x],
        fn_rates,
        width,
        label="False Negative Rate\n(No questions for unclear)",
        alpha=0.8,
    )

    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Error Rate", fontsize=12)
    plt.title("Error Pattern Analysis by Model", fontsize=14, fontweight="bold")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig(output_dir / "error_patterns.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save error analysis as JSON
    with open(output_dir / "error_analysis.json", "w") as f:
        json.dump(error_analysis, f, indent=2)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize multi-model clarity evaluation results"
    )
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument(
        "--output-dir",
        default="./visualizations",
        help="Output directory for visualizations (default: ./visualizations)",
    )

    args = parser.parse_args()

    # Load results
    try:
        report = load_results(args.results_file)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Creating visualizations in {output_dir}...")

    # Generate all visualizations
    try:
        create_accuracy_vs_cost_plot(report, output_dir)
        print("✅ Created accuracy vs cost plot")

        create_cost_effectiveness_chart(report, output_dir)
        print("✅ Created cost effectiveness chart")

        create_detailed_accuracy_breakdown(report, output_dir)
        print("✅ Created detailed accuracy breakdown")

        create_cost_breakdown_chart(report, output_dir)
        print("✅ Created cost breakdown chart")

        generate_summary_table(report, output_dir)
        print("✅ Generated summary table (HTML & CSV)")

        analyze_error_patterns(report, output_dir)
        print("✅ Created error pattern analysis")

        print("\n🎉 All visualizations created successfully!")
        print(f"📁 Output directory: {output_dir.absolute()}")
        print(f"📋 Open {output_dir}/summary_table.html for interactive results")

    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
