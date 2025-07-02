#!/usr/bin/env python3
"""
Script to analyze user code trajectories.

This script processes user data files and generates analysis results.
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from tom_module.analyzer import CodeAnalyzer
from utils.data_utils import get_user_data_files, load_json, save_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("analyze_users.log"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def analyze_user(user_file: Path, analyzer: CodeAnalyzer) -> Dict[str, Any]:
    """Analyze a user's code trajectory.

    Args:
        user_file: Path to the user data file.
        analyzer: CodeAnalyzer instance.

    Returns:
        Dict containing analysis results.
    """
    logger.info(f"Analyzing user data from {user_file}")

    # Load user data
    user_data = load_json(user_file)

    if not user_data:
        logger.warning(f"Empty or invalid user data file: {user_file}")
        return {}

    user_id = user_data.get("user_id", "unknown")
    logger.info(f"Processing user {user_id}")

    # Get code snapshots
    snapshots = user_data.get("code_snapshots", [])

    if not snapshots:
        logger.warning(f"No code snapshots found for user {user_id}")
        return {
            "user_id": user_id,
            "analysis_timestamp": datetime.datetime.now().isoformat(),
            "results": [],
            "summary": {"error": "No code snapshots found"},
        }

    # Analyze each snapshot
    results = []

    for snapshot in snapshots:
        timestamp = snapshot.get("timestamp", "")
        code = snapshot.get("code", "")

        if not timestamp or not code:
            logger.warning(f"Invalid snapshot for user {user_id}: missing timestamp or code")
            continue

        try:
            analysis = analyzer.analyze(code)

            results.append({"timestamp": timestamp, "analysis": analysis})

            logger.info(f"Analyzed snapshot from {timestamp} for user {user_id}")
        except Exception as e:
            logger.error(f"Error analyzing snapshot from {timestamp} for user {user_id}: {e}")

    # Generate summary
    summary = generate_summary(results)

    # Compile final results
    analysis_results = {
        "user_id": user_id,
        "analysis_timestamp": datetime.datetime.now().isoformat(),
        "results": results,
        "summary": summary,
    }

    logger.info(f"Completed analysis for user {user_id}")
    return analysis_results


def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary of analysis results.

    Args:
        results: List of analysis results.

    Returns:
        Dict containing summary information.
    """
    if not results:
        return {"error": "No results to summarize"}

    # Extract progression of complexity
    complexity_values = [
        result.get("analysis", {}).get("cyclomatic_complexity", 1) for result in results
    ]

    # Extract progression of intent
    intent_values = [
        result.get("analysis", {}).get("intent_analysis", {}).get("inferred_intent", "")
        for result in results
    ]

    # Determine progression pattern
    if len(complexity_values) >= 2:
        first = complexity_values[0]
        last = complexity_values[-1]

        if last > first:
            progression = "The user's code increased in complexity over time."
        elif last < first:
            progression = "The user's code decreased in complexity over time."
        else:
            progression = "The user's code maintained similar complexity over time."
    else:
        progression = "Insufficient data to determine complexity progression."

    # Determine learning curve
    if len(results) >= 3:
        # Check if code structure evolved
        first_functions = results[0].get("analysis", {}).get("functions", 0)
        last_functions = results[-1].get("analysis", {}).get("functions", 0)

        first_classes = results[0].get("analysis", {}).get("classes", 0)
        last_classes = results[-1].get("analysis", {}).get("classes", 0)

        if last_classes > first_classes:
            learning_curve = "The user progressed from procedural to object-oriented programming."
        elif last_functions > first_functions:
            learning_curve = "The user added more functionality over time."
        else:
            learning_curve = (
                "The user refined their initial approach without major structural changes."
            )
    else:
        learning_curve = "Insufficient data to determine learning curve."

    # Assess code quality
    if len(results) >= 2:
        # Check for docstrings
        first_code = results[0].get("analysis", {}).get("parse_error") is None
        last_code = results[-1].get("analysis", {}).get("parse_error") is None

        if not first_code and last_code:
            code_quality = "The user's code quality improved from invalid to valid syntax."
        elif first_code and last_code:
            code_quality = "The user maintained valid code throughout the session."
        else:
            code_quality = "The user's code had syntax issues."
    else:
        code_quality = "Insufficient data to assess code quality."

    # Analyze intent evolution
    if len(intent_values) >= 2:
        if intent_values[0] != intent_values[-1]:
            intent_evolution = "The user's intent evolved during the session."
        else:
            intent_evolution = "The user maintained a consistent intent throughout the session."
    else:
        intent_evolution = "Insufficient data to analyze intent evolution."

    return {
        "progression": progression,
        "learning_curve": learning_curve,
        "code_quality": code_quality,
        "intent_evolution": intent_evolution,
    }


def save_analysis_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save analysis results to a file.

    Args:
        results: Analysis results.
        output_dir: Directory to save the results.
    """
    user_id = results.get("user_id", "unknown")
    output_file = output_dir / f"analysis_{user_id}.json"

    if save_json(results, output_file):
        logger.info(f"Saved analysis results to {output_file}")
    else:
        logger.error(f"Failed to save analysis results to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze user code trajectories")
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory containing user data files"
    )
    parser.add_argument(
        "--user-id", type=str, default=None, help="Specific user ID to analyze (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save analysis results (defaults to data_dir)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize analyzer
    analyzer = CodeAnalyzer()

    # Get user files
    if args.user_id:
        user_files = [data_dir / f"user_{args.user_id}.json"]
        if not user_files[0].exists():
            logger.error(f"User file not found: {user_files[0]}")
            return
    else:
        user_files = get_user_data_files(data_dir)

    if not user_files:
        logger.error(f"No user data files found in {data_dir}")
        return

    logger.info(f"Found {len(user_files)} user data files")

    # Process each user file
    for user_file in user_files:
        try:
            results = analyze_user(user_file, analyzer)
            if results:
                save_analysis_results(results, output_dir)
        except Exception as e:
            logger.error(f"Error processing {user_file}: {e}")

    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
