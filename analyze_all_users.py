#!/usr/bin/env python3
"""
Script to analyze all users' code trajectories.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from tom_module.analyzer import CodeAnalyzer
from visualization.trajectory_viewer import plot_trajectory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("analyze_users.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


def load_user_data(user_id: str) -> Dict[str, Any]:
    """Load user data from JSON file."""
    user_file = DATA_DIR / f"user_{user_id}.json"
    if not user_file.exists():
        logger.error(f"User file not found: {user_file}")
        return {}
    
    try:
        with open(user_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {user_file}")
        return {}


def analyze_user(user_id: str) -> Dict[str, Any]:
    """Analyze a single user's code trajectory."""
    logger.info(f"Analyzing user {user_id}")
    
    user_data = load_user_data(user_id)
    if not user_data:
        return {}
    
    analyzer = CodeAnalyzer()
    
    # Analyze each code snapshot
    results = []
    for snapshot in user_data.get("code_snapshots", []):
        code = snapshot.get("code", "")
        timestamp = snapshot.get("timestamp", "")
        
        analysis = analyzer.analyze(code)
        results.append({
            "timestamp": timestamp,
            "analysis": analysis
        })
    
    return {
        "user_id": user_id,
        "results": results
    }


def save_analysis(user_id: str, analysis: Dict[str, Any]) -> None:
    """Save analysis results to file."""
    output_file = DATA_DIR / f"analysis_{user_id}.json"
    
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Analysis saved to {output_file}")


def visualize_analysis(user_id: str, analysis: Dict[str, Any]) -> None:
    """Create visualization of analysis results."""
    if not analysis or "results" not in analysis:
        logger.warning(f"No analysis results to visualize for user {user_id}")
        return
    
    try:
        plot_trajectory(analysis["results"], user_id)
        logger.info(f"Visualization created for user {user_id}")
    except Exception as e:
        logger.error(f"Error creating visualization for user {user_id}: {e}")


def main() -> None:
    """Main function to analyze all users."""
    logger.info("Starting analysis of all users")
    
    # Get all user IDs from data directory
    user_files = list(DATA_DIR.glob("user_*.json"))
    user_ids = [f.stem.replace("user_", "") for f in user_files]
    
    logger.info(f"Found {len(user_ids)} users to analyze")
    
    for user_id in user_ids:
        analysis = analyze_user(user_id)
        if analysis:
            save_analysis(user_id, analysis)
            visualize_analysis(user_id, analysis)
    
    logger.info("Analysis of all users completed")


if __name__ == "__main__":
    main()