"""Tools for analyzing user interactions with code."""

from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np


def analyze_user_edits(original_code: str, edited_code: str) -> Dict[str, Any]:
    """
    Analyze the differences between original and edited code.

    Args:
        original_code: The original code snippet.
        edited_code: The edited code snippet.

    Returns:
        A dictionary with analysis results.
    """
    if not original_code and not edited_code:
        return {
            "changes": 0,
            "lines_added": 0,
            "lines_removed": 0,
            "change_percentage": 0
        }
    
    # Split into lines
    original_lines = original_code.split('\n')
    edited_lines = edited_code.split('\n')
    
    # Calculate basic metrics
    original_line_count = len(original_lines)
    edited_line_count = len(edited_lines)
    
    lines_added = max(0, edited_line_count - original_line_count)
    lines_removed = max(0, original_line_count - edited_line_count)
    
    # Simple diff calculation
    from difflib import Differ
    differ = Differ()
    diff = list(differ.compare(original_lines, edited_lines))
    
    changes = sum(1 for line in diff if line.startswith('+ ') or line.startswith('- '))
    
    # Calculate change percentage
    if original_line_count > 0:
        change_percentage = (changes / original_line_count) * 100
    else:
        change_percentage = 100 if edited_line_count > 0 else 0
    
    return {
        "changes": changes,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "change_percentage": change_percentage
    }


def track_user_behavior(user_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze user behavior based on a sequence of actions.

    Args:
        user_actions: A list of user action records.

    Returns:
        A dictionary with analysis results.
    """
    if not user_actions:
        return {
            "total_actions": 0,
            "action_types": {},
            "average_time_between_actions": 0,
            "total_session_time": 0
        }
    
    # Count action types
    action_types = {}
    for action in user_actions:
        action_type = action.get("type", "unknown")
        action_types[action_type] = action_types.get(action_type, 0) + 1
    
    # Calculate time metrics
    timestamps = [action.get("timestamp", 0) for action in user_actions if "timestamp" in action]
    
    if len(timestamps) >= 2:
        time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        average_time = sum(time_diffs) / len(time_diffs)
        total_time = timestamps[-1] - timestamps[0]
    else:
        average_time = 0
        total_time = 0
    
    return {
        "total_actions": len(user_actions),
        "action_types": action_types,
        "average_time_between_actions": average_time,
        "total_session_time": total_time
    }


def visualize_user_behavior(user_behavior: Dict[str, Any]) -> None:
    """
    Create visualizations of user behavior.

    Args:
        user_behavior: User behavior analysis results.
    """
    if not user_behavior or user_behavior.get("total_actions", 0) == 0:
        print("No user behavior data to visualize")
        return
    
    # Create action type distribution plot
    action_types = user_behavior.get("action_types", {})
    
    if action_types:
        plt.figure(figsize=(10, 6))
        
        # Sort by frequency
        sorted_actions = sorted(action_types.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_actions]
        values = [item[1] for item in sorted_actions]
        
        plt.bar(labels, values, color='skyblue')
        plt.title('Distribution of User Actions')
        plt.xlabel('Action Type')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def generate_user_behavior_report(user_behavior: Dict[str, Any]) -> str:
    """
    Generate a text report of user behavior.

    Args:
        user_behavior: User behavior analysis results.

    Returns:
        A formatted text report.
    """
    if not user_behavior:
        return "No user behavior data available."
    
    total_actions = user_behavior.get("total_actions", 0)
    action_types = user_behavior.get("action_types", {})
    avg_time = user_behavior.get("average_time_between_actions", 0)
    total_time = user_behavior.get("total_session_time", 0)
    
    # Format times
    avg_time_str = f"{avg_time:.2f} seconds" if avg_time < 60 else f"{avg_time/60:.2f} minutes"
    total_time_str = f"{total_time:.2f} seconds" if total_time < 60 else f"{total_time/60:.2f} minutes"
    
    # Build report
    report = [
        "User Behavior Analysis Report",
        "==============================",
        f"Total actions: {total_actions}",
        f"Total session time: {total_time_str}",
        f"Average time between actions: {avg_time_str}",
        "",
        "Action Type Distribution:",
    ]
    
    # Add action type breakdown
    if action_types:
        sorted_actions = sorted(action_types.items(), key=lambda x: x[1], reverse=True)
        for action_type, count in sorted_actions:
            percentage = (count / total_actions) * 100
            report.append(f"  - {action_type}: {count} ({percentage:.1f}%)")
    else:
        report.append("  No action type data available.")
    
    return "\n".join(report)