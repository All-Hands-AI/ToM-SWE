"""User behavior visualization for ToM-SWE."""

import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def analyze_user_edits(original_code: str, edited_code: str) -> Dict[str, Any]:
    """Analyze differences between original and edited code.

    Args:
        original_code: Original code.
        edited_code: Edited code.

    Returns:
        Dict containing analysis results.
    """
    logger.info("Analyzing user edits")

    # Split into lines
    original_lines = original_code.split("\n")
    edited_lines = edited_code.split("\n")

    # Count lines
    original_count = len(original_lines)
    edited_count = len(edited_lines)

    # Calculate changes
    lines_added = max(0, edited_count - original_count)
    lines_removed = max(0, original_count - edited_count)

    # Calculate change percentage
    total_lines = max(original_count, edited_count)
    change_percentage = (
        ((lines_added + lines_removed) / total_lines) * 100 if total_lines > 0 else 0
    )

    # Count modified lines (simple approach)
    modified_count = 0
    for i in range(min(original_count, edited_count)):
        if original_lines[i] != edited_lines[i]:
            modified_count += 1

    return {
        "changes": modified_count + lines_added + lines_removed,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "lines_modified": modified_count,
        "change_percentage": change_percentage,
        "original_line_count": original_count,
        "edited_line_count": edited_count,
    }


def track_user_behavior(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Track user behavior based on actions.

    Args:
        actions: List of user actions.

    Returns:
        Dict containing behavior analysis.
    """
    logger.info("Tracking user behavior")

    if not actions:
        return {
            "total_actions": 0,
            "action_types": {},
            "average_time_between_actions": 0,
            "total_session_time": 0,
        }

    # Count action types
    action_types: Dict[str, int] = {}
    for action in actions:
        action_type = action.get("type", "unknown")
        action_types[action_type] = action_types.get(action_type, 0) + 1

    # Calculate time between actions
    time_diffs = []
    for i in range(1, len(actions)):
        prev_time = actions[i - 1].get("timestamp", 0)
        curr_time = actions[i].get("timestamp", 0)
        if prev_time and curr_time:
            time_diffs.append(curr_time - prev_time)

    avg_time_between_actions = sum(time_diffs) / len(time_diffs) if time_diffs else 0

    # Calculate total session time
    if len(actions) >= 2:
        first_time = actions[0].get("timestamp", 0)
        last_time = actions[-1].get("timestamp", 0)
        total_session_time = last_time - first_time
    else:
        total_session_time = 0

    return {
        "total_actions": len(actions),
        "action_types": action_types,
        "average_time_between_actions": avg_time_between_actions,
        "total_session_time": total_session_time,
    }


def visualize_user_behavior(behavior_data: Dict[str, Any]) -> None:
    """Visualize user behavior.

    Args:
        behavior_data: User behavior data.
    """
    logger.info("Visualizing user behavior")

    if not behavior_data or "action_types" not in behavior_data:
        logger.warning("No behavior data to visualize")
        return

    action_types = behavior_data.get("action_types", {})
    if not action_types:
        logger.warning("No action types in behavior data")
        return

    # Create bar chart of action types
    plt.figure(figsize=(10, 6))

    types = list(action_types.keys())
    counts = list(action_types.values())

    plt.bar(types, counts, color="blue")
    plt.title("User Actions by Type")
    plt.xlabel("Action Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def generate_behavior_report(behavior_data: Dict[str, Any]) -> str:
    """Generate a report of user behavior.

    Args:
        behavior_data: User behavior data.

    Returns:
        Text report.
    """
    logger.info("Generating behavior report")

    if not behavior_data:
        return "No user behavior data available."

    # Create report
    report = "User Behavior Analysis Report\n"
    report += "=============================\n\n"

    total_actions = behavior_data.get("total_actions", 0)
    report += f"Total actions: {total_actions}\n"

    action_types = behavior_data.get("action_types", {})
    if action_types:
        report += "\nAction types:\n"
        for action_type, count in action_types.items():
            report += f"  {action_type}: {count}\n"

    avg_time = behavior_data.get("average_time_between_actions", 0)
    report += f"\nAverage time between actions: {avg_time:.2f} seconds\n"

    total_time = behavior_data.get("total_session_time", 0)
    report += f"Total session time: {total_time:.2f} seconds\n"

    # Add insights
    report += "\nInsights:\n"

    if total_actions > 0:
        # Most common action
        if action_types:
            most_common_action = max(
                action_types.items(), key=lambda x: x[1] if x[1] is not None else 0
            )
        else:
            most_common_action = ("none", 0)
        report += f"  Most common action: {most_common_action[0]} ({most_common_action[1]} times)\n"

        # Action frequency
        if total_time > 0:
            actions_per_minute = (total_actions / total_time) * 60
            report += f"  Actions per minute: {actions_per_minute:.2f}\n"

        # Edit vs. run ratio
        edit_count = action_types.get("edit", 0)
        run_count = action_types.get("run", 0)
        if run_count > 0:
            edit_run_ratio = edit_count / run_count
            report += f"  Edit to run ratio: {edit_run_ratio:.2f}\n"

            if edit_run_ratio > 5:
                report += "  User makes many edits before running the code, suggesting careful planning.\n"
            elif edit_run_ratio < 1:
                report += "  User runs the code frequently, suggesting an exploratory approach.\n"

    return report


def save_behavior_visualization(behavior_data: Dict[str, Any], output_path: str) -> None:
    """Save behavior visualization to file.

    Args:
        behavior_data: User behavior data.
        output_path: Path to save the visualization.
    """
    logger.info(f"Saving behavior visualization to {output_path}")

    if not behavior_data or "action_types" not in behavior_data:
        logger.warning("No behavior data to visualize")
        return

    action_types = behavior_data.get("action_types", {})
    if not action_types:
        logger.warning("No action types in behavior data")
        return

    # Create bar chart of action types
    plt.figure(figsize=(10, 6))

    types = list(action_types.keys())
    counts = list(action_types.values())

    plt.bar(types, counts, color="blue")
    plt.title("User Actions by Type")
    plt.xlabel("Action Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save to file
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Behavior visualization saved to {output_path}")
