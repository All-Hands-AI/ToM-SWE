#!/usr/bin/env python3
"""
Flask web application for viewing trajectory data.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, redirect, render_template, request, url_for

app = Flask(__name__, template_folder="../templates")

# Use module-level caches instead of global statements
studio_metadata_cache: Dict[str, Dict[str, Any]] = {}
conversation_metadata_cache: Dict[str, Dict[str, Any]] = {}

# Cache for individual conversations
_individual_conversation_cache: Dict[str, Dict[str, Any]] = {}

# Use sets for source comparisons
AGENT_SOURCES = {"agent", "assistant"}
SYSTEM_SOURCES = {"environment", "system"}

# Global variable to store the target user ID
TARGET_USER_ID = None


def load_studio_results_metadata() -> Dict[str, Dict[str, Any]]:
    """Load studio results metadata from CSV file with caching."""
    if studio_metadata_cache:
        return studio_metadata_cache

    metadata_path = Path("./data/studio_results_20250604_1645.csv")
    metadata: Dict[str, Dict[str, Any]] = {}

    if not metadata_path.exists():
        print(f"Warning: Studio results metadata not found at {metadata_path}")
        studio_metadata_cache.update(metadata)
        return metadata

    try:
        with open(metadata_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                conv_id = row.get("conversation_id", "")
                if conv_id:
                    metadata[conv_id] = {
                        "github_user_id": row.get("github_user_id", ""),
                        "selected_repository": row.get("selected_repository", ""),
                        "title": row.get("title", ""),
                        "trigger": row.get("trigger", ""),
                        "user_id": row.get("user_id", ""),
                        "accumulated_cost": float(row.get("accumulated_cost", 0) or 0),
                        "prompt_tokens": int(row.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(row.get("completion_tokens", 0) or 0),
                        "total_tokens": int(row.get("total_tokens", 0) or 0),
                        "pr_number": row.get("pr_number", ""),
                        "created_at": row.get("created_at", ""),
                        "last_updated_at": row.get("last_updated_at", ""),
                        "selected_branch": row.get("selected_branch", ""),
                    }
    except Exception as e:
        print(f"Error loading studio results metadata: {e}")

    studio_metadata_cache.update(metadata)
    return studio_metadata_cache


def load_conversation_metadata_only(user_id: str) -> Dict[str, Dict[str, Any]]:
    """Load only conversation metadata without event data for faster initial loading."""
    if conversation_metadata_cache:
        return conversation_metadata_cache

    data_path = Path(f"./data/processed_data/{user_id}.json")

    if not data_path.exists():
        raise FileNotFoundError(f"Trajectory data not found: {data_path}")

    print(f"Loading conversation metadata for user: {user_id}")

    metadata = {}
    with open(data_path) as f:
        data = json.load(f)

        for conv_id, conv_data in data.items():
            # Only extract metadata, don't process events
            events = conv_data.get("convo_events", [])
            metadata[conv_id] = {
                "conv_id": conv_id,
                "convo_start": conv_data.get("convo_start", ""),
                "convo_end": conv_data.get("convo_end", ""),
                "event_count": len(events),
                "user_message_count": count_real_user_messages_lazy(events),
            }

    conversation_metadata_cache.update(metadata)
    print(f"Loaded metadata for {len(metadata)} conversations for user {user_id}")
    return metadata


def load_single_conversation(user_id: str, conv_id: str) -> Optional[Dict[str, Any]]:
    """Load a single conversation's full data with caching."""
    cache_key = f"{user_id}:{conv_id}"

    if cache_key in _individual_conversation_cache:
        return _individual_conversation_cache[cache_key]

    data_path = Path(f"./data/processed_data/{user_id}.json")

    if not data_path.exists():
        raise FileNotFoundError(f"Trajectory data not found: {data_path}")

    # Load only the specific conversation from the file
    with open(data_path) as f:
        data = json.load(f)

        if conv_id not in data:
            return None

        conversation_data: Dict[str, Any] = data[conv_id]
        _individual_conversation_cache[cache_key] = conversation_data
        return conversation_data


def count_real_user_messages_lazy(events: List[Dict[str, Any]]) -> int:
    """Count user messages excluding recall info - optimized for metadata loading."""
    count = 0
    for event in events:
        if event.get("source", "") == "user":
            # Quick check for recall patterns without full processing
            content = event.get("content", "").strip().lower()
            if not any(
                pattern in content
                for pattern in [
                    "recall info:",
                    "recalling:",
                    "recalled information:",
                    "from memory:",
                    "previously discussed:",
                    "recall from",
                    "recallinfo",
                    "[recall]",
                ]
            ):
                count += 1
    return count


def is_recall_info_message(event: Dict[str, Any]) -> bool:
    """Check if a message is recall info and should not count as user message."""
    content = event.get("content", "").strip().lower()
    action = event.get("action", "")

    # Check for recall patterns
    recall_patterns = [
        "recall info:",
        "recalling:",
        "recalled information:",
        "from memory:",
        "previously discussed:",
        "recall from",
        "recallinfo",
        "[recall]",
    ]

    for pattern in recall_patterns:
        if pattern in content:
            return True

    # Check if action indicates recall
    if action in ["recall", "memory_recall", "recall_info"]:
        return True

    return False


def count_real_user_messages(events: List[Dict[str, Any]]) -> int:
    """Count user messages excluding recall info."""
    count = 0
    for event in events:
        if event.get("source", "") == "user" and not is_recall_info_message(event):
            count += 1
    return count


def sort_conversations_by_time(
    trajectory_data: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Sort conversations by start time, earliest first."""
    conversations = []
    for conv_id, conv_data in trajectory_data.items():
        try:
            # Parse the start time
            start_time = datetime.fromisoformat(conv_data.get("convo_start", ""))
            conversations.append((conv_id, conv_data, start_time))
        except (ValueError, TypeError):
            # If timestamp parsing fails, put at the end
            conversations.append((conv_id, conv_data, datetime.max))

    # Sort by start time
    conversations.sort(key=lambda x: x[2])

    # Return just the conv_id and conv_data
    return [(conv_id, conv_data) for conv_id, conv_data, _ in conversations]


def sort_conversations_by_user_messages(
    trajectory_data: Dict[str, Dict[str, Any]], descending: bool = True
) -> List[Tuple[str, Dict[str, Any]]]:
    """Sort conversations by number of user messages."""
    conversations = []
    for conv_id, conv_data in trajectory_data.items():
        user_msg_count = conv_data.get("user_message_count", 0)
        conversations.append((conv_id, conv_data, user_msg_count))

    # Sort by user message count
    conversations.sort(key=lambda x: x[2] if x[2] is not None else 0, reverse=descending)

    # Return just the conv_id and conv_data
    return [(conv_id, conv_data) for conv_id, conv_data, _ in conversations]


def format_datetime(timestamp_str: str) -> str:
    """Format timestamp string to a readable format."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return timestamp_str


def calculate_duration(start_str: str, end_str: str) -> str:
    """Calculate duration between start and end timestamps."""
    try:
        start_dt = datetime.fromisoformat(start_str)
        end_dt = datetime.fromisoformat(end_str)
        duration = end_dt - start_dt

        # Format duration nicely
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except (ValueError, TypeError):
        return "Unknown"


def calculate_conversation_stats(conv_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate basic statistics for a conversation."""
    events = conv_data.get("convo_events", [])
    start_time = conv_data.get("convo_start", "Unknown")
    end_time = conv_data.get("convo_end", "Unknown")

    stats = {
        "total_events": len(events),
        "user_messages": 0,
        "real_user_messages": 0,  # Excluding recall info
        "agent_messages": 0,
        "environment_messages": 0,
        "system_messages": 0,
        "actions": 0,
        "unique_sources": set(),
        "first_event_id": None,
        "last_event_id": None,
        "start_time": start_time,
        "end_time": end_time,
        "duration": calculate_duration(start_time, end_time),
        "formatted_start": format_datetime(start_time),
        "formatted_end": format_datetime(end_time),
    }

    if not events:
        return stats

    # Get first and last event IDs
    stats["first_event_id"] = events[0].get("id", "N/A")
    stats["last_event_id"] = events[-1].get("id", "N/A")

    # Count different types of events
    for event in events:
        source = event.get("source", "unknown")

        stats["unique_sources"].add(source)

        if source == "user":
            stats["user_messages"] += 1
            if not is_recall_info_message(event):
                stats["real_user_messages"] += 1
        elif source in AGENT_SOURCES:
            stats["agent_messages"] += 1
        elif source in SYSTEM_SOURCES:
            stats["environment_messages"] += 1

        # For processed events, count all non-user messages as actions
        if source not in {"user", "agent", "assistant", "environment", "system"}:
            stats["actions"] += 1

    # Calculate rounds (approximate user-agent interaction cycles)
    stats["estimated_rounds"] = max(stats["real_user_messages"], 1)

    return stats


def filter_user_messages(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter events to only include user messages."""
    return [event for event in events if event.get("source", "") == "user"]


def get_color_for_source(source: str) -> str:
    """Get a consistent color class for each source type."""
    color_map = {
        "user": "primary",
        "agent": "success",
        "environment": "info",
        "system": "warning",
        "unknown": "secondary",
    }
    return color_map.get(source, "secondary")


def truncate_content(content: str, max_length: int = 500) -> str:
    """Truncate content if too long."""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "\n\n... (content truncated for readability)"


def enrich_conversation_with_metadata(
    conv_id: str, conv_data: Dict[str, Any], metadata: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Enrich conversation data with studio results metadata."""
    enriched = conv_data.copy()

    if conv_id in metadata:
        meta = metadata[conv_id]
        enriched["metadata"] = meta
        enriched["is_gui"] = meta.get("trigger", "").lower() == "gui"
        enriched["repository"] = meta.get("selected_repository", "Unknown")
        enriched["title"] = meta.get("title", "Untitled")
        enriched["cost"] = meta.get("accumulated_cost", 0)
        enriched["tokens"] = {
            "prompt": meta.get("prompt_tokens", 0),
            "completion": meta.get("completion_tokens", 0),
            "total": meta.get("total_tokens", 0),
        }
    else:
        enriched["metadata"] = {}
        enriched["is_gui"] = None  # Unknown
        enriched["repository"] = "Unknown"
        enriched["title"] = "Untitled"
        enriched["cost"] = 0
        enriched["tokens"] = {"prompt": 0, "completion": 0, "total": 0}

    return enriched


@app.route("/")
def index():
    """Main page - redirect to the target user's conversations."""
    if TARGET_USER_ID:
        return redirect(url_for("user_conversations", user_id=TARGET_USER_ID))
    else:
        return render_template(
            "error.html",
            error="No user ID specified. Please provide a user ID when starting the server.",
        )


@app.route("/user/<user_id>")
def user_conversations(user_id: str):
    """List all conversations for a user with pagination."""
    try:
        # Get pagination parameters
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))  # Default 50 conversations per page

        # Load only metadata for listing
        conversation_metadata = load_conversation_metadata_only(user_id)
        studio_metadata = load_studio_results_metadata()

        # Get filter and sort parameters
        gui_only = request.args.get("gui_only", "false").lower() == "true"
        sort_by = request.args.get("sort_by", "time")  # 'time' or 'user_messages'

        # Convert metadata dict to list and enrich
        conversation_list = []
        for conv_id, conv_meta in conversation_metadata.items():
            # Enrich with studio metadata
            enriched_meta = enrich_conversation_metadata(conv_id, conv_meta, studio_metadata)

            # Apply GUI filter if requested
            if gui_only and not enriched_meta.get("is_gui", False):
                continue

            conversation_list.append(enriched_meta)

        # Sort conversations
        if sort_by == "user_messages":
            conversation_list.sort(key=lambda x: x.get("user_message_count", 0), reverse=True)
        else:
            conversation_list.sort(key=lambda x: x.get("convo_start", ""))

        # Calculate pagination
        total_conversations = len(conversation_list)
        total_pages = (total_conversations + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_conversations = conversation_list[start_idx:end_idx]

        # Process conversations for display
        processed_conversations = []
        for i, conv_meta in enumerate(paginated_conversations, start=start_idx):
            stats = calculate_conversation_stats(conv_meta)

            processed_conversations.append(
                {
                    "index": i,
                    "conv_id": conv_meta["conv_id"],
                    "stats": stats,
                    "metadata": conv_meta.get("studio_metadata", {}),
                    "is_gui": conv_meta.get("is_gui"),
                    "repository": conv_meta.get("repository", "Unknown"),
                    "title": conv_meta.get("title", "Untitled"),
                    "cost": conv_meta.get("cost", 0),
                    "tokens": conv_meta.get("tokens", {}),
                    "event_count": conv_meta.get("event_count", 0),
                }
            )

        # Pagination info
        pagination = {
            "page": page,
            "per_page": per_page,
            "total": total_conversations,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages,
            "prev_page": page - 1,
            "next_page": page + 1,
            "start_idx": start_idx + 1,
            "end_idx": min(end_idx, total_conversations),
        }

        return render_template(
            "conversations.html",
            user_id=user_id,
            conversations=processed_conversations,
            pagination=pagination,
            total_conversations=total_conversations,
            total_original=len(conversation_metadata),
            gui_only=gui_only,
            sort_by=sort_by,
        )

    except FileNotFoundError:
        return render_template("error.html", error=f"Trajectory data not found for user: {user_id}")
    except Exception as e:
        return render_template("error.html", error=f"Error loading data: {e!s}")


@app.route("/user/<user_id>/conversation/<int:conv_index>")
def view_conversation(user_id: str, conv_index: int):
    """View a specific conversation with lazy loading."""
    try:
        # Get filter and sort parameters to determine which conversation to show
        gui_only = request.args.get("gui_only", "false").lower() == "true"
        sort_by = request.args.get("sort_by", "time")
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))

        # Get conversation metadata to determine the correct conversation
        conversation_metadata = load_conversation_metadata_only(user_id)
        studio_metadata = load_studio_results_metadata()

        # Filter and sort metadata the same way as the list view
        filtered_metadata = []
        for conv_id, conv_meta in conversation_metadata.items():
            enriched_meta = enrich_conversation_metadata(conv_id, conv_meta, studio_metadata)

            if gui_only and not enriched_meta.get("is_gui", False):
                continue

            filtered_metadata.append(enriched_meta)

        # Sort the filtered metadata
        if sort_by == "user_messages":
            filtered_metadata.sort(key=lambda x: x.get("user_message_count", 0), reverse=True)
        else:
            filtered_metadata.sort(key=lambda x: x.get("convo_start", ""))

        if conv_index < 0 or conv_index >= len(filtered_metadata):
            return render_template("error.html", error=f"Invalid conversation index: {conv_index}")

        # Now load the full conversation data for the specific conversation
        target_conv_meta = filtered_metadata[conv_index]
        conv_id = target_conv_meta["conv_id"]
        conv_data = load_single_conversation(user_id, conv_id)

        if not conv_data:
            return render_template("error.html", error=f"Conversation not found: {conv_id}")

        enriched_conv_data = enrich_conversation_with_metadata(conv_id, conv_data, studio_metadata)
        stats = calculate_conversation_stats(enriched_conv_data)

        # Get display parameters
        user_only = request.args.get("user_only", "false").lower() == "true"

        events = enriched_conv_data.get("convo_events", [])
        if user_only:
            events = filter_user_messages(events)

        # Process events for display
        processed_events = []
        for event in events:
            is_recall = is_recall_info_message(event)
            processed_event = {
                "id": event.get("id", "N/A"),
                "source": event.get("source", "unknown"),
                "action": event.get(
                    "action", "message"
                ),  # Default to 'message' for processed events
                "content": event.get("content", ""),
                "truncated_content": truncate_content(event.get("content", "")),
                "color_class": get_color_for_source(event.get("source", "unknown")),
                "is_recall": is_recall,
            }
            processed_events.append(processed_event)

        # Navigation info
        navigation = {
            "current_index": conv_index,
            "total": len(filtered_metadata),
            "has_prev": conv_index > 0,
            "has_next": conv_index < len(filtered_metadata) - 1,
            "prev_index": conv_index - 1,
            "next_index": conv_index + 1,
        }

        return render_template(
            "conversation.html",
            user_id=user_id,
            conv_id=conv_id,
            conv_index=conv_index,
            stats=stats,
            metadata=enriched_conv_data.get("metadata", {}),
            is_gui=enriched_conv_data.get("is_gui"),
            repository=enriched_conv_data.get("repository", "Unknown"),
            title=enriched_conv_data.get("title", "Untitled"),
            cost=enriched_conv_data.get("cost", 0),
            tokens=enriched_conv_data.get("tokens", {}),
            events=processed_events,
            navigation=navigation,
            user_only=user_only,
            gui_only=gui_only,
            sort_by=sort_by,
            page=page,
            per_page=per_page,
            filtered_count=len(events),
        )

    except FileNotFoundError:
        return render_template("error.html", error=f"Trajectory data not found for user: {user_id}")
    except Exception as e:
        return render_template("error.html", error=f"Error loading conversation: {e!s}")


@app.route("/api/users")
def api_list_users():
    """API endpoint to return the target user."""
    if TARGET_USER_ID:
        return jsonify({"users": [TARGET_USER_ID]})
    else:
        return jsonify({"users": []})


def get_conversation_metadata(user_id: str, conv_id: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a specific conversation without loading full event data."""
    conversation_metadata = load_conversation_metadata_only(user_id)
    return conversation_metadata.get(conv_id)


def get_all_conversation_metadata(user_id: str) -> List[Dict[str, Any]]:
    """Get metadata for all conversations without loading full event data."""
    conversation_metadata = load_conversation_metadata_only(user_id)
    return list(conversation_metadata.values())


def get_single_conversation(user_id: str, conv_id: str) -> Optional[Dict[str, Any]]:
    """Get a single conversation's full data."""
    return load_single_conversation(user_id, conv_id)


def enrich_conversation_metadata(
    conv_id: str, conv_meta: Dict[str, Any], studio_metadata: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Enrich conversation metadata with studio results metadata."""
    enriched = conv_meta.copy()

    if conv_id in studio_metadata:
        meta = studio_metadata[conv_id]
        enriched["studio_metadata"] = meta
        enriched["is_gui"] = meta.get("trigger", "").lower() == "gui"
        enriched["repository"] = meta.get("selected_repository", "Unknown")
        enriched["title"] = meta.get("title", "Untitled")
        enriched["cost"] = meta.get("accumulated_cost", 0)
        enriched["tokens"] = {
            "prompt": meta.get("prompt_tokens", 0),
            "completion": meta.get("completion_tokens", 0),
            "total": meta.get("total_tokens", 0),
        }
    else:
        enriched["studio_metadata"] = {}
        enriched["is_gui"] = None  # Unknown
        enriched["repository"] = "Unknown"
        enriched["title"] = "Untitled"
        enriched["cost"] = 0
        enriched["tokens"] = {"prompt": 0, "completion": 0, "total": 0}

    return enriched


def calculate_conversation_stats_from_metadata(conv_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate basic statistics from conversation metadata (without loading full events)."""
    start_time = conv_meta.get("convo_start", "Unknown")
    end_time = conv_meta.get("convo_end", "Unknown")

    stats = {
        "total_events": conv_meta.get("event_count", 0),
        "real_user_messages": conv_meta.get("user_message_count", 0),
        "user_messages": conv_meta.get("user_message_count", 0),  # Template expects this
        "agent_messages": 0,  # Can't calculate without loading events, default to 0
        "environment_messages": 0,  # Can't calculate without loading events, default to 0
        "system_messages": 0,  # Can't calculate without loading events, default to 0
        "actions": 0,  # Can't calculate without loading events, default to 0
        "start_time": start_time,
        "end_time": end_time,
        "duration": calculate_duration(start_time, end_time),
        "formatted_start": format_datetime(start_time),
        "formatted_end": format_datetime(end_time),
        "estimated_rounds": max(conv_meta.get("user_message_count", 0), 1),
    }

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web-based trajectory viewer")
    parser.add_argument("--user-id", required=True, help="User ID to load trajectory data for")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    # Set the global user ID
    TARGET_USER_ID = args.user_id

    # Validate that the user data exists
    data_path = Path(f"./data/processed_data/{TARGET_USER_ID}.json")
    if not data_path.exists():
        print(f"Error: Trajectory data not found for user '{TARGET_USER_ID}' at {data_path}")
        sys.exit(1)

    print(
        f"Starting web trajectory viewer for user '{TARGET_USER_ID}' on http://{args.host}:{args.port}"
    )
    print("Available endpoints:")
    print(f"  - Main page: http://{args.host}:{args.port}/")
    print(f"  - User conversations: http://{args.host}:{args.port}/user/{TARGET_USER_ID}")
    print(
        f"  - Specific conversation: http://{args.host}:{args.port}/user/{TARGET_USER_ID}/conversation/<index>"
    )

    app.run(host=args.host, port=args.port, debug=args.debug)
