#!/usr/bin/env python3
"""
Rich-based trajectory viewer for processed trajectory data.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

# Constants
MAX_CONTENT_LENGTH = 500
MIN_CONVERSATION_INDEX = 0
MAX_CONVERSATION_INDEX = 2
MIN_COMMAND_PARTS = 2


@dataclass
class ConversationDisplayConfig:
    """Configuration for displaying a conversation."""

    conv_id: str
    conv_data: Dict[str, Any]
    conv_index: int
    total_convos: int
    user_only: bool = False


def load_trajectory_data(user_id: str) -> Dict[str, Dict[str, Any]]:
    """Load trajectory data from processed_data/{user_id}.json."""
    data_path = Path(f"./data/processed_data/{user_id}.json")

    if not data_path.exists():
        raise FileNotFoundError(f"Trajectory data not found: {data_path}")

    with open(data_path) as f:
        return json.load(f)  # type: ignore[no-any-return]


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


def get_color_for_source(source: str, color_map: Dict[str, str]) -> str:
    """Get a consistent color for each source type."""
    if source not in color_map:
        colors = ["cyan", "magenta", "yellow", "green", "blue", "red"]
        color_map[source] = colors[len(color_map) % len(colors)]
    return color_map[source]


def format_content(content: str, source: str = "", action: str = "", max_length: int = 500) -> str:
    """Format content with truncation if too long, but preserve user messages."""
    # Don't truncate user messages or important content
    if source == "user" or action == "message":
        return escape(content)

    # For other content, use a more generous limit
    if len(content) <= max_length:
        return escape(content)

    return escape(content[:max_length]) + "\n\n[dim]... (content truncated for readability)[/dim]"


def filter_user_messages(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter events to only include user messages."""
    return [event for event in events if event.get("source", "") == "user"]


def calculate_conversation_stats(conv_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate basic statistics for a conversation."""
    events = conv_data.get("convo_events", [])
    start_time = conv_data.get("convo_start", "Unknown")
    end_time = conv_data.get("convo_end", "Unknown")

    stats = {
        "total_events": len(events),
        "user_messages": 0,
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
        action = event.get("action", "unknown")

        stats["unique_sources"].add(source)

        if source == "user":
            stats["user_messages"] += 1
        elif source == "agent":
            stats["agent_messages"] += 1
        elif source == "environment":
            stats["environment_messages"] += 1

        if action == "system":
            stats["system_messages"] += 1
        elif action in ["run", "edit", "write", "read"]:
            stats["actions"] += 1

    # Calculate rounds (approximate user-agent interaction cycles)
    stats["estimated_rounds"] = max(stats["user_messages"], 1)

    return stats


def display_conversation_summary(stats: Dict[str, Any], console: Console):
    """Display a summary box with conversation statistics."""
    summary_text = f"""[bold]📊 Conversation Summary[/bold]
• Total Events: [yellow]{stats['total_events']}[/yellow]
• User Messages: [cyan]{stats['user_messages']}[/cyan]
• Agent Messages: [green]{stats['agent_messages']}[/green]
• Environment Messages: [blue]{stats['environment_messages']}[/blue]
• Estimated Rounds: [magenta]{stats['estimated_rounds']}[/magenta]
• Start Time: [bright_green]{stats['formatted_start']}[/bright_green]
• End Time: [bright_red]{stats['formatted_end']}[/bright_red]
• Duration: [bright_yellow]{stats['duration']}[/bright_yellow]
• Event Range: [dim]{stats['first_event_id']} → {stats['last_event_id']}[/dim]"""

    console.print(
        Panel(
            summary_text,
            title="[bold blue]Stats[/bold blue]",
            style="dim white",
            border_style="blue",
            padding=(0, 1),
        )
    )
    console.print()


def display_conversation(config: ConversationDisplayConfig, console: Console) -> None:
    """Display a single conversation's events."""
    console.clear()

    # Calculate stats for the conversation
    original_stats = calculate_conversation_stats(config.conv_data)

    # Get events and filter if needed
    events = config.conv_data.get("convo_events", [])
    if config.user_only:
        events = filter_user_messages(events)
        title_suffix = " [USER MESSAGES ONLY]"
    else:
        title_suffix = ""

    console.print(
        Rule(
            f"[bold blue]Conversation {config.conv_index + 1}/{config.total_convos}: "
            f"{escape(config.conv_id)}{title_suffix}[/bold blue]"
        )
    )

    # Display conversation summary
    display_conversation_summary(original_stats, console)

    if not events:
        console.print("[yellow]No user messages found in this conversation.[/yellow]")
        console.print()
        return

    color_map: Dict[str, str] = {}

    for event in events:
        event_id = event.get("id", "N/A")
        source = event.get("source", "unknown")
        action = event.get("action", "unknown")
        content = event.get("content", "")

        # Get color for source
        source_color = get_color_for_source(source, color_map)

        # Format title
        title = (
            f"[{source_color}]{escape(source)}[/{source_color}] | "
            f"{escape(action)} | ID: {escape(str(event_id))}"
        )

        # Format content
        formatted_content = format_content(content, source, action)

        # Choose panel style based on action type
        if action == "system":
            style = "dim blue"
        elif action == "message":
            style = source_color
        elif action == "change_agent_state":
            style = "yellow"
        else:
            style = "white"

        console.print(Panel(formatted_content, title=title, style=style, title_align="left"))

    console.print()  # Add spacing


def display_navigation_help(console: Console):
    """Display navigation help."""
    help_table = Table(title="Navigation Commands", show_header=True, header_style="bold magenta")
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description", style="white")

    help_table.add_row("next / n", "Show next conversation")
    help_table.add_row("prev / p", "Show previous conversation")
    help_table.add_row("list / l", "Show conversation list")
    help_table.add_row("jump <num>", "Jump to conversation number")
    help_table.add_row("user / u", "Toggle user messages only mode")
    help_table.add_row("help / h", "Show this help")
    help_table.add_row("quit / q", "Exit viewer")

    console.print(help_table)
    console.print()


def display_conversation_list(
    conversations: List[Tuple[str, Dict[str, Any]]], console: Console, current_index: int = -1
):
    """Display a paginated list of conversations."""
    console.clear()

    # Create table
    table = Table(title="Available Conversations", show_header=True, header_style="bold green")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Conversation ID", style="white")
    table.add_column("Start Time", style="bright_green")
    table.add_column("Duration", style="yellow")
    table.add_column("Events", style="yellow", justify="right")
    table.add_column("Status", style="magenta")

    for i, (conv_id, conv_data) in enumerate(conversations):
        stats = calculate_conversation_stats(conv_data)
        event_count = len(conv_data.get("convo_events", []))
        status = "→ Current" if i == current_index else ""
        style = "bold green" if i == current_index else None

        table.add_row(
            str(i + 1),
            escape(conv_id),
            stats["formatted_start"],
            stats["duration"],
            str(event_count),
            status,
            style=style,
        )

    console.print(table)
    console.print()


def handle_next_command(
    current_index: int,
    conversations: List[Tuple[str, Dict[str, Any]]],
    user_only_mode: bool,
    console: Console,
) -> Tuple[int, bool]:
    """Handle the 'next' command."""
    if current_index < len(conversations) - 1:
        current_index += 1
        conv_id, conv_data = conversations[current_index]
        config = ConversationDisplayConfig(
            conv_id=conv_id,
            conv_data=conv_data,
            conv_index=current_index,
            total_convos=len(conversations),
            user_only=user_only_mode,
        )
        display_conversation(config, console)
    else:
        console.print("[yellow]Already at the last conversation![/yellow]")
    return current_index, user_only_mode


def handle_prev_command(
    current_index: int,
    conversations: List[Tuple[str, Dict[str, Any]]],
    user_only_mode: bool,
    console: Console,
) -> Tuple[int, bool]:
    """Handle the 'prev' command."""
    if current_index > 0:
        current_index -= 1
        conv_id, conv_data = conversations[current_index]
        config = ConversationDisplayConfig(
            conv_id=conv_id,
            conv_data=conv_data,
            conv_index=current_index,
            total_convos=len(conversations),
            user_only=user_only_mode,
        )
        display_conversation(config, console)
    else:
        console.print("[yellow]Already at the first conversation![/yellow]")
    return current_index, user_only_mode


def handle_jump_command(
    command: str,
    current_index: int,
    conversations: List[Tuple[str, Dict[str, Any]]],
    user_only_mode: bool,
    console: Console,
) -> Tuple[int, bool]:
    """Handle the 'jump' command."""
    try:
        parts = command.split()
        if len(parts) >= MIN_COMMAND_PARTS:
            target = int(parts[1]) - 1  # Convert to 0-based index
            if 0 <= target < len(conversations):
                current_index = target
                conv_id, conv_data = conversations[current_index]
                config = ConversationDisplayConfig(
                    conv_id=conv_id,
                    conv_data=conv_data,
                    conv_index=current_index,
                    total_convos=len(conversations),
                    user_only=user_only_mode,
                )
                display_conversation(config, console)
            else:
                console.print(
                    f"[red]Invalid conversation number. Range: 1-{len(conversations)}[/red]"
                )
        else:
            console.print("[red]Usage: jump <number>[/red]")
    except ValueError:
        console.print("[red]Invalid number![/red]")
    return current_index, user_only_mode


def handle_user_command(
    current_index: int,
    conversations: List[Tuple[str, Dict[str, Any]]],
    user_only_mode: bool,
    console: Console,
) -> Tuple[int, bool]:
    """Handle the 'user' command."""
    user_only_mode = not user_only_mode
    mode_status = "enabled" if user_only_mode else "disabled"
    console.print(f"[green]User messages only mode {mode_status}![/green]")
    # Redisplay current conversation with new mode
    conv_id, conv_data = conversations[current_index]
    config = ConversationDisplayConfig(
        conv_id=conv_id,
        conv_data=conv_data,
        conv_index=current_index,
        total_convos=len(conversations),
        user_only=user_only_mode,
    )
    display_conversation(config, console)
    return current_index, user_only_mode


def handle_digit_command(
    command: str,
    current_index: int,
    conversations: List[Tuple[str, Dict[str, Any]]],
    user_only_mode: bool,
    console: Console,
) -> Tuple[int, bool]:
    """Handle direct number input."""
    try:
        target = int(command) - 1
        if 0 <= target < len(conversations):
            current_index = target
            conv_id, conv_data = conversations[current_index]
            config = ConversationDisplayConfig(
                conv_id=conv_id,
                conv_data=conv_data,
                conv_index=current_index,
                total_convos=len(conversations),
                user_only=user_only_mode,
            )
            display_conversation(config, console)
        else:
            console.print(f"[red]Invalid conversation number. Range: 1-{len(conversations)}[/red]")
    except ValueError:
        console.print("[red]Invalid number![/red]")
    return current_index, user_only_mode


def handle_command(
    command: str,
    current_index: int,
    conversations: List[Tuple[str, Dict[str, Any]]],
    user_only_mode: bool,
    console: Console,
) -> Tuple[int, bool, bool]:
    """Handle user command in interactive viewer.

    Args:
        command: The user's command
        current_index: Current conversation index
        conversations: List of conversations
        user_only_mode: Whether to show only user messages
        console: Console for output

    Returns:
        Tuple containing:
        - Updated current_index
        - Updated user_only_mode
        - Whether to exit the viewer
    """
    if command in ["quit", "q", "exit"]:
        console.print("[green]Goodbye![/green]")
        return current_index, user_only_mode, True  # True indicates should exit

    elif command in ["next", "n"]:
        current_index, user_only_mode = handle_next_command(
            current_index, conversations, user_only_mode, console
        )

    elif command in ["prev", "p", "previous"]:
        current_index, user_only_mode = handle_prev_command(
            current_index, conversations, user_only_mode, console
        )

    elif command in ["list", "l"]:
        display_conversation_list(conversations, console, current_index)

    elif command in ["user", "u"]:
        current_index, user_only_mode = handle_user_command(
            current_index, conversations, user_only_mode, console
        )

    elif command in ["help", "h"]:
        display_navigation_help(console)

    elif command.startswith("jump ") or command.startswith("j "):
        current_index, user_only_mode = handle_jump_command(
            command, current_index, conversations, user_only_mode, console
        )

    elif command.isdigit():
        current_index, user_only_mode = handle_digit_command(
            command, current_index, conversations, user_only_mode, console
        )

    else:
        console.print(
            f"[red]Unknown command: '{command}'. " f"Type 'help' for available commands.[/red]"
        )

    return current_index, user_only_mode, False  # False indicates should continue


def interactive_viewer(user_id: str) -> None:
    """Interactive trajectory viewer with navigation."""
    console = Console()

    try:
        # Load trajectory data
        trajectory_data = load_trajectory_data(user_id)

        # Sort conversations by start time
        conversations = sort_conversations_by_time(trajectory_data)

        if not conversations:
            console.print("[red]No conversations found![/red]")
            return

        console.print(
            f"\n[bold green]Interactive Trajectory Viewer for User: {user_id}[/bold green]"
        )
        console.print(f"Total conversations: {len(conversations)} (sorted by start time)")
        console.print("[dim]Type 'help' for navigation commands[/dim]\n")

        current_index = 0
        user_only_mode = False

        # Display first conversation
        conv_id, conv_data = conversations[current_index]
        config = ConversationDisplayConfig(
            conv_id=conv_id,
            conv_data=conv_data,
            conv_index=current_index,
            total_convos=len(conversations),
            user_only=user_only_mode,
        )
        display_conversation(config, console)

        while True:
            # Show navigation prompt with mode indicator
            mode_indicator = " [USER ONLY]" if user_only_mode else ""
            console.print(
                f"[dim]Conversation {current_index + 1}/{len(conversations)}"
                f"{mode_indicator} | Type 'help' for commands[/dim]"
            )
            command = Prompt.ask("[bold cyan]Navigation", default="next").strip().lower()

            current_index, user_only_mode, should_exit = handle_command(
                command, current_index, conversations, user_only_mode, console
            )

            if should_exit:
                break

            # Display current conversation
            conv_id, conv_data = conversations[current_index]
            config = ConversationDisplayConfig(
                conv_id=conv_id,
                conv_data=conv_data,
                conv_index=current_index,
                total_convos=len(conversations),
                user_only=user_only_mode,
            )
            display_conversation(config, console)

    except FileNotFoundError as e:
        console.print(f"[red]Error: {escape(str(e))}[/red]")
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON: {escape(str(e))}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {escape(str(e))}[/red]")


def display_trajectory(
    user_id: str, conversation_id: Optional[str] = None, user_only: bool = False
) -> None:
    """Display trajectory data for a user."""
    console = Console()

    try:
        # Load trajectory data
        trajectory_data = load_trajectory_data(user_id)

        # Sort conversations by start time
        conversations = sort_conversations_by_time(trajectory_data)

        mode_text = " [USER MESSAGES ONLY]" if user_only else ""
        console.print(
            f"\n[bold green]Trajectory Viewer for User: {user_id}{mode_text}[/bold green]"
        )
        console.print(f"Total conversations: {len(conversations)} (sorted by start time)\n")

        # Display specific conversation or all conversations
        if conversation_id:
            # Find the conversation in the sorted list
            found_conv = None
            for i, (conv_id, conv_data) in enumerate(conversations):
                if conv_id == conversation_id:
                    found_conv = (i, conv_data)
                    break

            if found_conv:
                i, conv_data = found_conv
                config = ConversationDisplayConfig(
                    conv_id=conversation_id,
                    conv_data=conv_data,
                    conv_index=i,
                    total_convos=len(conversations),
                    user_only=user_only,
                )
                display_conversation(config, console)
            else:
                console.print(f"[red]Conversation {conversation_id} not found![/red]")
                console.print(
                    f"Available conversations: {[conv_id for conv_id, _ in conversations]}"
                )
        else:
            # Display all conversations
            for i, (conv_id, conv_data) in enumerate(conversations):
                config = ConversationDisplayConfig(
                    conv_id=conv_id,
                    conv_data=conv_data,
                    conv_index=i,
                    total_convos=len(conversations),
                    user_only=user_only,
                )
                display_conversation(config, console)

    except FileNotFoundError as e:
        console.print(f"[red]Error: {escape(str(e))}[/red]")
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON: {escape(str(e))}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {escape(str(e))}[/red]")


def main():
    """Main entry point for the trajectory viewer."""
    parser = argparse.ArgumentParser(description="Rich-based trajectory viewer")
    parser.add_argument("user_id", help="User ID to display trajectory for")
    parser.add_argument("--conversation", "-c", help="Specific conversation ID to display")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available conversations only"
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive viewer")
    parser.add_argument("--user-only", "-u", action="store_true", help="Show only user messages")

    args = parser.parse_args()

    if args.interactive:
        # Start interactive mode
        interactive_viewer(args.user_id)
    elif args.list:
        # Just list available conversations
        try:
            trajectory_data = load_trajectory_data(args.user_id)
            conversations = sort_conversations_by_time(trajectory_data)
            console = Console()
            console.print(
                f"\n[bold green]Available conversations for user {args.user_id} (sorted by start time):[/bold green]"
            )
            for i, (conv_id, conv_data) in enumerate(conversations):
                stats = calculate_conversation_stats(conv_data)
                events = conv_data.get("convo_events", [])
                if args.user_only:
                    user_events = filter_user_messages(events)
                    console.print(
                        f"  {i+1:3d}. {conv_id} | {stats['formatted_start']} | {stats['duration']} | ("
                        f"{len(user_events)} user messages)"
                    )
                else:
                    console.print(
                        f"  {i+1:3d}. {conv_id} | {stats['formatted_start']} | {stats['duration']} | ("
                        f"{len(events)} events)"
                    )
            console.print()
        except Exception as e:
            console = Console()
            console.print(f"[red]Error: {escape(str(e))}[/red]")
    else:
        display_trajectory(args.user_id, args.conversation, args.user_only)


if __name__ == "__main__":
    main()
