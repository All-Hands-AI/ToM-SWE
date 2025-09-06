#!/usr/bin/env python3
"""
Session Washer for Stateful SWE Benchmark

Simplified version that modifies user raw sessions to inject user preference signals
based on realistic power user profiles. Uses LLM-based message cleaning and natural
preference signal injection.
"""

import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def count_tokens(session_collection: List[Dict[str, Any]]) -> int:
    """Count total tokens in a collection of sessions using tiktoken.

    Args:
        session_collection: List of session dictionaries

    Returns:
        Total token count across all sessions
    """
    total_tokens = 0

    # Initialize tiktoken encoder
    if TIKTOKEN_AVAILABLE:
        try:
            # Use cl100k_base encoding (used by GPT-4 and GPT-3.5-turbo)
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to simple counting if tiktoken fails
            enc = None
    else:
        enc = None

    for session in session_collection:
        conversation_messages = session.get("conversation_messages", [])
        for message in conversation_messages:
            content = message.get("content", "")
            if content:
                if enc is not None:
                    # Use tiktoken for accurate token counting
                    tokens = len(enc.encode(content))
                else:
                    # Fallback to simple whitespace-based counting
                    tokens = len(content.split())
                total_tokens += tokens

    return total_tokens


@dataclass
class WasherConfig:
    """Configuration for SessionWasher behavior."""

    modification_probability: float = 0.4
    random_seed: int = 42


class SessionWasher:
    """Simplified session washer that injects user preference signals."""

    def __init__(self, config: Optional[WasherConfig] = None):
        self.config = config or WasherConfig()
        random.seed(self.config.random_seed)

    def load_profiles(self, profiles_path: str) -> List[Dict[str, Any]]:
        """Load user profiles from JSONL file."""
        profiles = []

        with open(profiles_path, "r") as f:
            for line in f:
                if line.strip():
                    profile = json.loads(line.strip())
                    profiles.append(profile)

        print(f"👤 Loaded {len(profiles)} user profiles from {profiles_path}")
        return profiles

    def load_raw_sessions(
        self, sessions_dir: str = "data/stateful_swe/user_raw_sessions"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load raw session files from directory, organized by file."""
        sessions_by_file = {}
        sessions_path = Path(sessions_dir)
        total_conversations = 0

        for session_file in sessions_path.glob("*.json"):
            try:
                file_id = session_file.stem  # Get filename without extension
                file_sessions = []

                with open(session_file, "r") as f:
                    session_data = json.load(f)
                    # Extract individual conversations from the session data
                    for convo_id, convo_data in session_data.items():
                        convo_events = convo_data.get("convo_events", [])

                        # Check if conversation has user messages (filter out system-only conversations)
                        user_messages = [
                            msg for msg in convo_events if msg.get("source") == "user"
                        ]

                        if user_messages:  # Only add conversations with user messages
                            file_sessions.append(
                                {
                                    "original_session_id": convo_id,
                                    "convo_start": convo_data.get("convo_start"),
                                    "convo_end": convo_data.get("convo_end"),
                                    "convo_events": convo_events,  # Keep full conversation
                                }
                            )

                sessions_by_file[file_id] = file_sessions
                total_conversations += len(file_sessions)
                print(f"  📄 {file_id}: {len(file_sessions)} conversations")

            except Exception as e:
                print(f"⚠️  Warning: Could not load {session_file}: {e}")

        print(
            f"📁 Loaded {total_conversations} conversations from {len(sessions_by_file)} files in {sessions_dir}"
        )
        return sessions_by_file

    def process_user_message(self, message_content: str) -> str:
        """Clean user message content. Placeholder for LLM-based cleaning."""
        # TODO: Replace with LLM-based cleaning
        # For now, simple rule-based cleaning
        return ""

    def wash_session_for_profile(
        self, session: Dict[str, Any], profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Wash a single session for a specific profile by processing the entire conversation flow."""

        # Process the entire conversation event sequence
        washed_conversation_messages = []

        for event in session["convo_events"]:
            if event.get("source") == "user":
                # This is a user message - wash it
                original_content = event.get("content", "")
                processed_content = self.process_user_message(original_content)

                # Create washed user message in the expected format
                washed_event = {"role": "user", "content": processed_content}
                washed_conversation_messages.append(washed_event)

            elif event.get("source") == "assistant":
                # Keep assistant messages unchanged
                washed_event = {
                    "role": "assistant",
                    "content": event.get("content", ""),
                }
                washed_conversation_messages.append(washed_event)

            elif event.get("source") == "system":
                # Keep system messages unchanged
                washed_event = {"role": "system", "content": event.get("content", "")}
                washed_conversation_messages.append(washed_event)

            elif event.get("source") == "tool":
                # Keep tool messages unchanged
                washed_event = {"role": "tool", "content": event.get("content", "")}
                washed_conversation_messages.append(washed_event)

        # Create the washed session in the expected format
        washed_session = {
            "session_id": f"{profile['profile_id']}_{session['original_session_id']}",
            "start_time": session.get("convo_start"),
            "end_time": session.get("convo_end"),
            "event_count": len(session["convo_events"]),
            "message_count": len(washed_conversation_messages),
            "conversation_messages": washed_conversation_messages,
        }

        return washed_session

    def generate_washed_dataset(
        self,
        sessions_dir: str = "data/stateful_swe/user_raw_sessions",
        profiles_path: str = "data/stateful_swe/user_profiles.jsonl",
        output_dir: str = "data/stateful_swe/washed_sessions",
        sessions_per_profile: int = 20,
    ) -> None:
        """Generate the complete washed dataset."""

        # Load data
        print("🔄 Loading data...")
        sessions_by_file = self.load_raw_sessions(sessions_dir)
        profiles = self.load_profiles(profiles_path)

        if not sessions_by_file:
            raise ValueError(f"No sessions found in {sessions_dir}")
        if not profiles:
            raise ValueError(f"No profiles found in {profiles_path}")

        total_conversations = sum(
            len(sessions) for sessions in sessions_by_file.values()
        )
        print("\n🧼 Starting dataset generation:")
        print(f"  - {len(profiles)} profiles")
        print(
            f"  - {total_conversations} raw conversations from {len(sessions_by_file)} files"
        )
        print(f"  - {sessions_per_profile} sessions per profile")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process each profile
        total_generated = 0
        for i, profile in enumerate(profiles):
            profile_id = profile["profile_id"]
            print(f"\n[{i+1}/{len(profiles)}] Processing: {profile_id}")

            # Select one file ID for this profile, then sample sessions from that file
            file_id = random.choice(list(sessions_by_file.keys()))
            file_sessions = sessions_by_file[file_id]
            print(f"  Using file: {file_id} ({len(file_sessions)} available sessions)")

            # Sample sessions from the chosen file
            if len(file_sessions) >= sessions_per_profile:
                selected_sessions = random.sample(file_sessions, sessions_per_profile)
            else:
                # Sample with replacement if not enough sessions in this file
                selected_sessions = [
                    random.choice(file_sessions) for _ in range(sessions_per_profile)
                ]

            # Wash sessions for this profile
            profile_output = []
            for j, session in enumerate(selected_sessions):
                washed_session = self.wash_session_for_profile(session, profile)
                washed_session["washed_session_id"] = f"{profile_id}_{j:02d}"
                profile_output.append(washed_session)

            # Save profile sessions as JSONL
            profile_file = output_path / f"{profile_id}.jsonl"
            profile_final = {
                "metadata": {
                    "session_token_count": count_tokens(profile_output),
                    "original_user_id": file_id,
                },
                "user_profile": profile,
                "user_sessions": profile_output,
            }
            with open(profile_file, "w") as f:
                f.write(json.dumps(profile_final, indent=2) + "\n")

            total_generated += len(profile_output)
            print(f"  ✅ Saved {len(profile_output)} sessions")

        print("\n🎉 Dataset generation complete!")
        print(f"  - Generated {total_generated} total sessions")
        print(f"  - Saved to {output_dir}/")


def main():
    """Generate the stateful SWE benchmark dataset."""
    config = WasherConfig(modification_probability=0.4, random_seed=42)

    washer = SessionWasher(config)
    washer.generate_washed_dataset()


if __name__ == "__main__":
    main()
