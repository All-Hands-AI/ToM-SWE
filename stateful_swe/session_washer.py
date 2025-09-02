#!/usr/bin/env python3
"""
Session Washer for Stateful SWE Benchmark

Modifies existing conversation sessions to inject specific user preference signals,
creating training data where user behavior consistently reflects target preferences.
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path
import copy


class SessionWasher:
    """Modifies conversation sessions to inject user preference signals."""

    def __init__(self):
        self.random_seed = 42
        random.seed(self.random_seed)

    def load_sessions(self, sessions_dir: str) -> List[Dict[str, Any]]:
        """Load all session files from directory."""
        sessions = []
        sessions_path = Path(sessions_dir)

        for session_file in sessions_path.glob("*.json"):
            try:
                with open(session_file, "r") as f:
                    session_data = json.load(f)
                    sessions.append(session_data)
            except Exception as e:
                print(f"⚠️  Warning: Could not load {session_file}: {e}")

        print(f"📁 Loaded {len(sessions)} sessions from {sessions_dir}")
        return sessions

    def wash_session_for_profile(
        self, session: Dict[str, Any], profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Modify a session to match a specific user profile."""
        washed_session = copy.deepcopy(session)

        # Get behavioral signals for this profile
        signals = profile.get("behavioral_signals", {})

        # Modify user messages to inject preference signals
        conversation_messages = washed_session.get("conversation_messages", [])
        modified_messages = []

        for i, message in enumerate(conversation_messages):
            if message.get("role") == "user":
                # This is a user message - potentially modify it
                modified_message = self._modify_user_message(
                    message, profile, signals, i, len(conversation_messages)
                )
                modified_messages.append(modified_message)
            else:
                # Keep agent messages unchanged
                modified_messages.append(message)

        washed_session["conversation_messages"] = modified_messages

        # Add metadata about the washing process
        washed_session["washing_metadata"] = {
            "profile_id": profile["profile_id"],
            "original_session_id": session.get("session_id", "unknown"),
            "modifications_applied": True,
            "profile_preferences": {
                "communication_style": profile["communication_style"],
                "risk_tolerance": profile["risk_tolerance"],
                "question_preference": profile["question_preference"],
            },
        }

        return washed_session

    def _modify_user_message(
        self,
        message: Dict[str, Any],
        profile: Dict[str, Any],
        signals: Dict[str, List[str]],
        message_index: int,
        total_messages: int,
    ) -> Dict[str, Any]:
        """Modify a user message to inject preference signals."""
        modified_message = copy.deepcopy(message)
        content = message.get("content", "")

        # Skip very short messages or system-heavy content
        if len(content.strip()) < 10:
            return modified_message

        # Clean content to check actual user text
        from tom_swe.memory.conversation_processor import _clean_user_message

        clean_content = _clean_user_message(content)
        if len(clean_content) < 10:
            return modified_message

        # Apply modifications based on profile with some probability
        modification_probability = 0.3  # 30% chance to modify any given message

        if random.random() < modification_probability:
            modified_content = self._apply_preference_modifications(
                content, profile, signals, message_index, total_messages
            )
            modified_message["content"] = modified_content

        return modified_message

    def _apply_preference_modifications(
        self,
        content: str,
        profile: Dict[str, Any],
        signals: Dict[str, List[str]],
        message_index: int,
        total_messages: int,
    ) -> str:
        """Apply specific preference modifications to message content."""

        # Start with original content
        modified_content = content

        # Communication Style modifications
        comm_style = profile["communication_style"]
        if comm_style == "concise":
            modified_content = self._add_conciseness_signals(modified_content, signals)
        elif comm_style == "detailed":
            modified_content = self._add_detail_requests(modified_content, signals)

        # Risk Tolerance modifications
        risk_tolerance = profile["risk_tolerance"]
        if risk_tolerance == "cautious":
            modified_content = self._add_caution_signals(modified_content, signals)
        elif risk_tolerance == "bold":
            modified_content = self._add_boldness_signals(modified_content, signals)

        # Question Preference modifications
        question_pref = profile["question_preference"]
        if question_pref == "minimal":
            modified_content = self._add_question_aversion(modified_content, signals)
        elif question_pref == "high":
            modified_content = self._add_question_appreciation(
                modified_content, signals
            )

        return modified_content

    def _add_conciseness_signals(
        self, content: str, signals: Dict[str, List[str]]
    ) -> str:
        """Add signals indicating preference for concise responses."""
        explicit_requests = signals.get("explicit_requests", [])

        # Sometimes add explicit conciseness request
        if random.random() < 0.4 and explicit_requests:
            signal = random.choice(explicit_requests)
            content += f"\n\n({signal})"

        return content

    def _add_detail_requests(self, content: str, signals: Dict[str, List[str]]) -> str:
        """Add signals indicating preference for detailed explanations."""
        explicit_requests = signals.get("explicit_requests", [])

        if random.random() < 0.4 and explicit_requests:
            signal = random.choice(explicit_requests)
            content += f"\n\n{signal}"

        return content

    def _add_caution_signals(self, content: str, signals: Dict[str, List[str]]) -> str:
        """Add signals indicating cautious approach."""
        explicit_requests = signals.get("explicit_requests", [])

        if random.random() < 0.5 and explicit_requests:
            signal = random.choice(explicit_requests)
            content += f"\n\n{signal}"

        return content

    def _add_boldness_signals(self, content: str, signals: Dict[str, List[str]]) -> str:
        """Add signals indicating bold/direct approach."""
        positive_responses = signals.get("positive_responses", [])
        negative_responses = signals.get("negative_responses", [])

        # Sometimes add direct action encouragement
        if random.random() < 0.3:
            if positive_responses and random.random() < 0.7:
                signal = random.choice(positive_responses)
            elif negative_responses:
                signal = random.choice(negative_responses)
            else:
                return content
            content += f"\n\n{signal}"

        return content

    def _add_question_aversion(
        self, content: str, signals: Dict[str, List[str]]
    ) -> str:
        """Add signals indicating dislike of many questions."""
        negative_signals = signals.get("negative_responses", [])

        if random.random() < 0.3 and negative_signals:
            signal = random.choice(negative_signals)
            content += f"\n\n{signal}"

        return content

    def _add_question_appreciation(
        self, content: str, signals: Dict[str, List[str]]
    ) -> str:
        """Add signals indicating appreciation for questions."""
        positive_signals = signals.get("positive_responses", [])

        if random.random() < 0.4 and positive_signals:
            signal = random.choice(positive_signals)
            content += f"\n\n{signal}"

        return content

    def wash_sessions_for_profile(
        self,
        sessions: List[Dict[str, Any]],
        profile: Dict[str, Any],
        num_sessions: int = 20,
    ) -> List[Dict[str, Any]]:
        """Wash multiple sessions for a specific profile."""

        # Select random sessions (with replacement if needed)
        if len(sessions) >= num_sessions:
            selected_sessions = random.sample(sessions, num_sessions)
        else:
            # If we don't have enough sessions, sample with replacement
            selected_sessions = [random.choice(sessions) for _ in range(num_sessions)]

        # Wash each selected session
        washed_sessions = []
        for i, session in enumerate(selected_sessions):
            washed_session = self.wash_session_for_profile(session, profile)
            # Add unique ID to avoid conflicts
            washed_session["washed_session_id"] = f"{profile['profile_id']}_{i:02d}"
            washed_sessions.append(washed_session)

        return washed_sessions

    def save_washed_sessions(
        self, washed_sessions: List[Dict[str, Any]], profile_id: str, output_dir: str
    ) -> None:
        """Save washed sessions to individual files."""
        output_path = Path(output_dir) / profile_id
        output_path.mkdir(parents=True, exist_ok=True)

        for session in washed_sessions:
            filename = f"{session['washed_session_id']}.json"
            filepath = output_path / filename

            with open(filepath, "w") as f:
                json.dump(session, f, indent=2)

        print(f"💾 Saved {len(washed_sessions)} washed sessions for {profile_id}")


def main():
    """Main function to demonstrate session washing."""
    washer = SessionWasher()

    # Load original sessions
    sessions = washer.load_sessions("data/example_sessions")

    # Load profiles
    with open("data/stateful_swe/user_profiles.json", "r") as f:
        profiles = json.load(f)

    # Wash sessions for each profile
    for profile in profiles:  # Process all profiles
        print(f"\n🧼 Washing sessions for profile: {profile['profile_id']}")

        washed_sessions = washer.wash_sessions_for_profile(
            sessions, profile, num_sessions=20  # 20 sessions per profile
        )

        washer.save_washed_sessions(
            washed_sessions, profile["profile_id"], "data/stateful_swe/washed_sessions"
        )


if __name__ == "__main__":
    main()
