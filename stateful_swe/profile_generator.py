#!/usr/bin/env python3
"""
Profile Generator for Stateful SWE Benchmark

Creates synthetic user profiles with specific preferences across three dimensions:
- Communication Style: concise, detailed, balanced
- Risk Tolerance: cautious, bold, moderate
- Question Preference: minimal, moderate, high
"""

import json
import itertools
from typing import List, Dict, Any
from pathlib import Path


class UserProfile:
    """Represents a user's behavioral preferences."""

    def __init__(
        self,
        profile_id: str,
        communication_style: str,
        risk_tolerance: str,
        question_preference: str,
    ):
        self.profile_id = profile_id
        self.communication_style = communication_style
        self.risk_tolerance = risk_tolerance
        self.question_preference = question_preference

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "profile_id": self.profile_id,
            "communication_style": self.communication_style,
            "risk_tolerance": self.risk_tolerance,
            "question_preference": self.question_preference,
            "description": self.get_description(),
            "behavioral_signals": self.get_behavioral_signals(),
        }

    def get_description(self) -> str:
        """Get human-readable description of the profile."""
        return (
            f"User prefers {self.communication_style} communication, "
            f"has {self.risk_tolerance} risk tolerance, and wants "
            f"{self.question_preference} questions from the agent."
        )

    def get_behavioral_signals(self) -> Dict[str, List[str]]:
        """Get specific signals this user type would exhibit."""
        signals: Dict[str, List[str]] = {
            "positive_responses": [],
            "negative_responses": [],
            "explicit_requests": [],
        }

        # Communication style signals
        if self.communication_style == "concise":
            signals["negative_responses"].extend(
                [
                    "that's too long",
                    "please be more brief",
                    "can you summarize?",
                    "too verbose",
                ]
            )
            signals["explicit_requests"].extend(
                ["keep it short", "just the essentials please", "be more concise"]
            )
        elif self.communication_style == "detailed":
            signals["positive_responses"].extend(
                [
                    "thanks for the detailed explanation",
                    "that's very helpful",
                    "good breakdown",
                ]
            )
            signals["explicit_requests"].extend(
                [
                    "can you explain that in more detail?",
                    "tell me more about that",
                    "walk me through this step by step",
                ]
            )

        # Risk tolerance signals
        if self.risk_tolerance == "cautious":
            signals["explicit_requests"].extend(
                [
                    "are you sure about that?",
                    "can you double-check?",
                    "let me review this first",
                    "please confirm before making changes",
                ]
            )
        elif self.risk_tolerance == "bold":
            signals["positive_responses"].extend(
                ["just do it", "go ahead", "looks good, proceed"]
            )
            signals["negative_responses"].extend(
                ["stop asking for confirmation", "I trust you, just proceed"]
            )

        # Question preference signals
        if self.question_preference == "minimal":
            signals["negative_responses"].extend(
                [
                    "stop asking so many questions",
                    "just proceed with what makes sense",
                    "figure it out yourself",
                ]
            )
        elif self.question_preference == "high":
            signals["positive_responses"].extend(
                ["good question", "thanks for checking", "I appreciate you asking"]
            )

        return signals


class ProfileGenerator:
    """Generates synthetic user profiles for the benchmark."""

    def __init__(self):
        self.communication_styles = ["concise", "detailed", "balanced"]
        self.risk_tolerances = ["cautious", "bold", "moderate"]
        self.question_preferences = ["minimal", "moderate", "high"]

    def generate_all_combinations(self) -> List[UserProfile]:
        """Generate profiles for all possible combinations of preferences."""
        profiles = []

        for i, (comm, risk, questions) in enumerate(
            itertools.product(
                self.communication_styles,
                self.risk_tolerances,
                self.question_preferences,
            )
        ):
            profile_id = f"{comm}_{risk}_{questions}_{i:02d}"
            profile = UserProfile(profile_id, comm, risk, questions)
            profiles.append(profile)

        return profiles

    def generate_selected_profiles(self) -> List[UserProfile]:
        """Generate a curated set of distinct, interesting profiles."""
        selected_combinations = [
            # Extreme combinations
            ("concise", "bold", "minimal"),  # Impatient power user
            ("detailed", "cautious", "high"),  # Careful learner
            ("concise", "cautious", "moderate"),  # Efficient but careful
            ("detailed", "bold", "minimal"),  # Wants details but hates questions
            # Balanced combinations
            ("balanced", "moderate", "moderate"),  # Typical user
            ("balanced", "bold", "minimal"),  # Confident user
            ("balanced", "cautious", "high"),  # Uncertain user
            # Interesting edge cases
            ("concise", "bold", "high"),  # Fast but double-checks
            ("detailed", "cautious", "minimal"),  # Wants explanations, no questions
            ("concise", "moderate", "high"),  # Brief but likes clarification
            # Additional coverage
            ("detailed", "moderate", "moderate"),  # Standard detailed user
            ("concise", "moderate", "minimal"),  # Quick and efficient
        ]

        profiles = []
        for i, (comm, risk, questions) in enumerate(selected_combinations):
            profile_id = f"{comm}_{risk}_{questions}_{i:02d}"
            profile = UserProfile(profile_id, comm, risk, questions)
            profiles.append(profile)

        return profiles

    def save_profiles(self, profiles: List[UserProfile], output_path: str) -> None:
        """Save profiles to JSON file."""
        profiles_data = [profile.to_dict() for profile in profiles]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(profiles_data, f, indent=2)

        print(f"💾 Saved {len(profiles)} profiles to {output_path}")


def main():
    """Generate and save user profiles."""
    generator = ProfileGenerator()

    # Generate selected profiles (12 distinct combinations)
    profiles = generator.generate_selected_profiles()

    # Save to output directory
    output_path = "data/stateful_swe/user_profiles.json"
    generator.save_profiles(profiles, output_path)

    # Print summary
    print("\n📊 Generated profiles summary:")
    for profile in profiles:
        print(f"  - {profile.profile_id}: {profile.get_description()}")


if __name__ == "__main__":
    main()
