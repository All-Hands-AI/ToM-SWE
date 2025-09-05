#!/usr/bin/env python3
"""
Concrete Profile Generator for Stateful SWE Benchmark

Creates realistic user profiles with specific interaction preferences and concrete coding preferences.
Each profile includes roleplay prompts and evaluation criteria for objective assessment.
"""

import json
from typing import List, Dict, Any, TypedDict
from pathlib import Path


class ProfileConfig(TypedDict):
    """Type definition for profile configuration dictionary."""

    profile_id: str
    verbosity: str
    question_timing: str
    coding_preferences: List[str]


class ConcreteUserProfile:
    """Represents a realistic user with specific preferences and coding standards."""

    def __init__(
        self,
        profile_id: str,
        verbosity: str,
        question_timing: str,
        coding_preferences: List[str],
    ):
        self.profile_id = profile_id
        self.verbosity = verbosity
        self.question_timing = question_timing
        self.coding_preferences = coding_preferences

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "profile_id": self.profile_id,
            "interaction_preferences": {
                "verbosity": self.verbosity,
                "question_timing": self.question_timing,
            },
            "coding_preferences": self.coding_preferences,
            "user_roleplay_prompt": self.get_roleplay_prompt(),
        }

    def get_roleplay_prompt(self) -> str:
        """Generate roleplay prompt for LLM to simulate this user type."""
        verbosity_desc = {
            "concise": "You prefer brief, to-the-point responses. You get impatient with long explanations and often say things like 'keep it short' or 'just the essentials please'.",
            "verbose": "You appreciate detailed explanations and comprehensive responses. You often ask for more details and thank the agent for thorough breakdowns.",
        }

        timing_desc = {
            "upfront": "You prefer to ask all your clarifying questions at the beginning before any work starts. You like to understand the full scope upfront. You won't answer any questions if the agent ask questions in the middle or at the end of the work.",
            "ongoing": "You're comfortable with questions being asked throughout the process as they arise. You prefer iterative clarification.",
        }

        prefs_desc = "You have specific coding preferences: " + ", ".join(
            self.coding_preferences[:3]
        )
        if len(self.coding_preferences) > 3:
            prefs_desc += (
                f", and {len(self.coding_preferences) - 3} other specific standards."
            )

        return f"""You are roleplaying as a software developer with these characteristics:

INTERACTION STYLE:
- {verbosity_desc[self.verbosity]}
- {timing_desc[self.question_timing]}

CODING STANDARDS:
- {prefs_desc}

Respond naturally as this type of user would, incorporating these preferences into your messages. Be authentic to this persona while working with the SWE agent."""


class ConcreteProfileGenerator:
    """Generates concrete, realistic user profiles for the benchmark."""

    def __init__(self):
        self.verbosity_options = ["concise", "verbose"]
        self.question_timing_options = ["upfront", "ongoing"]

        # Pool of concrete coding preferences
        self.coding_preference_pool = [
            "Use snake_case for all function and variable names",
            "Include type hints for all function parameters and return values",
            "Write comprehensive docstrings for all public functions",
            "Add explicit error handling with try/except blocks",
            "Use meaningful variable names, avoid single-letter variables",
            "Keep functions under 20 lines when possible",
            "Add comments for complex logic or algorithms",
            "Use f-strings for string formatting instead of .format()",
            "Prefer list comprehensions over explicit loops where readable",
            "Import modules at the top of the file, grouped by standard/third-party/local",
            "Use dataclasses or Pydantic models for data structures",
            "Include unit tests for all new functions",
            "Follow PEP 8 style guidelines strictly",
            "Use context managers (with statements) for file operations",
            "Prefer pathlib over os.path for file path operations",
            "Use logging instead of print statements for debugging",
            "Add assertions for function preconditions",
            "Use enum classes instead of string constants",
            "Separate business logic from presentation logic",
            "Use async/await for I/O operations when possible",
        ]

    def generate_15_profiles(self) -> List[ConcreteUserProfile]:
        """Generate 15 diverse, concrete user profiles."""
        profiles = []

        profile_configs: List[ProfileConfig] = [
            # Concise + Upfront combinations
            {
                "profile_id": "concise_upfront_minimalist_01",
                "verbosity": "concise",
                "question_timing": "upfront",
                "coding_preferences": [
                    "Use snake_case for all function and variable names",
                    "Keep functions under 20 lines when possible",
                    "Use meaningful variable names, avoid single-letter variables",
                ],
            },
            {
                "profile_id": "concise_upfront_typed_02",
                "verbosity": "concise",
                "question_timing": "upfront",
                "coding_preferences": [
                    "Include type hints for all function parameters and return values",
                    "Use f-strings for string formatting instead of .format()",
                    "Import modules at the top of the file, grouped by standard/third-party/local",
                    "Use enum classes instead of string constants",
                ],
            },
            {
                "profile_id": "concise_upfront_functional_03",
                "verbosity": "concise",
                "question_timing": "upfront",
                "coding_preferences": [
                    "Prefer list comprehensions over explicit loops where readable",
                    "Use context managers (with statements) for file operations",
                    "Prefer pathlib over os.path for file path operations",
                    "Use dataclasses or Pydantic models for data structures",
                ],
            },
            # Concise + Ongoing combinations
            {
                "profile_id": "concise_ongoing_pragmatic_04",
                "verbosity": "concise",
                "question_timing": "ongoing",
                "coding_preferences": [
                    "Add explicit error handling with try/except blocks",
                    "Use logging instead of print statements for debugging",
                    "Add assertions for function preconditions",
                ],
            },
            {
                "profile_id": "concise_ongoing_modern_05",
                "verbosity": "concise",
                "question_timing": "ongoing",
                "coding_preferences": [
                    "Use async/await for I/O operations when possible",
                    "Follow PEP 8 style guidelines strictly",
                    "Use f-strings for string formatting instead of .format()",
                    "Prefer pathlib over os.path for file path operations",
                ],
            },
            {
                "profile_id": "concise_ongoing_structured_06",
                "verbosity": "concise",
                "question_timing": "ongoing",
                "coding_preferences": [
                    "Separate business logic from presentation logic",
                    "Use dataclasses or Pydantic models for data structures",
                    "Import modules at the top of the file, grouped by standard/third-party/local",
                    "Use meaningful variable names, avoid single-letter variables",
                    "Keep functions under 20 lines when possible",
                ],
            },
            # Verbose + Upfront combinations
            {
                "profile_id": "verbose_upfront_documented_07",
                "verbosity": "verbose",
                "question_timing": "upfront",
                "coding_preferences": [
                    "Write comprehensive docstrings for all public functions",
                    "Add comments for complex logic or algorithms",
                    "Include type hints for all function parameters and return values",
                    "Use meaningful variable names, avoid single-letter variables",
                ],
            },
            {
                "profile_id": "verbose_upfront_robust_08",
                "verbosity": "verbose",
                "question_timing": "upfront",
                "coding_preferences": [
                    "Add explicit error handling with try/except blocks",
                    "Add assertions for function preconditions",
                    "Include unit tests for all new functions",
                    "Use logging instead of print statements for debugging",
                    "Write comprehensive docstrings for all public functions",
                ],
            },
            {
                "profile_id": "verbose_upfront_enterprise_09",
                "verbosity": "verbose",
                "question_timing": "upfront",
                "coding_preferences": [
                    "Follow PEP 8 style guidelines strictly",
                    "Separate business logic from presentation logic",
                    "Use dataclasses or Pydantic models for data structures",
                    "Import modules at the top of the file, grouped by standard/third-party/local",
                ],
            },
            # Verbose + Ongoing combinations
            {
                "profile_id": "verbose_ongoing_teacher_10",
                "verbosity": "verbose",
                "question_timing": "ongoing",
                "coding_preferences": [
                    "Write comprehensive docstrings for all public functions",
                    "Add comments for complex logic or algorithms",
                    "Include unit tests for all new functions",
                    "Use meaningful variable names, avoid single-letter variables",
                ],
            },
            {
                "profile_id": "verbose_ongoing_perfectionist_11",
                "verbosity": "verbose",
                "question_timing": "ongoing",
                "coding_preferences": [
                    "Follow PEP 8 style guidelines strictly",
                    "Include type hints for all function parameters and return values",
                    "Add explicit error handling with try/except blocks",
                    "Keep functions under 20 lines when possible",
                    "Use f-strings for string formatting instead of .format()",
                ],
            },
            {
                "profile_id": "verbose_ongoing_architect_12",
                "verbosity": "verbose",
                "question_timing": "ongoing",
                "coding_preferences": [
                    "Separate business logic from presentation logic",
                    "Use dataclasses or Pydantic models for data structures",
                    "Use async/await for I/O operations when possible",
                    "Use enum classes instead of string constants",
                ],
            },
            # Mixed edge cases
            {
                "profile_id": "concise_upfront_tester_13",
                "verbosity": "concise",
                "question_timing": "upfront",
                "coding_preferences": [
                    "Include unit tests for all new functions",
                    "Add assertions for function preconditions",
                    "Use logging instead of print statements for debugging",
                    "Add explicit error handling with try/except blocks",
                ],
            },
            {
                "profile_id": "verbose_ongoing_pythonic_14",
                "verbosity": "verbose",
                "question_timing": "ongoing",
                "coding_preferences": [
                    "Prefer list comprehensions over explicit loops where readable",
                    "Use context managers (with statements) for file operations",
                    "Prefer pathlib over os.path for file path operations",
                    "Use f-strings for string formatting instead of .format()",
                    "Use enum classes instead of string constants",
                ],
            },
            {
                "profile_id": "concise_ongoing_performance_15",
                "verbosity": "concise",
                "question_timing": "ongoing",
                "coding_preferences": [
                    "Use async/await for I/O operations when possible",
                    "Prefer list comprehensions over explicit loops where readable",
                    "Use logging instead of print statements for debugging",
                ],
            },
        ]

        for config in profile_configs:
            profile = ConcreteUserProfile(
                profile_id=config["profile_id"],
                verbosity=config["verbosity"],
                question_timing=config["question_timing"],
                coding_preferences=config["coding_preferences"],
            )
            profiles.append(profile)

        return profiles

    def save_profiles(
        self, profiles: List[ConcreteUserProfile], output_path: str
    ) -> None:
        """Save profiles to JSON file."""
        profiles_data = [profile.to_dict() for profile in profiles]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(profiles_data, f, indent=2)

        print(f"💾 Saved {len(profiles)} concrete profiles to {output_path}")


def main():
    """Generate and save concrete user profiles."""
    generator = ConcreteProfileGenerator()

    # Generate 15 concrete profiles
    profiles = generator.generate_15_profiles()

    # Save to output directory
    output_path = "data/stateful_swe/user_profiles.json"
    generator.save_profiles(profiles, output_path)

    # Print summary
    print("\n📊 Generated concrete profiles summary:")
    for profile in profiles:
        print(
            f"  - {profile.profile_id}: {profile.verbosity}/{profile.question_timing}, "
            f"{len(profile.coding_preferences)} coding preferences"
        )


if __name__ == "__main__":
    main()
