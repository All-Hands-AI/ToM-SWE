#!/usr/bin/env python3
"""
Dataset Builder for Stateful SWE Benchmark

Combines user profiles with washed conversation sessions to create the final
benchmark dataset format for testing ToM agents' preference learning abilities.
"""

import json
from typing import List, Dict, Any
from pathlib import Path
import random


class DatasetBuilder:
    """Builds the final benchmark dataset by combining profiles with washed sessions."""

    def __init__(self):
        self.random_seed = 42
        random.seed(self.random_seed)

    def load_profiles(self, profiles_path: str) -> List[Dict[str, Any]]:
        """Load user profiles from JSON file."""
        with open(profiles_path, "r") as f:
            profiles: List[Dict[str, Any]] = json.load(f)

        print(f"📋 Loaded {len(profiles)} user profiles")
        return profiles

    def load_washed_sessions_for_profile(
        self, profile_id: str, washed_sessions_dir: str
    ) -> List[Dict[str, Any]]:
        """Load all washed sessions for a specific profile."""
        profile_dir = Path(washed_sessions_dir) / profile_id

        if not profile_dir.exists():
            print(f"⚠️  Warning: No washed sessions found for profile {profile_id}")
            return []

        sessions = []
        for session_file in profile_dir.glob("*.json"):
            try:
                with open(session_file, "r") as f:
                    session_data = json.load(f)
                    sessions.append(session_data)
            except Exception as e:
                print(f"⚠️  Warning: Could not load {session_file}: {e}")

        return sessions

    def create_dataset_instance(
        self,
        instance_id: str,
        profile: Dict[str, Any],
        washed_sessions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create a single dataset instance combining profile with sessions."""

        # Extract core profile information (remove internal details)
        ground_truth_profile = {
            "profile_id": profile["profile_id"],
            "communication_style": profile["communication_style"],
            "risk_tolerance": profile["risk_tolerance"],
            "question_preference": profile["question_preference"],
            "description": profile["description"],
        }

        # Create memory structure with washed sessions
        memory: Dict[str, Any] = {
            "user_id": f"synthetic_{profile['profile_id']}",
            "total_sessions": len(washed_sessions),
            "sessions": [],
        }

        # Add each session to memory (keep essential information)
        for session in washed_sessions:
            session_summary = {
                "session_id": session.get(
                    "washed_session_id", session.get("session_id", "unknown")
                ),
                "start_time": session.get("start_time", ""),
                "end_time": session.get("end_time", ""),
                "message_count": len(session.get("conversation_messages", [])),
                "conversation_messages": session.get("conversation_messages", []),
                "washing_metadata": session.get("washing_metadata", {}),
            }
            memory["sessions"].append(session_summary)

        # Create final instance
        instance = {
            "instance_id": instance_id,
            "person_ground_truth_profile": ground_truth_profile,
            "memory": memory,
            "metadata": {
                "created_from_profile": profile["profile_id"],
                "num_washed_sessions": len(washed_sessions),
                "dataset_version": "0.1.0",
                "purpose": "Testing ToM agent preference learning and adaptation",
            },
        }

        return instance

    def build_dataset(
        self,
        profiles_path: str,
        washed_sessions_dir: str,
        instances_per_profile: int = 1,
    ) -> List[Dict[str, Any]]:
        """Build complete dataset from profiles and washed sessions."""

        # Load profiles
        profiles = self.load_profiles(profiles_path)

        dataset = []
        instance_counter = 1

        for profile in profiles:
            profile_id = profile["profile_id"]
            print(f"\n🔨 Building instances for profile: {profile_id}")

            # Load washed sessions for this profile
            washed_sessions = self.load_washed_sessions_for_profile(
                profile_id, washed_sessions_dir
            )

            if not washed_sessions:
                print(f"⏭️  Skipping profile {profile_id} - no washed sessions found")
                continue

            # Create multiple instances per profile if requested
            for i in range(instances_per_profile):
                instance_id = f"{instance_counter:03d}"

                # For multiple instances, potentially sample different subsets
                if instances_per_profile > 1 and len(washed_sessions) > 10:
                    # Sample a subset of sessions for variety
                    sampled_sessions = random.sample(
                        washed_sessions, min(20, len(washed_sessions))
                    )
                else:
                    sampled_sessions = washed_sessions

                # Create instance
                instance = self.create_dataset_instance(
                    instance_id, profile, sampled_sessions
                )

                dataset.append(instance)
                instance_counter += 1

                print(
                    f"  ✅ Created instance {instance_id} with {len(sampled_sessions)} sessions"
                )

        return dataset

    def save_dataset(self, dataset: List[Dict[str, Any]], output_path: str) -> None:
        """Save complete dataset to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create dataset metadata
        dataset_metadata = {
            "total_instances": len(dataset),
            "profiles_covered": len(
                set(
                    inst["person_ground_truth_profile"]["profile_id"]
                    for inst in dataset
                )
            ),
            "preference_dimensions": [
                "communication_style",
                "risk_tolerance",
                "question_preference",
            ],
            "format_version": "1.0",
            "description": "Stateful SWE benchmark dataset for testing ToM agent preference learning",
        }

        # Complete dataset structure
        complete_dataset = {"metadata": dataset_metadata, "instances": dataset}

        with open(output_file, "w") as f:
            json.dump(complete_dataset, f, indent=2)

        print(f"\n💾 Saved complete dataset to {output_path}")
        print("📊 Dataset summary:")
        print(f"  - Total instances: {dataset_metadata['total_instances']}")
        print(f"  - Profiles covered: {dataset_metadata['profiles_covered']}")
        print(
            f"  - Total sessions: {sum(len(inst['memory']['sessions']) for inst in dataset)}"
        )

    def generate_dataset_statistics(
        self, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate statistics about the created dataset."""

        stats: Dict[str, Any] = {
            "total_instances": len(dataset),
            "preference_distributions": {
                "communication_style": {},
                "risk_tolerance": {},
                "question_preference": {},
            },
            "session_counts": [],
            "message_counts": [],
        }

        for instance in dataset:
            profile = instance["person_ground_truth_profile"]

            # Count preference distributions
            for pref_type in [
                "communication_style",
                "risk_tolerance",
                "question_preference",
            ]:
                pref_value = profile[pref_type]
                if pref_value not in stats["preference_distributions"][pref_type]:
                    stats["preference_distributions"][pref_type][pref_value] = 0
                stats["preference_distributions"][pref_type][pref_value] += 1

            # Collect session and message counts
            sessions = instance["memory"]["sessions"]
            stats["session_counts"].append(len(sessions))

            total_messages = sum(session["message_count"] for session in sessions)
            stats["message_counts"].append(total_messages)

        # Calculate averages
        stats["avg_sessions_per_instance"] = sum(stats["session_counts"]) / len(
            stats["session_counts"]
        )
        stats["avg_messages_per_instance"] = sum(stats["message_counts"]) / len(
            stats["message_counts"]
        )

        return stats


def main():
    """Build the complete benchmark dataset."""
    builder = DatasetBuilder()

    # Paths
    profiles_path = "data/stateful_swe/user_profiles.json"
    washed_sessions_dir = "data/stateful_swe/washed_sessions"
    output_path = "data/stateful_swe/final_dataset.json"

    # Build dataset
    print("🏗️  Building Stateful SWE Benchmark Dataset")
    print("=" * 50)

    dataset = builder.build_dataset(
        profiles_path=profiles_path,
        washed_sessions_dir=washed_sessions_dir,
        instances_per_profile=1,  # One instance per profile for now
    )

    # Generate and print statistics
    stats = builder.generate_dataset_statistics(dataset)
    print("\n📈 Dataset Statistics:")
    print(
        f"  - Communication styles: {stats['preference_distributions']['communication_style']}"
    )
    print(f"  - Risk tolerance: {stats['preference_distributions']['risk_tolerance']}")
    print(
        f"  - Question preferences: {stats['preference_distributions']['question_preference']}"
    )
    print(f"  - Avg sessions per instance: {stats['avg_sessions_per_instance']:.1f}")
    print(f"  - Avg messages per instance: {stats['avg_messages_per_instance']:.1f}")

    # Save dataset
    builder.save_dataset(dataset, output_path)

    print("\n🎉 Dataset building complete!")
    print(f"   Ready for ToM agent testing with {len(dataset)} instances")


if __name__ == "__main__":
    main()
