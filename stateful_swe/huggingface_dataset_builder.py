#!/usr/bin/env python3
"""
HuggingFace Dataset Builder for cmu-lti/stateful

Creates a HuggingFace dataset following the cmu-lti/interactive-swe structure
but with added user profile assignments for stateful interactions.

This script:
1. Downloads the cmu-lti/interactive-swe dataset
2. Loads user profiles from our generated profiles
3. Assigns profiles to instances randomly
4. Creates a new dataset ready for HuggingFace Hub upload as cmu-lti/stateful
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import logging
from datetime import datetime

from datasets import Dataset, load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASETS_AVAILABLE = True


class StatefulSWEDatasetBuilder:
    """Builds the cmu-lti/stateful dataset based on interactive-swe with user profiles."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)

    def load_user_profiles(self, profiles_path: str) -> List[Dict[str, Any]]:
        """Load user profiles from JSONL file."""
        profiles = []

        try:
            with open(profiles_path, "r") as f:
                for line in f:
                    if line.strip():
                        profile = json.loads(line.strip())
                        profiles.append(profile)

            print(f"👤 Loaded {len(profiles)} user profiles from {profiles_path}")
            return profiles

        except FileNotFoundError:
            logger.error(f"User profiles file not found: {profiles_path}")
            return []

    def create_stateful_dataset_generator(
        self, base_dataset: Any, user_profiles: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Generator that yields enriched instances with user profiles.

        Args:
            base_dataset: The original interactive-swe dataset
            user_profiles: List of user profiles to assign

        Yields:
            Enriched instances with assigned user profiles
        """
        profile_counts = {profile["profile_id"]: 0 for profile in user_profiles}

        for i, instance in enumerate(base_dataset):
            # Randomly assign a user profile
            selected_profile = random.choice(user_profiles)
            profile_counts[selected_profile["profile_id"]] += 1

            # Start with the original instance and add stateful columns
            enriched_instance = dict(instance)

            # Add new stateful columns
            enriched_instance.update(
                {
                    "user_profile_id": selected_profile["profile_id"],
                    "user_roleplay_prompt": selected_profile.get(
                        "user_roleplay_prompt", ""
                    ),
                    "interaction_preferences": json.dumps(
                        selected_profile.get("interaction_preferences", {})
                    ),
                    "coding_preferences": ",".join(
                        selected_profile.get("coding_preferences", [])
                    ),
                    "stateful_instance_id": f"stateful_{i:05d}",
                    "assignment_seed": self.random_seed,
                    "dataset_version": "1.0.0",
                    "created_at_stateful": datetime.now().isoformat(),
                }
            )

            yield enriched_instance

        # Log final distribution
        print("\n📊 Final profile distribution:")
        for profile_id, count in sorted(profile_counts.items()):
            print(f"  - {profile_id}: {count} instances")

    def build_dataset(
        self,
        profiles_path: str = "data/stateful_swe/user_profiles.jsonl",
        base_dataset_name: str = "cmu-lti/interactive-swe",
        split: str = "test",
    ) -> Optional[Dataset]:
        """Build the complete stateful dataset.

        Args:
            profiles_path: Path to user profiles JSONL file
            base_dataset_name: Name of the base HuggingFace dataset
            split: Dataset split to use

        Returns:
            The built Dataset object or None if failed
        """
        if not DATASETS_AVAILABLE:
            logger.error("Cannot build dataset: required libraries not available")
            return None

        print("🚀 Building Stateful SWE Dataset")
        print("=" * 50)

        # Load user profiles
        user_profiles = self.load_user_profiles(profiles_path)
        if not user_profiles:
            logger.error("Failed to load user profiles")
            return None

        # Load base dataset
        print(f"📥 Loading base dataset: {base_dataset_name} ({split})")
        try:
            base_dataset = load_dataset(base_dataset_name, split=split)
            print(f"✅ Loaded {len(base_dataset)} instances from {base_dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load base dataset: {e}")
            return None

        # Create the stateful dataset using generator for memory efficiency
        print("🔄 Creating stateful dataset with profile assignments...")

        stateful_dataset = Dataset.from_generator(
            lambda: self.create_stateful_dataset_generator(base_dataset, user_profiles)
        )

        print(f"✅ Built stateful dataset with {len(stateful_dataset)} instances")

        return stateful_dataset

    def create_dataset_card(self, num_instances: int, num_profiles: int) -> str:
        """Create a dataset card for the stateful dataset."""

        card_content = f"""---
language:
- en
tags:
- software-engineering
- code
- swe-bench
- stateful
- user-modeling
- theory-of-mind
size_categories:
- n<1K
task_categories:
- text-generation
- question-answering
dataset_info:
  features:
  - name: repo
    dtype: string
  - name: instance_id
    dtype: string
  - name: base_commit
    dtype: string
  - name: patch
    dtype: string
  - name: test_patch
    dtype: string
  - name: problem_statement
    dtype: string
  - name: hints_text
    dtype: string
  - name: created_at
    dtype: string
  - name: difficulty
    dtype: string
  - name: original_issue
    dtype: string
  - name: files
    dtype: string
  - name: user_profile_id
    dtype: string
  - name: user_roleplay_prompt
    dtype: string
  - name: interaction_preferences
    dtype: string
  - name: coding_preferences
    dtype: string
  - name: stateful_instance_id
    dtype: string
  - name: assignment_seed
    dtype: int32
  - name: dataset_version
    dtype: string
  - name: created_at_stateful
    dtype: string
  configs:
  - config_name: default
    data_files:
    - split: test
      path: data/test-*
  download_size: 0
  dataset_size: 0
---

# Stateful SWE Dataset

## Dataset Summary

The **Stateful SWE Dataset** extends the [cmu-lti/interactive-swe](https://huggingface.co/datasets/cmu-lti/interactive-swe) dataset with user profile assignments for studying stateful interactions in software engineering tasks. Each instance from the original interactive-swe dataset is enriched with a randomly assigned user profile that defines interaction preferences and coding standards.

This dataset enables research into:
- **Theory of Mind (ToM)** modeling for AI agents
- **Stateful user interactions** in software engineering
- **Personalized code assistance** based on user preferences
- **User behavior modeling** in programming contexts

## Dataset Details

- **Total instances**: {num_instances}
- **User profiles**: {num_profiles} distinct profiles
- **Base dataset**: cmu-lti/interactive-swe
- **Assignment**: Random profile assignment with seed {self.random_seed}
- **Version**: 1.0.0

## Dataset Structure

### Original Interactive-SWE Columns
- `repo`: Repository name
- `instance_id`: Unique identifier from original dataset
- `base_commit`: Base commit hash
- `patch`: Code changes
- `test_patch`: Test-related changes
- `problem_statement`: Description of the issue
- `hints_text`: Additional hints
- `created_at`: Original timestamp
- `difficulty`: Problem difficulty level
- `original_issue`: Link to original issue
- `files`: List of affected files

### New Stateful Columns
- `user_profile_id`: Assigned user profile identifier
- `user_roleplay_prompt`: Second-person narrative describing the user
- `interaction_preferences`: Dictionary with verbosity, timing, and response style preferences
- `coding_preferences`: List of user's technical preferences
- `stateful_instance_id`: New unique identifier for stateful instances
- `assignment_seed`: Random seed used for profile assignment
- `dataset_version`: Version of the stateful dataset
- `created_at_stateful`: Timestamp when stateful instance was created

## User Profile Types

The dataset includes {num_profiles} diverse user profiles with varying:

- **Verbosity preferences**: concise vs verbose
- **Question timing**: upfront vs ongoing clarification
- **Response style**: short vs long responses
- **Coding preferences**: frameworks, testing, documentation, etc.

## Usage Example

```python
from datasets import load_dataset

# Load the stateful dataset
dataset = load_dataset("cmu-lti/stateful")

# Access an instance with its user profile
instance = dataset["test"][0]
print(f"Problem: {{instance['problem_statement']}}")
print(f"User Profile: {{instance['user_profile_id']}}")
print(f"Interaction Style: {{instance['interaction_preferences']}}")
```

## Citation

If you use this dataset, please cite both the original interactive-swe dataset and this stateful extension:

```bibtex
@dataset{{stateful_swe_2025,
  title={{Stateful SWE Dataset: User Profile Extensions for Interactive Software Engineering}},
  author={{CMU ToM-SWE Team}},
  year={{2025}},
  url={{https://huggingface.co/datasets/cmu-lti/stateful}}
}}
```

## License

This dataset follows the same license as the original cmu-lti/interactive-swe dataset.

## Dataset Creation

Created using the ToM-SWE framework for Theory of Mind modeling in software engineering contexts.
"""

        return card_content

    def save_dataset_locally(
        self, dataset: Dataset, output_dir: str = "data/stateful_swe/dataset_for_upload"
    ) -> None:
        """Save the dataset locally in HuggingFace format.

        Args:
            dataset: The built dataset
            output_dir: Directory to save the dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving dataset locally to: {output_path}")

        # Save as parquet (recommended format for HF)
        dataset.save_to_disk(str(output_path / "dataset"))

        # Save as parquet file for direct upload
        dataset.to_parquet(str(output_path / "test.parquet"))

        # Create and save dataset card
        num_profiles = len(set(dataset["user_profile_id"]))
        card_content = self.create_dataset_card(len(dataset), num_profiles)

        with open(output_path / "README.md", "w") as f:
            f.write(card_content)

        # Save dataset info
        dataset_info = {
            "name": "cmu-lti/stateful",
            "num_instances": len(dataset),
            "num_profiles": num_profiles,
            "columns": list(dataset.column_names),
            "assignment_seed": self.random_seed,
            "created_at": datetime.now().isoformat(),
            "files": ["test.parquet", "README.md"],
        }

        with open(output_path / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        print("✅ Dataset saved locally:")
        print(f"  - Dataset files: {output_path}/dataset/")
        print(f"  - Parquet file: {output_path}/test.parquet")
        print(f"  - Dataset card: {output_path}/README.md")
        print(f"  - Info file: {output_path}/dataset_info.json")

    def build_and_save_complete_dataset(
        self,
        profiles_path: str = "data/stateful_swe/user_profiles.jsonl",
        output_dir: str = "data/stateful_swe/dataset_for_upload",
    ) -> Optional[Dataset]:
        """Complete pipeline: build and save the stateful dataset.

        Args:
            profiles_path: Path to user profiles
            output_dir: Output directory

        Returns:
            The built dataset or None if failed
        """
        # Build the dataset
        dataset = self.build_dataset(profiles_path=profiles_path)

        if dataset is None:
            logger.error("Failed to build dataset")
            return None

        # Save locally
        self.save_dataset_locally(dataset, output_dir)

        print("\n🎉 Stateful SWE Dataset created successfully!")
        print(f"📁 Ready for upload from: {output_dir}")
        print("\nNext steps:")
        print("1. Review the generated README.md")
        print("2. Upload to HuggingFace Hub using:")
        print(
            f"   huggingface-cli upload cmu-lti/stateful {output_dir}/test.parquet test.parquet"
        )
        print(
            f"   huggingface-cli upload cmu-lti/stateful {output_dir}/README.md README.md"
        )

        return dataset


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Build the cmu-lti/stateful dataset")
    parser.add_argument(
        "--profiles-path",
        default="data/stateful_swe/user_profiles.jsonl",
        help="Path to user profiles JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/stateful_swe/dataset_for_upload",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for profile assignment"
    )

    args = parser.parse_args()

    if not DATASETS_AVAILABLE:
        print("❌ Required libraries not available.")
        print("Install with: pip install datasets huggingface_hub")
        return

    # Build the dataset
    builder = StatefulSWEDatasetBuilder(random_seed=args.seed)
    dataset = builder.build_and_save_complete_dataset(
        profiles_path=args.profiles_path, output_dir=args.output_dir
    )

    if dataset:
        print("\n📊 Dataset summary:")
        print(f"  - Total instances: {len(dataset)}")
        print(f"  - Columns: {len(dataset.column_names)}")
        print("  - Profile distribution:")

        from collections import Counter

        profile_dist = Counter(dataset["user_profile_id"])
        for profile_id, count in profile_dist.most_common():
            print(f"    - {profile_id}: {count}")


if __name__ == "__main__":
    main()
