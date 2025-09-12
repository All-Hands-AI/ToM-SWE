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
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
import logging
from datetime import datetime

from datasets import Dataset, load_dataset

# Try to import LLM dependencies
try:
    from tom_swe.generation.generate import LLMClient, LLMConfig
    from pydantic import BaseModel, Field
    import os
    from dotenv import load_dotenv

    load_dotenv()

    LLM_AVAILABLE = True

    # Define VagueStatementResponse inside the try block
    class VagueStatementResponse(BaseModel):
        """Pydantic model for LLM response when making statements more vague."""

        modified_statement: str = Field(
            description="The problem statement rewritten to be more ambiguous/vague while maintaining the user's voice and profile characteristics"
        )
        reasoning: str = Field(
            description="Brief explanation of how the statement was made more vague while reflecting the user profile"
        )

except ImportError:
    LLM_AVAILABLE = False
    print(
        "⚠️ LLM dependencies not available. Problem statement modification will be skipped."
    )

    # Define a dummy class when dependencies aren't available
    class VagueStatementResponse:  # type: ignore[no-redef]
        pass


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASETS_AVAILABLE = True


class StatefulSWEDatasetBuilder:
    """Builds the cmu-lti/stateful dataset based on interactive-swe with user profiles."""

    def __init__(
        self,
        random_seed: int = 42,
        modify_statements: bool = True,
        llm_model: str = "gpt-5-2025-08-07",
    ):
        self.random_seed = random_seed
        self.modify_statements = modify_statements and LLM_AVAILABLE
        self.llm_model = llm_model
        random.seed(random_seed)

        # Initialize LLM client if needed
        if self.modify_statements:
            api_key = os.getenv("LITELLM_API_KEY")
            if not api_key:
                logger.warning(
                    "No LITELLM_API_KEY found. Problem statement modification will be skipped."
                )
                self.modify_statements = False
            else:
                llm_config = LLMConfig(
                    model=self.llm_model,
                )
                self.llm_client = LLMClient(llm_config)
                logger.info(f"LLM client initialized with model: {self.llm_model}")

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

    async def modify_problem_statement_async(
        self, problem_statement: str, user_profile: Dict[str, Any]
    ) -> str:
        """
        Modify a problem statement to be more ambiguous/vague while reflecting user profile characteristics.

        Args:
            problem_statement: Original clear problem statement
            user_profile: User profile to inform the modification

        Returns:
            Modified vague problem statement
        """
        if not self.modify_statements:
            return problem_statement

        try:
            # Extract relevant profile information
            user_prompt = user_profile.get("user_roleplay_prompt", "")

            # Create prompt for LLM to modify the statement
            prompt = f"""You are helping to create a more realistic user query by making a technical problem statement sound more ambiguous and vague, as a real user with specific characteristics would phrase it.

USER PROFILE CHARACTERISTICS:
{user_prompt}

ORIGINAL PROBLEM STATEMENT:
{problem_statement}

TASK:
Rewrite this problem statement to be more ambiguous/vague while maintaining the user's voice and profile characteristics. The modified statement should:

1. Sound more like how this specific user would phrase it based on their profile
2. Be less precise and more ambiguous than the original
3. Remove specific technical details to the degree that agent has to ask questions/engage with users to finish the task. And do not give any hints about what they should expect agents to behave (e.g., ask questions, etc.)
4. Use more conversational/informal language if it fits the user profile
5. Do not leak your preferences in the statement (e.g., tell the swe agent to keep it short), as this should be part of swe agent's job to figure that out.
6. Use at most 3 sentences. (One sentence is preferred, but you can adjust depending on the user profile)
7. The problem statement is for swe agent, so do not try to be polite or something. Also do not give agent any guidance about how to finish the task. The goal is for the agent to make follow up engagements with users to finish the task (however, do not tell them to engage with the users, we expect agents to figure that out themselves).

The goal is to make the statement sound more realistic, while staying true to the user's communication style."""

            # Combine system message and user prompt into a single prompt
            full_prompt = f"""You are an expert at understanding user communication patterns and rewriting technical statements to match specific user profiles.

{prompt}"""

            # Call LLM asynchronously
            response = await self.llm_client.call_structured_async(
                prompt=full_prompt, output_type=VagueStatementResponse
            )

            if response and response.modified_statement:
                logger.debug(
                    f"Modified statement: {response.modified_statement[:100]}..."
                )
                return response.modified_statement
            else:
                logger.warning(
                    "LLM did not return a modified statement, using original"
                )
                return problem_statement

        except Exception as e:
            logger.warning(f"Failed to modify problem statement: {e}. Using original.")
            return problem_statement

    async def process_statements_batch(
        self,
        statements_and_profiles: List[Tuple[str, Dict[str, Any]]],
        batch_size: int = 50,
    ) -> List[str]:
        """
        Process multiple problem statements in batches asynchronously.

        Args:
            statements_and_profiles: List of (statement, profile) tuples
            batch_size: Number of concurrent LLM calls

        Returns:
            List of modified statements
        """
        if not self.modify_statements:
            return [stmt for stmt, _ in statements_and_profiles]

        logger.info(
            f"Modifying {len(statements_and_profiles)} problem statements in batches of {batch_size}"
        )

        results = []
        for i in range(0, len(statements_and_profiles), batch_size):
            batch = statements_and_profiles[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(statements_and_profiles) + batch_size - 1)//batch_size}"
            )

            # Process batch concurrently
            batch_tasks = [
                self.modify_problem_statement_async(stmt, profile)
                for stmt, profile in batch
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle any exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.warning(
                        f"Error in batch item {i+j}: {result}. Using original statement."
                    )
                    results.append(batch[j][0])  # Use original statement
                else:
                    results.append(str(result))  # Ensure result is string

        logger.info(f"Successfully modified {len(results)} problem statements")
        return results

    async def create_stateful_dataset_async(
        self, base_dataset: Any, user_profiles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create enriched instances with user profiles and modified problem statements.

        Args:
            base_dataset: The original interactive-swe dataset
            user_profiles: List of user profiles to assign

        Returns:
            List of enriched instances with assigned user profiles
        """
        profile_counts = {profile["profile_id"]: 0 for profile in user_profiles}
        instances = []

        # First pass: collect all instances with profiles
        statements_and_profiles = []

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

            instances.append(enriched_instance)

            # Collect for batch processing if modification is enabled
            if self.modify_statements:
                statements_and_profiles.append(
                    (instance.get("problem_statement", ""), selected_profile)
                )

        # Modify problem statements in batches if enabled
        if self.modify_statements and statements_and_profiles:
            print(
                "🔄 Modifying problem statements to be more vague and profile-specific..."
            )
            modified_statements = await self.process_statements_batch(
                statements_and_profiles
            )

            # Update instances with modified statements
            for i, modified_statement in enumerate(modified_statements):
                instances[i]["problem_statement"] = modified_statement

        # Log final distribution
        print("\n📊 Final profile distribution:")
        for profile_id, count in sorted(profile_counts.items()):
            print(f"  - {profile_id}: {count} instances")

        return instances

    def create_stateful_dataset_generator(
        self, instances: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Generator that yields enriched instances.

        Args:
            instances: Pre-processed enriched instances

        Yields:
            Enriched instances with assigned user profiles
        """
        for instance in instances:
            yield instance

    async def build_dataset_async(
        self,
        profiles_path: str = "data/stateful_swe/user_profiles.jsonl",
        base_dataset_name: str = "cmu-lti/interactive-swe",
        split: str = "test",
        limit: Optional[int] = None,
    ) -> Optional[Dataset]:
        """Build the complete stateful dataset with async problem statement modification.

        Args:
            profiles_path: Path to user profiles JSONL file
            base_dataset_name: Name of the base HuggingFace dataset
            split: Dataset split to use
            limit: Optional limit on number of instances to process

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

            # Apply limit if specified
            if limit is not None:
                print(f"🔢 Limiting to first {limit} instances for testing")
                base_dataset = base_dataset.select(range(min(limit, len(base_dataset))))

            print(f"✅ Loaded {len(base_dataset)} instances from {base_dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load base dataset: {e}")
            return None

        # Create enriched instances with async processing
        print("🔄 Creating stateful dataset with profile assignments...")
        if self.modify_statements:
            print("🔄 Problem statement modification enabled")

        enriched_instances = await self.create_stateful_dataset_async(
            base_dataset, user_profiles
        )

        # Create the stateful dataset from the enriched instances
        stateful_dataset = Dataset.from_generator(
            lambda: self.create_stateful_dataset_generator(enriched_instances)
        )

        print(f"✅ Built stateful dataset with {len(stateful_dataset)} instances")

        return stateful_dataset

    def build_dataset(
        self,
        profiles_path: str = "data/stateful_swe/user_profiles.jsonl",
        base_dataset_name: str = "cmu-lti/interactive-swe",
        split: str = "test",
        limit: Optional[int] = None,
    ) -> Optional[Dataset]:
        """Build the complete stateful dataset (sync wrapper).

        Args:
            profiles_path: Path to user profiles JSONL file
            base_dataset_name: Name of the base HuggingFace dataset
            split: Dataset split to use

        Returns:
            The built Dataset object or None if failed
        """
        return asyncio.run(
            self.build_dataset_async(profiles_path, base_dataset_name, split, limit)
        )

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
        # dataset.save_to_disk(str(output_path / "dataset"))

        # Save as parquet file for direct upload
        dataset.to_parquet(str(output_path / "test-00000-of-00001.parquet"))

        # Save as JSON for inspection
        print("💾 Saving dataset as JSON for inspection...")
        dataset.to_json(
            str(output_path / "dataset.json"), orient="records", lines=False, indent=2
        )

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
        print(f"  - Parquet file: {output_path}/test-00000-of-00001.parquet")
        print(f"  - JSON file: {output_path}/dataset.json")
        print(f"  - Dataset card: {output_path}/README.md")
        print(f"  - Info file: {output_path}/dataset_info.json")

    def build_and_save_complete_dataset(
        self,
        profiles_path: str = "data/stateful_swe/user_profiles.jsonl",
        output_dir: str = "data/stateful_swe/dataset_for_upload",
        limit: Optional[int] = None,
    ) -> Optional[Dataset]:
        """Complete pipeline: build and save the stateful dataset.

        Args:
            profiles_path: Path to user profiles
            output_dir: Output directory

        Returns:
            The built dataset or None if failed
        """
        # Build the dataset
        dataset = self.build_dataset(profiles_path=profiles_path, limit=limit)

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
    parser.add_argument(
        "--modify-statements",
        action="store_true",
        help="Enable LLM-based problem statement modification to make them more vague/ambiguous",
    )
    parser.add_argument(
        "--no-modify-statements",
        action="store_true",
        help="Disable LLM-based problem statement modification",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-5-2025-08-07",
        help="LLM model to use for problem statement modification",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of instances to process (for testing)",
    )

    args = parser.parse_args()

    if not DATASETS_AVAILABLE:
        print("❌ Required libraries not available.")
        print("Install with: pip install datasets huggingface_hub")
        return

    # Determine whether to modify statements
    modify_statements = True  # Default to enabled
    if args.no_modify_statements:
        modify_statements = False
    elif args.modify_statements:
        modify_statements = True

    # Build the dataset
    builder = StatefulSWEDatasetBuilder(
        random_seed=args.seed,
        modify_statements=modify_statements,
        llm_model=args.llm_model,
    )
    dataset = builder.build_and_save_complete_dataset(
        profiles_path=args.profiles_path, output_dir=args.output_dir, limit=args.limit
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
