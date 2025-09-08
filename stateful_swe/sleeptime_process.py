#!/usr/bin/env python3
"""
Sleeptime Processing Script for Stateful SWE Benchmark

Processes washed session files through the sleeptime_compute function to generate
user modeling files saved under data/stateful_swe/usermodeling/{user_id}/
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm

from tom_swe.tom_agent import ToMAgentConfig
from tom_swe.tom_agent import ToMAgent
from tom_swe.memory.local import LocalFileStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_washed_session_file(file_path: str) -> Dict[str, Any]:
    """Load a washed session file.

    Args:
        file_path: Path to the washed session JSONL file

    Returns:
        Dictionary containing user_profile and user_sessions
    """
    with open(file_path, "r") as f:
        data: Dict[str, Any] = json.load(f)  # Single JSON object, not JSONL
    return data


def extract_sessions_for_sleeptime(washed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract session data in the format expected by sleeptime_compute.

    Args:
        washed_data: Washed session data with user_profile and user_sessions

    Returns:
        List of sessions formatted for sleeptime_compute
    """
    user_sessions = washed_data.get("user_sessions", [])

    # Convert washed session format to sleeptime format
    sleeptime_sessions = []
    for session in user_sessions:
        # Transform the session format to match what sleeptime_compute expects
        sleeptime_session = {
            "session_id": session.get("session_id", ""),
            "start_time": session.get("start_time", ""),
            "end_time": session.get("end_time", ""),
            "conversation_messages": session.get("conversation_messages", []),
        }
        sleeptime_sessions.append(sleeptime_session)

    return sleeptime_sessions


def anonymize_user_model(profile_id: str, usermodeling_dir: str) -> None:
    """Anonymize the user model by removing user_id to avoid leaking interactional traits.

    Args:
        profile_id: The profile ID used during processing
        usermodeling_dir: Base directory containing user modeling files
    """
    try:
        # Construct the expected path for the overall user model (expand ~ if present)
        usermodeling_dir_expanded = os.path.expanduser(usermodeling_dir)
        user_model_path = (
            Path(usermodeling_dir_expanded)
            / "usermodeling"
            / "users"
            / profile_id
            / "overall_user_model.json"
        )

        if user_model_path.exists():
            # Load the user model
            with open(user_model_path, "r") as f:
                user_model = json.load(f)

            # Remove/anonymize the user_id field to avoid leaking profile traits
            if "user_profile" in user_model and "user_id" in user_model["user_profile"]:
                user_model["user_profile"]["user_id"] = ""
                logger.info(f"Anonymized user_profile.user_id in {user_model_path}")
            elif "user_id" in user_model:
                user_model["user_id"] = ""
                logger.info(f"Anonymized user_id in {user_model_path}")

            # Save the anonymized model back
            with open(user_model_path, "w") as f:
                json.dump(user_model, f, indent=2)

            print(f"✅ Anonymized user model for profile {profile_id}")
        else:
            logger.warning(f"Overall user model not found at {user_model_path}")

    except Exception as e:
        logger.error(f"Failed to anonymize user model for {profile_id}: {e}")


def process_profile_with_sleeptime(profile_file: str, usermodeling_dir: str) -> None:
    """Process a single profile file through sleeptime_compute.

    Args:
        profile_file: Path to the profile's washed session file
        usermodeling_dir: Base directory for user modeling output
    """

    # Load washed session data
    washed_data = load_washed_session_file(profile_file)

    # Extract profile information
    user_profile = washed_data.get("user_profile", {})
    profile_id = user_profile.get("profile_id", "unknown_profile")

    # Extract sessions for sleeptime processing
    sessions_for_sleeptime = extract_sessions_for_sleeptime(washed_data)

    if not sessions_for_sleeptime:
        logger.warning(f"No sessions found for profile {profile_id}")
        return

    print(
        f"🧠 Processing {len(sessions_for_sleeptime)} sessions for profile {profile_id}"
    )

    # Initialize ToM Agent with custom file store pointing to the usermodeling directory
    file_store = LocalFileStore(root=f"{usermodeling_dir}")
    config = ToMAgentConfig(file_store=file_store)
    agent = ToMAgent(config)

    # Process sessions through sleeptime_compute
    # Use profile_id as user_id since each profile represents a synthetic user
    agent.sleeptime_compute(sessions_data=sessions_for_sleeptime, user_id=profile_id)

    # After sleeptime processing, clean up the user_id field in the overall user model
    anonymize_user_model(profile_id, usermodeling_dir)


def process_all_washed_sessions(
    washed_sessions_dir: str, usermodeling_dir: str
) -> None:
    """Process all washed session files through sleeptime_compute.

    Args:
        washed_sessions_dir: Directory containing washed session files
        usermodeling_dir: Base directory for user modeling output
    """
    washed_dir = Path(washed_sessions_dir)

    if not washed_dir.exists():
        logger.error(f"Washed sessions directory does not exist: {washed_sessions_dir}")
        return

    # Find all JSONL files in the washed sessions directory
    session_files = list(washed_dir.glob("*.jsonl"))

    if not session_files:
        logger.error(f"No washed session files found in {washed_sessions_dir}")
        return

    print(f"🔄 Found {len(session_files)} washed session files to process")
    print(f"📁 Output will be saved to: {usermodeling_dir}")

    # Process each profile file with progress bar
    with tqdm(
        total=len(session_files), desc="Processing profiles", unit="profile"
    ) as pbar:
        for session_file in session_files:
            profile_name = session_file.stem
            pbar.set_description(f"Processing {profile_name}")

            try:
                process_profile_with_sleeptime(str(session_file), usermodeling_dir)
                pbar.set_postfix({"status": "✅"})
            except Exception as e:
                logger.error(f"Failed to process {session_file}: {e}")
                pbar.set_postfix({"status": "❌"})

            pbar.update(1)

    print("\n🎉 Sleeptime processing complete!")
    print(f"  - Processed {len(session_files)} profiles")
    print(f"  - User modeling data saved to: {usermodeling_dir}/")


def main():
    """Main entry point for sleeptime processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process washed sessions through sleeptime_compute"
    )
    parser.add_argument(
        "--washed-dir",
        default="data/stateful_swe/washed_sessions",
        help="Directory containing washed session files",
    )
    parser.add_argument(
        "--output-dir",
        default="~/Projects/ToM-SWE/data/stateful_swe",
        help="Output directory for user modeling files",
    )
    parser.add_argument(
        "--profile-file", help="Process a single profile file instead of all files"
    )

    args = parser.parse_args()

    print("🧠 Stateful SWE - Sleeptime Processing")
    print("=" * 40)

    if args.profile_file:
        # Process single file
        print(f"Processing single file: {args.profile_file}")
        process_profile_with_sleeptime(args.profile_file, args.output_dir)
    else:
        # Process all files
        process_all_washed_sessions(args.washed_dir, args.output_dir)


if __name__ == "__main__":
    main()
