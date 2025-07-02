#!/usr/bin/env python3
"""
Script to analyze all users or a specific user in processed_data.

Usage:
    python analyze_all_users.py                                    # Analyze all users
    python analyze_all_users.py --user_id USER_ID                  # Analyze specific user
    python analyze_all_users.py --max_users 10                     # Analyze first 10 users
"""

import argparse
import asyncio
import os
import sys
from typing import List

from tom_module.tom_module import UserMentalStateAnalyzer


def get_all_user_ids(processed_data_dir: str) -> list[str]:
    """Get all user IDs from processed_data directory."""
    user_ids = []
    for filename in os.listdir(processed_data_dir):
        if filename.endswith(".json"):
            user_id = filename[:-5]  # Remove .json extension
            user_ids.append(user_id)
    return sorted(user_ids)


def check_session_already_processed(
    user_id: str, session_id: str, base_dir: str = "./data/user_model"
) -> bool:
    """Check if a session has already been processed by looking for the detailed analysis file."""
    detailed_file = os.path.join(base_dir, "user_model_detailed", user_id, f"{session_id}.json")
    return os.path.exists(detailed_file)


def get_unprocessed_sessions(
    user_id: str, session_ids: List[str], base_dir: str = "./data/user_model"
) -> List[str]:
    """Filter out sessions that have already been processed."""
    unprocessed = []
    for session_id in session_ids:
        if not check_session_already_processed(user_id, session_id, base_dir):
            unprocessed.append(session_id)
    return unprocessed


async def process_all_user_sessions(
    analyzer: UserMentalStateAnalyzer,
    user_id: str,
    base_dir: str = "./data/user_model",
    sample_size: int = 0,
    skip_existing: bool = True,
) -> bool:
    """
    Process all sessions for a user, reading metadata from studio_results files.
    Returns True if all sessions processed successfully, False otherwise.

    Args:
        analyzer: UserMentalStateAnalyzer instance
        user_id: The user ID to process
        base_dir: Output directory for analysis results
        sample_size: If > 0, limit to first N sessions
        skip_existing: If True, skip sessions that have already been processed
    """
    # Load metadata from studio_results CSV files
    print("Loading studio_results metadata...")
    studio_metadata = analyzer.load_studio_results_metadata()

    session_ids = await analyzer.get_user_session_ids(user_id)

    # Filter sessions based on studio metadata trigger field instead of session_id pattern
    gui_session_ids = []
    for session_id in session_ids:
        metadata = studio_metadata.get(session_id, {})
        trigger = metadata.get("trigger", "")
        if trigger == "gui":
            gui_session_ids.append(session_id)

    session_ids = gui_session_ids

    if sample_size > 0:
        session_ids = session_ids[:sample_size]

    if not session_ids:
        print(f"No GUI sessions found for user {user_id}")
        return False

    # Filter out already processed sessions if skip_existing is True
    original_count = len(session_ids)
    if skip_existing:
        session_ids = get_unprocessed_sessions(user_id, session_ids, base_dir)
        skipped_count = original_count - len(session_ids)
        if skipped_count > 0:
            print(f"Skipped {skipped_count} already processed sessions")

    if not session_ids:
        print(f"All sessions for user {user_id} have already been processed")
        return True

    print(
        f"Processing {len(session_ids)} GUI sessions for user {user_id} in batches of {analyzer.session_batch_size}"
    )

    all_session_summaries = []
    # Process sessions in batches to balance concurrency and time dependencies
    for i in range(0, len(session_ids), analyzer.session_batch_size):
        batch = session_ids[i : i + analyzer.session_batch_size]
        batch_num = i // analyzer.session_batch_size + 1
        total_batches = (
            len(session_ids) + analyzer.session_batch_size - 1
        ) // analyzer.session_batch_size

        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} sessions)")

        batch_session_summaries = await analyzer._process_session_batch(
            user_id, batch, studio_metadata, base_dir
        )
        all_session_summaries.extend(batch_session_summaries)

        print(
            f"Batch {batch_num} completed: {len(batch_session_summaries)}/{len(batch)} sessions successful"
        )

    # Update user profile once with all session summaries
    if all_session_summaries:
        await analyzer.save_updated_user_profile(user_id, all_session_summaries, base_dir)

    total_sessions = len(session_ids)
    success_count = len(all_session_summaries)
    print(f"✅ Completed {success_count}/{total_sessions} sessions for user {user_id}")
    return success_count == total_sessions


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Analyze user mental states from processed conversation data"
    )

    parser.add_argument("--user_id", type=str, help="Analyze specific user ID")
    parser.add_argument("--max_users", type=int, help="Maximum number of users to process")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="./data/processed_data",
        help="Directory containing processed user data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/user_model",
        help="Output directory for analysis results",
    )
    parser.add_argument("--model", type=str, help="LLM model to use for analysis")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all sessions, even if already processed",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Limit to first N sessions per user (0 = no limit)",
    )
    parser.add_argument(
        "--session_batch_size", type=int, default=50, help="Batch size for processing sessions"
    )
    return parser


def validate_and_get_user_ids(args) -> List[str]:
    """Validate arguments and return list of user IDs to process."""
    # Validate processed_data_dir exists
    if not os.path.exists(args.processed_data_dir):
        print(f"❌ Processed data directory not found: {args.processed_data_dir}")
        sys.exit(1)

    # Get user list
    if args.user_id:
        # Analyze specific user
        user_file = os.path.join(args.processed_data_dir, f"{args.user_id}.json")
        if not os.path.exists(user_file):
            print(f"❌ User {args.user_id} not found in processed_data")
            sys.exit(1)
        user_ids = [args.user_id]
    else:
        # Analyze all users
        user_ids = get_all_user_ids(args.processed_data_dir)
        if not user_ids:
            print(f"❌ No user files found in {args.processed_data_dir}")
            sys.exit(1)

        # Apply max_users limit
        if args.max_users and args.max_users < len(user_ids):
            user_ids = user_ids[: args.max_users]
            print(f"📝 Limited to first {args.max_users} users")

    return user_ids


async def process_users(user_ids: List[str], analyzer, args) -> None:
    """Process all users and print summary."""
    print(f"🚀 Starting analysis of {len(user_ids)} users")

    # Process each user
    successful = 0
    failed_users = []
    for i, user_id in enumerate(user_ids, 1):
        print(f"\n[{i}/{len(user_ids)}] Processing user: {user_id}")

        try:
            success = await process_all_user_sessions(
                analyzer,
                user_id,
                args.output_dir,
                sample_size=args.sample_size,
                skip_existing=not args.force,
            )
            if success:
                successful += 1
                print(f"✅ User {user_id} completed successfully")
            else:
                failed_users.append(user_id)
                print(f"❌ User {user_id} failed")
        except Exception as e:
            failed_users.append(user_id)
            print(f"❌ User {user_id} error: {e}")

    print_final_summary(user_ids, successful, failed_users)


def print_final_summary(user_ids: List[str], successful: int, failed_users: List[str]) -> None:
    """Print the final analysis summary."""
    print(f"\n{'='*60}")
    print("📊 ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"👥 Total users: {len(user_ids)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {len(failed_users)}")
    print(f"📊 Success rate: {successful/len(user_ids)*100:.1f}%")

    if failed_users:
        print(f"\n❌ Failed users: {', '.join(failed_users)}")

    # Exit with appropriate code
    sys.exit(0 if len(failed_users) == 0 else 1)


async def main():
    """Main function."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    user_ids = validate_and_get_user_ids(args)

    # Initialize analyzer
    analyzer = UserMentalStateAnalyzer(
        args.processed_data_dir, args.model, session_batch_size=args.session_batch_size
    )

    await process_users(user_ids, analyzer, args)


if __name__ == "__main__":
    asyncio.run(main())
