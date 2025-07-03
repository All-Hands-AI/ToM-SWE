#!/usr/bin/env python3
"""
Example script demonstrating the ToM Agent functionality.

This script shows how to:
1. Create a ToM agent that combines user behavior analysis with RAG
2. Analyze user context and mental state
3. Generate personalized instruction improvements
4. Suggest next actions based on user preferences
5. Provide comprehensive personalized guidance

Usage:
    python tom_agent_example.py --user_id USER_ID [--instruction "instruction text"]
"""

import argparse
import asyncio
import sys
from pathlib import Path

from tom_swe import ToMAgent, create_tom_agent


async def demonstrate_user_context_analysis(agent: ToMAgent, user_id: str) -> None:
    """Demonstrate user context analysis."""
    print("\n" + "=" * 60)
    print("USER CONTEXT ANALYSIS")
    print("=" * 60)

    user_context = await agent.analyze_user_context(
        user_id=user_id, current_query="Working on debugging a Python application"
    )

    print(f"User ID: {user_context.user_id}")
    print(f"Mental State Summary: {user_context.mental_state_summary or 'No profile available'}")
    print(f"Preferences: {', '.join(user_context.preferences or ['None identified'])}")
    print(f"Recent Sessions: {len(user_context.recent_sessions)} sessions analyzed")

    if user_context.recent_sessions:
        print("\nRecent Session Summary:")
        for i, session in enumerate(user_context.recent_sessions[-3:], 1):
            intents = list(session.intent_distribution.keys())
            emotions = list(session.emotion_distribution.keys())
            print(f"  Session {i}: {', '.join(intents)} | {', '.join(emotions)}")


async def demonstrate_instruction_improvement(
    agent: ToMAgent, user_id: str, instruction: str
) -> None:
    """Demonstrate personalized instruction improvement."""
    print("\n" + "=" * 60)
    print("INSTRUCTION IMPROVEMENT")
    print("=" * 60)

    user_context = await agent.analyze_user_context(user_id)

    recommendations = await agent.propose_instructions(
        user_context=user_context,
        original_instruction=instruction,
        domain_context="Python development and debugging",
    )

    print(f"Original instruction: {instruction}")
    print(f"Number of recommendations: {len(recommendations)}")

    for i, rec in enumerate(recommendations, 1):
        print(f"\nRecommendation {i}:")
        print(f"  Improved: {rec.improved_instruction}")
        print(f"  Reasoning: {rec.reasoning}")
        print(f"  Confidence: {rec.confidence_score:.2f}")
        print(f"  Personalization factors: {', '.join(rec.personalization_factors)}")


async def demonstrate_next_action_suggestions(agent: ToMAgent, user_id: str) -> None:
    """Demonstrate next action suggestions."""
    print("\n" + "=" * 60)
    print("NEXT ACTION SUGGESTIONS")
    print("=" * 60)

    user_context = await agent.analyze_user_context(user_id)

    suggestions = await agent.suggest_next_actions(
        user_context=user_context,
        current_task_context="Debugging and fixing errors in a Python web application",
    )

    print(f"Number of suggestions: {len(suggestions)}")

    for i, sug in enumerate(suggestions, 1):
        print(f"\nSuggestion {i} [{sug.priority.upper()}]:")
        print(f"  Action: {sug.action_description}")
        print(f"  Reasoning: {sug.reasoning}")
        print(f"  Expected outcome: {sug.expected_outcome}")
        print(f"  User alignment: {sug.user_preference_alignment:.2f}")


async def demonstrate_comprehensive_guidance(
    agent: ToMAgent, user_id: str, instruction: str
) -> None:
    """Demonstrate comprehensive personalized guidance."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE PERSONALIZED GUIDANCE")
    print("=" * 60)

    guidance = await agent.get_personalized_guidance(
        user_id=user_id,
        instruction=instruction,
        current_task="Debugging Python application with performance issues",
        domain_context="Web development with Flask/Django",
    )

    print(f"Overall Guidance: {guidance.overall_guidance}")
    print(f"Confidence Score: {guidance.confidence_score:.2f}")

    if guidance.instruction_recommendations:
        print(f"\nInstruction Improvements ({len(guidance.instruction_recommendations)}):")
        for rec in guidance.instruction_recommendations:
            print(f"  • {rec.improved_instruction}")
            print(f"    (Confidence: {rec.confidence_score:.2f})")

    if guidance.next_action_suggestions:
        print(f"\nNext Action Suggestions ({len(guidance.next_action_suggestions)}):")
        for sug in guidance.next_action_suggestions:
            print(f"  • [{sug.priority.upper()}] {sug.action_description}")
            print(f"    Expected: {sug.expected_outcome}")


async def main() -> None:
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="ToM Agent demonstration script")
    parser.add_argument(
        "--user_id",
        type=str,
        required=True,
        help="User ID to analyze (must exist in processed_data)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Debug the function that's causing errors and fix the performance issues",
        help="Instruction to improve (default: debugging example)",
    )
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="./data/processed_data",
        help="Directory containing processed user data",
    )
    parser.add_argument(
        "--user_model_dir",
        type=str,
        default="./data/user_model",
        help="Directory containing user model data",
    )

    args = parser.parse_args()

    # Validate that directories exist
    if not Path(args.processed_data_dir).exists():
        print(f"❌ Processed data directory not found: {args.processed_data_dir}")
        sys.exit(1)

    # Check if user exists
    user_file = Path(args.processed_data_dir) / f"{args.user_id}.json"
    if not user_file.exists():
        print(f"❌ User file not found: {user_file}")
        print("Available users:")
        for user_file in Path(args.processed_data_dir).glob("*.json"):
            print(f"  - {user_file.stem}")
        sys.exit(1)

    print("🤖 ToM Agent Demonstration")
    print(f"User ID: {args.user_id}")
    print(f"Instruction: {args.instruction}")

    # Create ToM agent
    print("\nInitializing ToM Agent...")
    agent = await create_tom_agent(
        processed_data_dir=args.processed_data_dir, user_model_dir=args.user_model_dir
    )
    print("✅ ToM Agent initialized successfully")

    try:
        # Run demonstrations
        await demonstrate_user_context_analysis(agent, args.user_id)
        await demonstrate_instruction_improvement(agent, args.user_id, args.instruction)
        await demonstrate_next_action_suggestions(agent, args.user_id)
        await demonstrate_comprehensive_guidance(agent, args.user_id, args.instruction)

        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
