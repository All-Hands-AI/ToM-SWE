#!/usr/bin/env python3
"""
Simple example demonstrating the ToM Agent's propose_instructions function.

This script shows how to use the Theory of Mind Agent to improve user instructions
using real LLM analysis instead of mocks.

Requirements:
- Set up environment: uv run tom-config
- Run with: uv run python example.py
"""

import asyncio
import os

from dotenv import load_dotenv

from tom_swe.tom_agent import create_tom_agent

# Load environment variables
load_dotenv()


async def main():
    """Demonstrate the propose_instructions function."""
    print("🤖 ToM Agent - Propose Instructions Example")
    print("=" * 50)

    # Check if API key is configured
    if not os.getenv("LITELLM_API_KEY"):
        print("❌ Please run 'uv run tom-config' to set up your LLM API credentials")
        return

    try:
        # Initialize ToM Agent
        print("\n1. Initializing ToM Agent...")
        agent = await create_tom_agent(
            processed_data_dir="./data/processed_data",
            user_model_dir="./data/user_model",
            enable_rag=True,
        )
        print("✅ Agent initialized successfully")

        # Test cases to demonstrate the function
        test_cases = [
            {
                "user_id": "example_user",
                "instruction": "Fix my code",
                "context": "User is working on a Python web application",
            },
            {
                "user_id": "debug_user",
                "instruction": "My function doesn't work",
                "context": "User has been debugging for 2 hours",
            },
        ]

        # Run examples
        for i, test in enumerate(test_cases, 1):
            print(f"\n{i + 1}. Testing: '{test['instruction']}'")
            print(f"   User: {test['user_id']}")
            print(f"   Context: {test['context']}")

            # Analyze user context
            user_context = await agent.analyze_user_context(test["user_id"], test["instruction"])

            # Get improved instructions
            recommendations = await agent.propose_instructions(
                user_context=user_context,
                original_instruction=test["instruction"],
                user_msg_context=test["context"],
            )

            # Show results
            if recommendations:
                rec = recommendations[0]
                print(f"   ✅ Confidence: {rec.confidence_score:.0%}")
                print(f"   📝 Improved: {rec.improved_instruction[:100]}...")
            else:
                print("   ❌ No recommendations generated")

        print("\n🏁 Demo completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have configured your API credentials in the .env file")


if __name__ == "__main__":
    asyncio.run(main())
