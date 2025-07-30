#!/usr/bin/env python3
"""
Simple example demonstrating the ToM Agent's propose_instructions function.

This script shows how to use the Theory of Mind Agent to improve user instructions
using real LLM analysis instead of mocks.

Requirements:
- Set up environment: uv run tom-config
- Run with: uv run python example.py
"""

import os

from dotenv import load_dotenv

from tom_swe.tom_agent import ToMAgent, ToMAgentConfig

# Load environment variables
load_dotenv()


def main():
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
        config = ToMAgentConfig(
            processed_data_dir="./data/processed_data",
            user_model_dir="./data/user_model",
            enable_rag=False,
        )
        agent = ToMAgent(config)
        print("✅ Agent initialized successfully")

        # Example instruction to improve
        user_id = "default_user"  # Use default_user for demo
        instruction = "Fix my code"
        context = "User is working on a Python web application"

        print(f"\n2. Testing instruction: '{instruction}'")
        print(f"   User: {user_id}")
        print(f"   Context: {context}")

        # Get improved instructions
        recommendation = agent.propose_instructions(
            user_id=user_id,
            original_instruction=instruction,
            user_msg_context=context,
        )

        # Show results
        if recommendation:
            rec = recommendation
            print(f"   ✅ Confidence: {rec.confidence_score:.0%}")
            print(f"   📝 Improved instruction:\n{rec.improved_instruction}")
        else:
            print("   ❌ No recommendations generated")

        print("\n🏁 Demo completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have configured your API credentials in the .env file")


if __name__ == "__main__":
    main()
