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
import json
from dotenv import load_dotenv

from tom_swe.tom_agent import ToMAgent, ToMAgentConfig

# Load environment variables
load_dotenv()


def test_sleeptime():
    """Test the sleeptime_compute function."""

    print("🔄 Testing sleeptime_compute function")
    print("=" * 40)

    # Example session data matching the format you provided
    # example_sessions_data = [json.load(open("example.json"))]
    sessions_data = []
    for file in os.listdir("./data/example_sessions"):
        session_data = json.load(open(f"./data/example_sessions/{file}"))
        sessions_data.append(session_data)

    agent = ToMAgent()
    agent.sleeptime_compute(sessions_data)

    print("✅ Sessions processed and saved successfully!")
    # Clean up test data after showing results
    # if test_dir.exists():
    #     shutil.rmtree(test_dir)
    #     print(f"\n🧹 Cleaned up test data: {test_dir}")


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
        config = ToMAgentConfig()
        agent = ToMAgent(config)
        print("✅ Agent initialized successfully")

        # Example instruction to improve
        user_id = ""  # Use default_user for demo
        instruction = "why not ask the ToM agent for help?"
        context = """Hello! I'm doing well, thank you for asking! I'm here and ready to help you with any tasks you might have.
I can see I'm currently in your /Users/xuhuizhou/Projects/OpenHands/workspace directory. Whether you need help with:
- Code development or debugging
- File management and editing
- Running tests or applications
- Git operations
- Or any other technical tasks
Just let me know what you'd like to work on, and I'll be happy to assist! What can I help you with today?"""

        print(f"\n2. Testing instruction: '{instruction}'")
        print(f"   User: {user_id}")
        print(f"   Context: {context}")

        # Create formatted messages with caching support
        formatted_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
                "cache_control": {"type": "ephemeral"},  # Cache system message
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": context}],  # Context message
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": instruction}],  # Main instruction
            },
        ]

        # Get improved instructions using new API
        recommendation = agent.propose_instructions(
            user_id=user_id,
            original_instruction=instruction,
            formatted_messages=formatted_messages,
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
    # Test the sleeptime function first
    # test_sleeptime()
    # print("\n" + "=" * 50 + "\n")

    # Then run the main ToM agent demo
    main()
