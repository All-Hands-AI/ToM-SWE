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
from tom_swe import sleeptime_compute

# Load environment variables
load_dotenv()


def test_sleeptime():
    """Test the sleeptime_compute function."""
    import shutil
    from pathlib import Path
    from tom_swe.memory.locations import get_cleaned_sessions_dir

    print("🔄 Testing sleeptime_compute function")
    print("=" * 40)

    # Example session data matching the format you provided
    example_sessions_data = [
        {
            "session_id": "test_session_001",
            "event_count": 10,
            "message_count": 4,
            "conversation_messages": [
                {
                    "role": "user",
                    "content": "I would like to compose a brief email from Graham at All Hands to be sent through resend.com to the OpenHands Email List audience. Please:\n1. Read the resend.com API docs thoroughly\n2. Take a look at the All-Hands-AI/company-website repo for the most recent two blog posts, and summarize both of them into a paragraph-long summary\n3. Read the All-Hands-AI/company-metrics repo's email_tools directory, and prepare a template for this new email.\n4. Send a test email to me graham@all-hands.dev\n\nI will provide you with a resend.com API key.",
                },
                {
                    "role": "assistant",
                    "content": "I'll help you compose and send that email. Let me break this down into steps and work through each one systematically...",
                },
                {"role": "user", "content": "Great, here's the API key: re_abc123"},
                {
                    "role": "assistant",
                    "content": "Thank you for the API key. I'll now proceed with the steps...",
                },
            ],
        }
    ]

    # Clean up any existing test data first
    test_dir = Path(get_cleaned_sessions_dir("", user_id=""))
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"🧹 Cleaned up existing test data: {test_dir}")

    # Process and automatically save the sessions
    clean_session_stores = sleeptime_compute(example_sessions_data)

    # Print results
    print(f"✅ Processed and saved {len(clean_session_stores)} sessions")
    for store in clean_session_stores:
        session = store.clean_session
        print(f"\n📋 Session: {session.session_id}")
        print(f"   Messages: {len(session.messages)}")

        # Show message details
        important_count = sum(1 for msg in session.messages if msg.is_important)
        print(f"   Important user messages: {important_count}")

        for i, msg in enumerate(session.messages):
            status = " (IMPORTANT)" if msg.is_important else ""
            print(f"   {i+1}. {msg.role}: {msg.content[:50]}...{status}")
    breakpoint()

    # Clean up test data after showing results
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"\n🧹 Cleaned up test data: {test_dir}")


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
        user_id = ""  # Use default_user for demo
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
    # Test the sleeptime function first
    test_sleeptime()
    print("\n" + "=" * 50 + "\n")

    # Then run the main ToM agent demo
    # main()
