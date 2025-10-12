#!/usr/bin/env python3
"""
Simple example demonstrating the ToM Agent's consultation functionality.

This script shows how to use the Theory of Mind Agent to provide guidance and
consultation for SWE agents using real LLM analysis instead of mocks.

Requirements:
- Set up environment: uv run tom-config
- Run with: uv run python example.py
"""

import os
import json
from dotenv import load_dotenv

from tom_swe.tom_agent import ToMAgent, ToMAgentConfig
from tom_swe.memory.local import LocalFileStore

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
    sessions_data = sessions_data[:1]

    agent = ToMAgent()
    agent.sleeptime_compute(sessions_data)

    print("✅ Sessions processed and saved successfully!")
    # Clean up test data after showing results
    # if test_dir.exists():
    #     shutil.rmtree(test_dir)
    #     print(f"\n🧹 Cleaned up test data: {test_dir}")


def test_pure_rag():
    """Test Pure RAG baseline mode."""
    print("🔍 Testing Pure RAG Baseline")
    print("=" * 30)
    config = ToMAgentConfig(
        file_store=LocalFileStore(root="~/.openhands"),
        llm_model="litellm_proxy/claude-sonnet-4-20250514",
    )
    agent = ToMAgent(config)
    query = "Help me fix a bug"

    # Pure RAG mode
    result = agent.give_suggestions(query=query, pure_rag=True)
    print(f"{result.suggestions}")


def main():
    """Demonstrate the ToM Agent consultation functionality."""
    print("🤖 ToM Agent - Consultation Example")
    print("=" * 50)

    # Check if API key is configured
    if not os.getenv("LITELLM_API_KEY"):
        print("❌ Please run 'uv run tom-config' to set up your LLM API credentials")
        return

    try:
        # Initialize ToM Agent
        print("\n1. Initializing ToM Agent...")
        config = ToMAgentConfig(
            file_store=LocalFileStore(root="~/.openhands"),
            # llm_model="litellm_proxy/claude-3-7-sonnet-20250219",
            llm_model="litellm_proxy/claude-sonnet-4-20250514",
        )
        agent = ToMAgent(config)
        print("✅ Agent initialized successfully")

        # Example instruction for consultation
        user_id = ""  # Use default_user for demo
        formatted_messages = []
        with open("./data/improve_instruction_example/context_swe_interact.jsonl") as f:
            lines = f.readlines()
            for line in lines:
                formatted_messages.append(json.loads(line))
        instruction = f"I am SWE agent. I need to understand the user's intent and expectations for fixing the Pipeline class issue. I need to consult ToM agent about the user's message: {formatted_messages[-1]['content'][0]['text']}"
        # Get consultation guidance using ToM API
        recommendation = agent.give_suggestions(
            user_id=user_id,
            query=instruction,
            formatted_messages=formatted_messages,
        )

        # Show results
        if recommendation:
            rec = recommendation
            print(f"   ✅ Confidence: {rec.confidence_score:.0%}")
            print(f"   📝 Consultation guidance:\n{rec.suggestions}")
        else:
            print("   ❌ No recommendations generated")

        print("\n🏁 Demo completed!")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Test the sleeptime function first
    # test_sleeptime()
    # print("\n" + "=" * 50 + "\n")

    # Test pure RAG baseline
    test_pure_rag()
    print("\n" + "=" * 50 + "\n")

    # Then run the main ToM agent demo
    # main()
