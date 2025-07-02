import argparse
import json
import os

from openhands.core.config.agent_config import AgentConfig
from openhands.memory.conversation_memory import ConversationMemory
from openhands.utils.prompt import PromptManager


def initialize_conversation_memory():
    # 1. Create an instance of AgentConfig with default values
    agent_config = AgentConfig(
        enable_browsing=True,
        enable_editor=True,
        enable_jupyter=True,
        enable_cmd=True,
        enable_think=True,
        enable_finish=True,
        enable_prompt_extensions=True,
        enable_som_visual_browsing=True,
    )

    # 2. Create an instance of PromptManager
    # You need to provide a directory containing prompt templates
    prompt_dir = "/home/xuhuizhou/OpenHands/openhands/agenthub/codeact_agent/prompts"
    prompt_manager = PromptManager(prompt_dir=prompt_dir)

    # 3. Create an instance of ConversationMemory
    conversation_memory = ConversationMemory(agent_config, prompt_manager)

    return conversation_memory


def combine_events_to_trajectory(events, output_dir):
    """
    Combine events into a trajectory.
    """
    trajectory = []
    current_session = None
    current_conversation = None

    for event in events:
        # Process event and add to trajectory
        if event.get("type") == "session_start":
            current_session = event
            current_conversation = None
        elif event.get("type") == "conversation_start":
            current_conversation = event
        elif event.get("type") == "message":
            if current_conversation:
                current_conversation["messages"].append(event)
        elif event.get("type") == "conversation_end":
            if current_session and current_conversation:
                current_session["conversations"].append(current_conversation)
            current_conversation = None
        elif event.get("type") == "session_end":
            if current_session:
                trajectory.append(current_session)
            current_session = None

    # Save trajectory
    output_file = os.path.join(output_dir, "trajectory.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)

    return trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Combine events from all conversations in a session into a trajectory file"
    )
    parser.add_argument("user_id", help="Session ID to process")

    args = parser.parse_args()
    combine_events_to_trajectory(args.user_id)


if __name__ == "__main__":
    main()
