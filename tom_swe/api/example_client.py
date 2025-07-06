#!/usr/bin/env python3
"""
Example client for the ToM Agent REST API.

This script demonstrates how to interact with the redesigned ToM Agent API
endpoints with the new bidirectional communication flow.
"""

import asyncio
from typing import Any, Dict, Optional, cast

import httpx


class ToMAgentClient:
    """Client for interacting with the ToM Agent API."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize the client with the API base URL."""
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the API server."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

    async def send_message(
        self,
        user_id: str,
        message: str,
        message_type: str,
        context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Send a message to the ToM agent (new API)."""
        data = {
            "user_id": user_id,
            "message": message,
            "message_type": message_type,
        }
        if context:
            data.update(context)

        response = await self.client.post(f"{self.base_url}/send_msg", json=data)
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

    async def get_propose_instructions(self, user_id: str) -> Dict[str, Any]:
        """Get cached instruction improvements for a user (new API)."""
        response = await self.client.get(
            f"{self.base_url}/propose_instructions", params={"user_id": user_id}
        )
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

    async def get_suggest_next_actions(self, user_id: str) -> Dict[str, Any]:
        """Get cached next action suggestions for a user (new API)."""
        response = await self.client.get(
            f"{self.base_url}/suggest_next_actions", params={"user_id": user_id}
        )
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

    async def get_conversation_status(self, user_id: str) -> Dict[str, Any]:
        """Get the current conversation status for a user."""
        response = await self.client.get(
            f"{self.base_url}/conversation_status", params={"user_id": user_id}
        )
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

    async def get_active_conversations(self) -> Dict[str, Any]:
        """Get information about all active conversations."""
        response = await self.client.get(f"{self.base_url}/active_conversations")
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())


async def check_api_health(client: ToMAgentClient) -> None:
    """Check API health and print status."""
    print("🔍 Checking API health...")
    health = await client.health_check()
    print(f"✅ API Status: {health['status']}")
    print(f"📊 ToM Agent Ready: {health['tom_agent_ready']}")
    print(f"🆔 API Version: {health.get('version', 'unknown')}")

    # Handle both old and new API versions
    if "active_conversations" in health:
        print(f"🗣️  Active Conversations: {health['active_conversations']}")
    else:
        print("🗣️  Active Conversations: Not supported in this API version")
    print()


async def check_conversation_status(client: ToMAgentClient, user_id: str) -> None:
    """Check and print conversation status for a user."""
    print(f"👤 Checking conversation status for user {user_id}...")
    try:
        status = await client.get_conversation_status(user_id)
        print("✅ Conversation status retrieved:")
        print(f"📝 Has pending instructions: {status['has_pending_instructions']}")
        print(f"🎯 Has pending next actions: {status['has_pending_next_actions']}")
        if status.get("last_activity"):
            print(f"⏱️  Last activity: {status['last_activity']}")
    except httpx.HTTPStatusError as e:
        print(f"⚠️  Status check failed: {e.response.status_code}")
    print()


async def send_user_message(client: ToMAgentClient, user_id: str) -> None:
    """Send a user message and handle response."""
    print("📤 Step 1: Sending user message...")
    try:
        context = {
            "original_instruction": "Debug the function that's causing errors",
            "task_context": "Debugging Python application",
            "domain_context": "Python web development",
        }
        send_response = await client.send_message(
            user_id=user_id,
            message="I need help debugging this Python function that keeps throwing IndexError",
            message_type="user_message",
            context=context,
        )
        print("✅ User message sent successfully:")
        print(f"📋 Processing type: {send_response['processing_type']}")
        print(f"💬 Message: {send_response['message']}")
        print(f"⏰ Timestamp: {send_response['timestamp']}")
    except httpx.HTTPStatusError as e:
        print(f"❌ Failed to send user message: {e.response.status_code}")
        print(f"   Response: {e.response.text}")
    print()


async def retrieve_instructions(client: ToMAgentClient, user_id: str) -> None:
    """Retrieve and display improved instructions."""
    print("📥 Step 2: Retrieving improved instructions...")
    try:
        instructions = await client.get_propose_instructions(user_id)
        if instructions["success"]:
            print(f"✅ Got {len(instructions['recommendations'])} instruction recommendations:")
            for i, rec in enumerate(instructions["recommendations"], 1):
                print(f"  {i}. Original: {rec['original_instruction']}")
                print(f"     Improved: {rec['improved_instruction']}")
                print(f"     Confidence: {rec['confidence_score']:.2f}")
                print(f"     Factors: {', '.join(rec['personalization_factors'])}")
            print(f"⏰ Calculated at: {instructions['calculated_at']}")
        else:
            print(f"⚠️  {instructions['message']}")
    except httpx.HTTPStatusError as e:
        print(f"❌ Failed to get instructions: {e.response.status_code}")
    print()


async def send_agent_response(client: ToMAgentClient, user_id: str) -> None:
    """Send an agent response message."""
    print("📤 Step 3: Sending agent response...")
    try:
        context = {"task_context": "Debugging IndexError in Python function"}
        agent_response = await client.send_message(
            user_id=user_id,
            message="I've analyzed the IndexError. The issue is likely with list bounds checking in your function. I've identified 3 potential locations where this could occur.",
            message_type="agent_response",
            context=context,
        )
        print("✅ Agent response sent successfully:")
        print(f"📋 Processing type: {agent_response['processing_type']}")
        print(f"💬 Message: {agent_response['message']}")
    except httpx.HTTPStatusError as e:
        print(f"❌ Failed to send agent response: {e.response.status_code}")
    print()


async def retrieve_next_actions(client: ToMAgentClient, user_id: str) -> None:
    """Retrieve and display next action suggestions."""
    print("📥 Step 4: Retrieving next action suggestions...")
    try:
        actions = await client.get_suggest_next_actions(user_id)
        if actions["success"]:
            print(f"✅ Got {len(actions['suggestions'])} action suggestions:")
            for i, suggestion in enumerate(actions["suggestions"], 1):
                print(
                    f"  {i}. [{suggestion['priority'].upper()}] {suggestion['action_description']}"
                )
                print(f"     Expected: {suggestion['expected_outcome']}")
                print(f"     Alignment: {suggestion['user_preference_alignment']:.2f}")
            print(f"📊 Based on: {actions['based_on_context']}")
            print(f"⏰ Calculated at: {actions['calculated_at']}")
        else:
            print(f"⚠️  {actions['message']}")
    except httpx.HTTPStatusError as e:
        print(f"❌ Failed to get next actions: {e.response.status_code}")
    print()


async def show_final_status(client: ToMAgentClient, user_id: str) -> None:
    """Show final conversation status and active conversations."""
    print("🔍 Final conversation status check...")
    try:
        final_status = await client.get_conversation_status(user_id)
        print("✅ Final status:")
        print(f"📝 Has pending instructions: {final_status['has_pending_instructions']}")
        print(f"🎯 Has pending next actions: {final_status['has_pending_next_actions']}")
        print(f"⏱️  Last activity: {final_status['last_activity']}")
    except httpx.HTTPStatusError as e:
        print(f"⚠️  Final status check failed: {e.response.status_code}")
    print()

    print("🗣️  Checking active conversations...")
    try:
        conversations = await client.get_active_conversations()
        print(f"✅ Total active conversations: {conversations['total_conversations']}")
        if conversations["user_ids"]:
            print(f"👥 User IDs: {', '.join(conversations['user_ids'])}")
    except httpx.HTTPStatusError as e:
        print(f"⚠️  Could not get active conversations: {e.response.status_code}")


async def main() -> None:
    """Example usage of the ToM Agent API client with the new flow."""
    client = ToMAgentClient()

    try:
        await check_api_health(client)

        # Use real user ID from the dataset
        user_id = "20d03f52-abb6-4414-b024-67cc89d53e12"

        await check_conversation_status(client, user_id)

        # Simulate the new bidirectional flow
        print("🔄 Testing new bidirectional communication flow...")
        print()

        await send_user_message(client, user_id)
        await retrieve_instructions(client, user_id)
        await send_agent_response(client, user_id)
        await retrieve_next_actions(client, user_id)
        await show_final_status(client, user_id)

    except httpx.ConnectError:
        print("❌ Could not connect to the API server.")
        print("   Make sure the server is running with: python -m tom_swe.api.main")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
