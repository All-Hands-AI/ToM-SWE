#!/usr/bin/env python3
"""
Example client for the ToM Agent REST API.

This script demonstrates how to interact with the ToM Agent API
endpoints for instruction improvement and next action suggestions.
"""

import asyncio
from typing import Any, Dict, cast

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

    async def propose_instructions(
        self,
        user_id: str,
        original_instruction: str,
        context: str,
    ) -> Dict[str, Any]:
        """Get improved, personalized instructions."""
        data = {
            "user_id": user_id,
            "original_instruction": original_instruction,
            "context": context,
        }
        response = await self.client.post(f"{self.base_url}/propose_instructions", json=data)
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

    async def suggest_next_actions(
        self,
        user_id: str,
        context: str,
    ) -> Dict[str, Any]:
        """Get personalized next action suggestions."""
        data = {
            "user_id": user_id,
            "context": context,
        }
        response = await self.client.post(f"{self.base_url}/suggest_next_actions", json=data)
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

    async def get_conversation_status(self, user_id: str) -> Dict[str, Any]:
        """Get the current conversation status for a user."""
        response = await self.client.get(
            f"{self.base_url}/conversation_status", params={"user_id": user_id}
        )
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())


async def check_api_health(client: ToMAgentClient) -> None:
    """Check API health and print status."""
    print("🔍 Checking API health...")
    health = await client.health_check()
    print(f"✅ API Status: {health['status']}")
    print(f"📊 ToM Agent Ready: {health['tom_agent_ready']}")
    print(f"🆔 API Version: {health.get('version', 'unknown')}")

    print()


async def check_conversation_status(client: ToMAgentClient, user_id: str) -> None:
    """Check and print conversation status for a user."""
    print(f"👤 Checking conversation status for user {user_id}...")
    try:
        status = await client.get_conversation_status(user_id)
        print("✅ Conversation status retrieved:")
        print(f"💬 Message: {status['message']}")
    except httpx.HTTPStatusError as e:
        print(f"⚠️  Status check failed: {e.response.status_code}")
    print()


async def get_improved_instructions(client: ToMAgentClient, user_id: str) -> None:
    """Get improved instructions for a user task."""
    print("📝 Step 1: Getting improved instructions...")
    try:
        context = ""

        instructions_response = await client.propose_instructions(
            user_id=user_id,
            original_instruction="Debug the function that's causing errors",
            context=context,
        )

        if instructions_response["success"]:
            print("✅ Improved instructions received:")
            print(f"📋 Original: {instructions_response['original_instruction']}")

            for i, rec in enumerate(instructions_response["recommendations"], 1):
                print(f"  {i}. Improved: {rec['improved_instruction']}")
        else:
            print(f"⚠️  {instructions_response['message']}")
    except httpx.HTTPStatusError as e:
        print(f"❌ Failed to get improved instructions: {e.response.status_code}")
        print(f"   Response: {e.response.text}")
    print()


async def get_next_actions(client: ToMAgentClient, user_id: str) -> None:
    """Get next action suggestions for a user."""
    print("🎯 Step 2: Getting next action suggestions...")
    try:
        context = "Agent response: I've analyzed the IndexError. The issue is likely with list bounds checking in your function. I've identified 3 potential locations where this could occur. Previous interactions show user prefers step-by-step debugging."

        actions_response = await client.suggest_next_actions(
            user_id=user_id,
            context=context,
        )

        if actions_response["success"]:
            print(f"✅ Got {len(actions_response['suggestions'])} action suggestions:")
            for i, suggestion in enumerate(actions_response["suggestions"], 1):
                print(
                    f"  {i}. [{suggestion['priority'].upper()}] {suggestion['action_description']}"
                )
                print(f"     Expected: {suggestion['expected_outcome']}")
                print(f"     Alignment: {suggestion['user_preference_alignment']:.2f}")
                print(f"     Reasoning: {suggestion['reasoning']}")
        else:
            print(f"⚠️  {actions_response['message']}")
    except httpx.HTTPStatusError as e:
        print(f"❌ Failed to get next actions: {e.response.status_code}")
    print()


async def show_final_status(client: ToMAgentClient, user_id: str) -> None:
    """Show final conversation status."""
    print("🔍 Final conversation status check...")
    try:
        final_status = await client.get_conversation_status(user_id)
        print("✅ Final status:")
        print(f"💬 Message: {final_status['message']}")
    except httpx.HTTPStatusError as e:
        print(f"⚠️  Final status check failed: {e.response.status_code}")
    print()


async def main() -> None:
    """Example usage of the ToM Agent API client with the new flow."""
    client = ToMAgentClient()

    try:
        await check_api_health(client)

        # Use real user ID from the dataset
        user_id = "20d03f52-abb6-4414-b024-67cc89d53e12"

        await check_conversation_status(client, user_id)

        # Test the direct API endpoints
        print("🔄 Testing ToM Agent API endpoints...")
        print()

        await get_improved_instructions(client, user_id)
        # await get_next_actions(client, user_id)
        # await show_final_status(client, user_id)

    except httpx.ConnectError:
        print("❌ Could not connect to the API server.")
        print("   Make sure the server is running with: python -m tom_swe.api.main")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
