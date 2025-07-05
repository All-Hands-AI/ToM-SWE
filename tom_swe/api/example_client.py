#!/usr/bin/env python3
"""
Example client for the ToM Agent REST API.

This script demonstrates how to interact with the ToM Agent API endpoints.
"""

import asyncio
import json
from typing import Dict, Any

import httpx


class ToMAgentClient:
    """Client for interacting with the ToM Agent API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client with the API base URL."""
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the API server."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def suggest_next_actions(
        self, user_id: str, current_task_context: str = None
    ) -> Dict[str, Any]:
        """Get next action suggestions for a user."""
        data = {"user_id": user_id}
        if current_task_context:
            data["current_task_context"] = current_task_context
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/suggest_next_actions",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    async def propose_instructions(
        self, user_id: str, original_instruction: str, domain_context: str = None
    ) -> Dict[str, Any]:
        """Get improved instruction proposals for a user."""
        data = {
            "user_id": user_id,
            "original_instruction": original_instruction
        }
        if domain_context:
            data["domain_context"] = domain_context
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/propose_instructions",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    async def send_message(
        self,
        user_id: str,
        message: str,
        instruction: str = None,
        current_task: str = None,
        domain_context: str = None
    ) -> Dict[str, Any]:
        """Send a message to the ToM agent and get comprehensive guidance."""
        data = {
            "user_id": user_id,
            "message": message
        }
        if instruction:
            data["instruction"] = instruction
        if current_task:
            data["current_task"] = current_task
        if domain_context:
            data["domain_context"] = domain_context
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/send_message",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get the current context for a user."""
        response = await self.client.get(
            f"{self.base_url}/api/v1/users/{user_id}/context"
        )
        response.raise_for_status()
        return response.json()


async def main():
    """Example usage of the ToM Agent API client."""
    client = ToMAgentClient()
    
    try:
        print("🔍 Checking API health...")
        health = await client.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"📊 ToM Agent Ready: {health['tom_agent_ready']}")
        print()
        
        user_id = "example_user_123"
        
        print(f"👤 Getting user context for {user_id}...")
        try:
            user_context = await client.get_user_context(user_id)
            print(f"✅ User context retrieved successfully")
            print(f"📝 User ID: {user_context['user_id']}")
        except httpx.HTTPStatusError as e:
            print(f"⚠️  User context not available: {e.response.status_code}")
        print()
        
        print("🎯 Suggesting next actions...")
        try:
            actions = await client.suggest_next_actions(
                user_id=user_id,
                current_task_context="Debugging Python application"
            )
            print(f"✅ Got {len(actions['suggestions'])} action suggestions:")
            for i, suggestion in enumerate(actions['suggestions'], 1):
                print(f"  {i}. [{suggestion['priority'].upper()}] {suggestion['action_description']}")
        except httpx.HTTPStatusError as e:
            print(f"⚠️  Next actions not available: {e.response.status_code}")
        print()
        
        print("📝 Proposing improved instructions...")
        try:
            instructions = await client.propose_instructions(
                user_id=user_id,
                original_instruction="Debug the function that's causing errors",
                domain_context="Python web development"
            )
            print(f"✅ Got {len(instructions['recommendations'])} instruction recommendations:")
            for i, rec in enumerate(instructions['recommendations'], 1):
                print(f"  {i}. Original: {rec['original_instruction']}")
                print(f"     Improved: {rec['improved_instruction']}")
                print(f"     Confidence: {rec['confidence_score']:.2f}")
        except httpx.HTTPStatusError as e:
            print(f"⚠️  Instruction proposals not available: {e.response.status_code}")
        print()
        
        print("💬 Sending comprehensive message...")
        try:
            guidance = await client.send_message(
                user_id=user_id,
                message="I need help debugging my Python web application",
                instruction="Debug the function that's causing errors",
                current_task="Debugging Python web application",
                domain_context="Web development"
            )
            print(f"✅ Got comprehensive guidance:")
            print(f"📊 Overall confidence: {guidance['guidance']['confidence_score']:.2f}")
            print(f"💡 Overall guidance: {guidance['guidance']['overall_guidance']}")
            print(f"📝 Instruction recommendations: {len(guidance['guidance']['instruction_recommendations'])}")
            print(f"🎯 Next action suggestions: {len(guidance['guidance']['next_action_suggestions'])}")
        except httpx.HTTPStatusError as e:
            print(f"⚠️  Comprehensive guidance not available: {e.response.status_code}")
        
    except httpx.ConnectError:
        print("❌ Could not connect to the API server.")
        print("   Make sure the server is running with: tom-api")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())