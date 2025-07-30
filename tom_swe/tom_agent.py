#!/usr/bin/env python3
"""
Theory of Mind (ToM) Agent for Personalized Instruction Generation

This module implements a ToM agent that combines user behavior analysis with RAG-based
context retrieval to propose improved instructions and next actions for individual users.
The agent uses the existing tom_module for user analysis and rag_module for context
retrieval to provide personalized guidance.

Key Features:
1. User context analysis using existing mental state models
2. RAG-based retrieval of relevant user behavior patterns
3. Personalized instruction generation based on user preferences
4. Next action recommendations tailored to user's mental state
5. Integration with existing ToM and RAG infrastructure
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional

# Third-party imports
import litellm
from dotenv import load_dotenv

# Local imports
from tom_swe.database import (
    InstructionImprovementResponse,
    InstructionRecommendation,
    UserContext,
)
from tom_swe.generation_utils.generate import (
    LLMConfig,
    call_llm_simple,
    call_llm_structured,
)
from tom_swe.rag_module import RAGAgent, create_rag_agent
from tom_swe.tom_module import UserMentalStateAnalyzer

# Load environment variables
load_dotenv()

# Configure logging - use environment variable for level
log_level = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

# Configure litellm
litellm.set_verbose = False

# LLM configuration
DEFAULT_LLM_MODEL = os.getenv(
    "DEFAULT_LLM_MODEL", "litellm_proxy/claude-sonnet-4-20250514"
)
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")

# Export list for better IDE support
__all__ = ["ToMAgent", "ToMAgentConfig", "create_tom_agent"]


@dataclass
class ToMAgentConfig:
    """Configuration for ToM Agent."""

    processed_data_dir: str = "./data/processed_data"
    user_model_dir: str = "./data/user_model"
    llm_model: Optional[str] = None
    enable_rag: bool = True


@dataclass
class InstructionScores:
    """Scores for instruction analysis."""

    confidence_score: float
    clarity_score: float


class ToMAgent:
    """
    Theory of Mind Agent for personalized instruction generation and action recommendations.

    This agent combines user mental state analysis with RAG-based context retrieval
    to provide highly personalized guidance for software engineering tasks.
    """

    def __init__(self, config: Optional[ToMAgentConfig] = None) -> None:
        """
        Initialize the ToM agent.

        Args:
            config: Configuration object for the ToM agent
        """
        if config is None:
            config = ToMAgentConfig()

        self.processed_data_dir = config.processed_data_dir
        self.user_model_dir = config.user_model_dir
        self.llm_model = config.llm_model or DEFAULT_LLM_MODEL
        self.enable_rag = config.enable_rag

        # Initialize ToM analyzer
        self.tom_analyzer = UserMentalStateAnalyzer(
            processed_data_dir=self.processed_data_dir, model=self.llm_model
        )

        # RAG agent will be initialized when needed (only if RAG is enabled)
        self._rag_agent: Optional[RAGAgent] = None

        rag_status = "enabled" if self.enable_rag else "disabled"
        logger.info(
            f"ToM Agent initialized with model: {self.llm_model}, RAG: {rag_status}"
        )

    async def _get_rag_agent(self) -> RAGAgent:
        """Get or initialize the RAG agent."""
        if self._rag_agent is None:
            logger.info("Initializing RAG agent...")
            self._rag_agent = await create_rag_agent(
                data_path=self.processed_data_dir,
                llm_model=self.llm_model,
            )
            logger.info("RAG agent initialized successfully")
        return self._rag_agent

    def _analyze_user_context(
        self, user_id: str, current_query: Optional[str] = None
    ) -> UserContext:
        """
        Analyze the current context for a user (synchronous internal method).

        Args:
            user_id: The user ID to analyze
            current_query: Optional current query/task the user is working on

        Returns:
            UserContext object with user's current state
        """
        logger.info(f"Analyzing user context for {user_id}")

        # Load user profile synchronously
        try:
            import asyncio

            # Run the async method in the current event loop if possible, otherwise create new one
            try:
                loop = asyncio.get_running_loop()
                # This is a temporary solution - ideally tom_analyzer should have sync methods
                user_analysis = asyncio.run_coroutine_threadsafe(
                    self.tom_analyzer.load_existing_user_profile(
                        user_id, self.user_model_dir
                    ),
                    loop,
                ).result()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                user_analysis = asyncio.run(
                    self.tom_analyzer.load_existing_user_profile(
                        user_id, self.user_model_dir
                    )
                )
        except Exception as e:
            logger.error(f"Error loading user profile: {e}")
            user_analysis = None

        user_profile = None
        recent_sessions: List[Any] = []
        preferences: List[str] = []
        mental_state_summary = ""

        if user_analysis:
            user_profile = user_analysis.user_profile
            recent_sessions = user_analysis.session_summaries[-5:]  # Last 5 sessions
            preferences = user_profile.preference_summary or []
            mental_state_summary = user_profile.overall_description or ""
        else:
            logger.warning(f"No user profile found for {user_id}")

        return UserContext(
            user_id=user_id,
            user_profile=user_profile,
            recent_sessions=recent_sessions,
            current_query=current_query,
            preferences=preferences,
            mental_state_summary=mental_state_summary,
        )

    async def _call_llm_structured(
        self,
        prompt: str,
        output_type: Any,
        temperature: float = 0.3,
    ) -> Optional[Any]:
        """Call the LLM with structured output using new generation utilities."""
        if not LITELLM_API_KEY:
            logger.error("LLM API key not configured")
            return None

        try:
            config = LLMConfig(
                model=self.llm_model,
                temperature=temperature,
                api_key=LITELLM_API_KEY,
                api_base=LITELLM_BASE_URL,
            )
            result = await call_llm_structured(
                prompt=prompt,
                output_type=output_type,
                config=config,
            )
            return result
        except Exception as e:
            logger.error(f"Error calling LLM with structured output: {e}")
            return None

    async def _call_llm_simple(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        """Call the LLM for simple text generation."""
        if not LITELLM_API_KEY:
            logger.error("LLM API key not configured")
            return None

        try:
            config = LLMConfig(
                model=self.llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=LITELLM_API_KEY,
                api_base=LITELLM_BASE_URL,
            )
            result = await call_llm_simple(
                prompt=prompt,
                config=config,
            )
            return result
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return None

    def propose_instructions(
        self,
        user_id: str,
        original_instruction: str,
        user_msg_context: Optional[str] = None,
    ) -> InstructionRecommendation:
        """
        Propose improved instructions (synchronous).

        Args:
            user_id: The user ID to analyze and generate instructions for
            original_instruction: The original instruction to improve
            user_msg_context: Optional user message context

        Returns:
            Instruction recommendation
        """
        logger.info(f"Proposing instructions for user {user_id}")

        # Load user context first (handle default_user case)
        if user_id == "default_user":
            # Create minimal user context for default user
            user_context = UserContext(user_id=user_id)
        else:
            # Load real user context for actual users
            user_context = self._analyze_user_context(user_id, original_instruction)

        # Get relevant user behavior from RAG (if enabled) - make this sync
        relevant_behavior = self._get_relevant_behavior_sync(original_instruction)

        # Build prompt for instruction improvement
        prompt = self._build_better_instruction_prompt(
            user_context, original_instruction, relevant_behavior, user_msg_context
        )

        # DEBUG: Check prompt length before LLM call
        self._debug_large_prompt(prompt, user_context, relevant_behavior)

        result = self._call_llm_structured_sync(
            prompt,
            output_type=InstructionImprovementResponse,
            temperature=0.2,
        )
        logger.info(f"🔍 PROPOSE_INSTRUCTIONS RESULT: {result}")
        if not result:
            # Return empty recommendation for failed calls
            return InstructionRecommendation(
                original_instruction=original_instruction,
                improved_instruction=original_instruction,
                reasoning="Failed to generate improvement",
                confidence_score=0.0,
                clarity_score=0.0,
            )

        # Post-process the instruction with formatted output
        scores = InstructionScores(
            confidence_score=result.confidence_score,
            clarity_score=result.clarity_score,
        )
        final_instruction = self._format_proposed_instruction(
            original_instruction=original_instruction,
            improved_instruction=result.improved_instruction,
            reasoning=result.reasoning,
            scores=scores,
        )

        return InstructionRecommendation(
            original_instruction=original_instruction,
            improved_instruction=final_instruction,
            reasoning=result.reasoning,
            confidence_score=result.confidence_score,
            clarity_score=result.clarity_score,
        )

    def _format_proposed_instruction(
        self,
        original_instruction: str,
        improved_instruction: str,
        reasoning: str,
        scores: InstructionScores,
    ) -> str:
        """
        Format the proposed instruction with clarity analysis and interpretation.

        Args:
            original_instruction: The user's original message
            improved_instruction: The AI's interpretation of what user meant
            reasoning: Reasoning for the interpretation
            scores: Scores containing confidence and clarity values

        Returns:
            Formatted instruction with analysis and clarification request
        """
        final_instruction = f"""The user's original message was: '{original_instruction}'
*****************ToM Agent Analysis Start Here*****************
(ToM agent reasoning is not from the actual user, but aims to help you better understand the user's intent)
The clarity score of the original message was: {scores.clarity_score*100:.0f}%, (here's the reasoning: {reasoning}).

Based on the conversation context and user patterns, here's a suggestion to help you better understand and help the user:

## Action Suggestions (IMPORTANT!)
{improved_instruction}

## Confidence in the suggestions
The ToM agent is {scores.confidence_score*100:.0f}% confident in the suggestions.
"""

        return final_instruction

    def _get_relevant_behavior_sync(self, original_instruction: str) -> str:
        """Get relevant user behavior from RAG if enabled (synchronous)."""
        if not self.enable_rag:
            return "RAG disabled - using user context only"

        rag_start_time = time.time()
        logger.info("⏱️  Starting RAG retrieval for instruction improvement...")

        rag_agent = self._get_rag_agent_sync()
        rag_init_time = time.time()
        logger.info(
            f"⏱️  RAG agent initialization: {rag_init_time - rag_start_time:.2f}s"
        )

        # Build direct query for user message search
        rag_query = original_instruction  # Search directly against user messages

        # Log the query content and token count
        query_tokens = len(rag_query.split()) * 1.3  # Rough estimate
        logger.info("🔍 RAG QUERY DEBUG:")
        logger.info(f"  - Query: '{rag_query}'")
        logger.info(f"  - Query length: {len(rag_query)} characters")
        logger.info(f"  - Estimated tokens: ~{query_tokens:.0f}")

        # Retrieve relevant user behavior (retrieval only, no generation)
        query_start_time = time.time()
        retrieved_docs = rag_agent.retrieve(rag_query, k=5)
        query_end_time = time.time()

        # Extract content from retrieved documents (now user messages with context)
        behavior_parts = []
        for i, doc in enumerate(retrieved_docs[:3]):  # Use top 3 for context
            # Format user message with surrounding context
            user_msg = doc.content
            metadata = doc.metadata

            # Build context string
            context_parts = [f"User msg: {user_msg}"]

            # Add session context if available
            if metadata.get("session_title"):
                context_parts.append(f"Session: {metadata['session_title']}")
            if metadata.get("repository_context"):
                context_parts.append(f"Project: {metadata['repository_context']}")

            # Add surrounding context (now a string)
            surrounding = metadata.get("surrounding_context", "")
            if surrounding:
                context_parts.append(f"Context: {surrounding}")

            behavior_parts.append(f"Context {i+1}:\n" + "\n".join(context_parts))

        relevant_behavior = (
            "\n\n".join(behavior_parts)
            if behavior_parts
            else "No relevant patterns found"
        )

        total_rag_time = query_end_time - rag_start_time
        query_time = query_end_time - query_start_time
        logger.info(f"⏱️  RAG query execution: {query_time:.2f}s")
        logger.info(f"⏱️  Total RAG time for instructions: {total_rag_time:.2f}s")

        return relevant_behavior

    def _get_rag_agent_sync(self) -> RAGAgent:
        """Get or initialize the RAG agent (synchronous)."""
        if self._rag_agent is None:
            logger.info("Initializing RAG agent...")
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                self._rag_agent = asyncio.run_coroutine_threadsafe(
                    create_rag_agent(
                        data_path=self.processed_data_dir,
                        llm_model=self.llm_model,
                    ),
                    loop,
                ).result()
            except RuntimeError:
                # No event loop running
                self._rag_agent = asyncio.run(
                    create_rag_agent(
                        data_path=self.processed_data_dir,
                        llm_model=self.llm_model,
                    )
                )
            logger.info("RAG agent initialized successfully")
        return self._rag_agent

    def _call_llm_structured_sync(
        self,
        prompt: str,
        output_type: Any,
        temperature: float = 0.3,
    ) -> Optional[Any]:
        """Call the LLM with structured output (synchronous)."""
        if not LITELLM_API_KEY:
            logger.error("LLM API key not configured")
            return None

        try:
            config = LLMConfig(
                model=self.llm_model,
                temperature=temperature,
                api_key=LITELLM_API_KEY,
                api_base=LITELLM_BASE_URL,
            )
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                result = asyncio.run_coroutine_threadsafe(
                    call_llm_structured(
                        prompt=prompt,
                        output_type=output_type,
                        config=config,
                    ),
                    loop,
                ).result()
            except RuntimeError:
                # No event loop running
                result = asyncio.run(
                    call_llm_structured(
                        prompt=prompt,
                        output_type=output_type,
                        config=config,
                    )
                )
            return result
        except Exception as e:
            logger.error(f"Error calling LLM with structured output: {e}")
            return None

    def _build_better_instruction_prompt(
        self,
        user_context: UserContext,
        original_instruction: str,
        relevant_behavior: str,
        user_msg_context: Optional[str],
    ) -> str:
        """Build the prompt for instruction improvement."""
        return f"""
Based on the following information, provide suggestions to help the agent better understand and help the user.

## Original Current User Instruction:
"{original_instruction}"

## Context
### User Context:
{user_context.model_dump_json(indent=2)}

### Current context (interactions happening before the original instruction, if any):
```
{user_msg_context}
```

### Past context--Relevant Past User Interaction with the Agent:
(Note the past context is automatically retrieved from the RAG agent. RAG agent could make mistakes, so use this with caution.)
```
{relevant_behavior}
```

Please generate suggestions to help the **agent** (note that you are giving suggestions to the agent, not the user) better understand and help the **user** (following the format below).
"""

    def _debug_large_prompt(
        self, prompt: str, user_context: UserContext, relevant_behavior: str
    ) -> None:
        """Debug large prompts by logging details and saving to file."""
        prompt_length = len(prompt)

        if prompt_length > 50000:  # If prompt is suspiciously large
            logger.warning(f"⚠️  LARGE PROMPT DETECTED: {prompt_length:,} characters")
            logger.info(
                f"  - Mental state summary length: {len(user_context.mental_state_summary or ''):,}"
            )
            logger.info(f"  - Preferences count: {len(user_context.preferences or [])}")
            logger.info(f"  - RAG behavior snippet: {relevant_behavior[:200]}...")

            # Save full prompt to file for inspection
            with open("/tmp/large_prompt_debug.txt", "w") as f:
                f.write(prompt)
            logger.info("  - Full prompt saved to /tmp/large_prompt_debug.txt")


# Convenience function for quick access
async def create_tom_agent(
    processed_data_dir: str = "./data/processed_data",
    user_model_dir: str = "./data/user_model",
    **kwargs: Any,
) -> ToMAgent:
    """
    Create and initialize a ToM agent.

    Args:
        processed_data_dir: Directory containing processed user data
        user_model_dir: Directory containing user model data
        **kwargs: Additional arguments for ToMAgent

    Returns:
        Initialized ToMAgent
    """
    agent = ToMAgent(
        config=ToMAgentConfig(
            processed_data_dir=processed_data_dir,
            user_model_dir=user_model_dir,
            **kwargs,
        )
    )
    return agent


# Example usage
def main() -> None:
    """Entry point for the tom-agent command."""

    async def async_main() -> None:
        # Create ToM agent
        agent = await create_tom_agent()

        # Test user and instruction
        user_id = "20d03f52-abb6-4414-b024-67cc89d53e12"
        instruction = "Hi hiiiiiii"

        print(f"Testing ToM Agent with user: {user_id}")
        print(f"Original instruction: '{instruction}'")
        print("=" * 60)

        # Test propose_instructions specifically
        print("\n1. Testing propose_instructions...")
        instruction_recommendation = agent.propose_instructions(
            user_id, instruction, user_msg_context=""
        )

        print("✓ Generated instruction recommendation")

        if instruction_recommendation:
            print("\nInstruction Improvement:")
            print(
                f"Improved instruction:\n{instruction_recommendation.improved_instruction}"
            )
        else:
            print("❌ No instruction recommendation generated")

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
