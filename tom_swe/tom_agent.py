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

import logging
import os
import json
import time
import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional, Dict

# Third-party imports
import litellm
from dotenv import load_dotenv

# Local imports
from tom_swe.generation.dataclass import (
    InstructionImprovement,
    AnalyzeSessionParams,
    InitializeUserProfileParams,
)
from tom_swe.generation import (
    LLMConfig,
    LLMClient,
    ActionType,
    ActionResponse,
    ActionExecutor,
)
from tom_swe.prompts import PROMPTS
from tom_swe.rag_module import RAGAgent, create_rag_agent
from tom_swe.tom_module import ToMAnalyzer
from tom_swe.memory.conversation_processor import clean_sessions
from tom_swe.memory.local import LocalFileStore
from tom_swe.memory.locations import (
    get_cleaned_session_filename,
    get_cleaned_sessions_dir,
    get_overall_user_model_filename,
    get_session_model_filename,
)
from tom_swe.memory.store import FileStore, load_user_model
from tom_swe.utils import format_proposed_instruction

# Load environment variables
load_dotenv()

# Get logger that properly integrates with parent applications like OpenHands
try:
    from tom_swe.logging_config import get_tom_swe_logger, CLI_DISPLAY_LEVEL

    logger = get_tom_swe_logger(__name__)
except ImportError:
    # Fallback for standalone use
    logger = logging.getLogger(__name__)
    CLI_DISPLAY_LEVEL = 25
    logging.addLevelName(CLI_DISPLAY_LEVEL, "CLI_DISPLAY")

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

    file_store: Optional[FileStore] = None
    llm_model: Optional[str] = None
    enable_rag: bool = False
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    skip_memory_collection: bool = (
        False  # If True, skip workflow and directly predict user mental states
    )


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

        self.llm_model = config.llm_model or DEFAULT_LLM_MODEL
        self.enable_rag = config.enable_rag
        self.skip_memory_collection = config.skip_memory_collection

        # LLM configuration - use config values if provided, otherwise fallback to env vars
        self.api_key = config.api_key or LITELLM_API_KEY
        self.api_base = config.api_base or LITELLM_BASE_URL
        self.file_store = config.file_store or LocalFileStore(root="~/.openhands")

        # Create LLM client with our configuration
        llm_config = LLMConfig(
            model=self.llm_model,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        self.llm_client = LLMClient(llm_config)

        # Initialize ToM analyzer with LLM client and FileStore
        self.tom_analyzer = ToMAnalyzer(
            llm_client=self.llm_client,
            user_id="",  # Default to empty string as per the new API
        )

        # RAG agent will be initialized when needed (only if RAG is enabled)
        self._rag_agent: Optional[RAGAgent] = None

        # Initialize action executor with reference to this agent
        self.action_executor = ActionExecutor(
            user_id="", agent_context=self, file_store=self.file_store
        )

        rag_status = "enabled" if self.enable_rag else "disabled"
        logger.info(
            f"ToM Agent initialized with model: {self.llm_model}, RAG: {rag_status}"
        )

    def propose_instructions(
        self,
        user_id: str | None = "",
        original_instruction: str = "",
        formatted_messages: List[Dict[str, Any]] | None = None,
    ) -> InstructionImprovement | None:
        """
        Propose improved instructions using workflow controller.

        Args:
            user_id: The user ID to analyze and generate instructions for
            formatted_messages: List of formatted message dicts with cache support

        Returns:
            Instruction recommendation
        """
        # return InstructionImprovement(
        #     original_instruction=original_instruction,
        #     improved_instruction="test tom improve instruction",
        #     confidence_score=0.9,
        # )
        logger.info(f"🎯 Proposing instructions for user {user_id}")
        if user_id is None:
            user_id = ""
        assert isinstance(
            user_id, str
        ), f"user_id must be a string, got {type(user_id)}"

        # Update the action executor's user_id
        self.action_executor.user_id = user_id
        # Build comprehensive prompt for instruction improvement
        user_model = load_user_model(user_id, self.file_store)

        # Ensure formatted_messages is not None
        if formatted_messages is None:
            formatted_messages = []
        # Early stop: Quick clarity assessment for caching optimization
        logger.info("🔍 Performing early clarity assessment")
        propose_instructions_messages: List[Dict[str, Any]] = (
            [
                {
                    "role": "system",
                    "content": PROMPTS["propose_instructions"],
                    "cache_control": {"type": "ephemeral"},  # Cache the system prompt
                }
            ]
            + [
                {
                    "role": "user",
                    "content": f"Here is the content of the overall_user_model (`overall_user_model.json`): {user_model}\n Below is the agent-user interaction context:\n-------------context start-------------",
                }
            ]
            + formatted_messages
            + [
                {
                    "role": "user",
                    "content": f"-------------context end-------------\n Here is the my instruction for improvement: {original_instruction}\n Now I want you to role-play as me and improve the instruction.",
                }
            ]
        )
        assert (
            propose_instructions_messages[-1]["role"] == "user"
        ), "Last message must be a user message"

        try:
            # clarity_result = self.llm_client.call_structured_messages(
            #     messages=propose_instructions_messages,
            #     output_type=ClarityAssessment,
            #     temperature=0.1,
            # )
            clarity_result = None
            # Early stop if clarity is sufficient
            if clarity_result and clarity_result.is_clear:
                logger.info(
                    f"✅ Early stop: Intent is clear - {clarity_result.reasoning}"
                )
                # Return minimal improvement for clear instructions
                return None
            else:
                # propose_instructions_messages.append(
                #     {
                #         "role": "assistant",
                #         "content": f"Clarity assessment: {clarity_result.reasoning} (The instruction is clear: {clarity_result.is_clear})",
                #     }
                # )
                result = self._step(
                    messages=propose_instructions_messages,
                )
        except Exception as e:
            logger.warning(f"Clarity assessment failed: {e}, exit")
            return None

        # Post-process the instruction with formatted output
        final_instruction = format_proposed_instruction(
            original_instruction=original_instruction,
            improved_instruction=result.improved_instruction,
            confidence_score=result.confidence_score,
        )

        return InstructionImprovement(
            original_instruction=original_instruction,
            improved_instruction=final_instruction,
            confidence_score=result.confidence_score,
        )

    def _get_relevant_behavior_sync(self, original_instruction: str) -> str:
        """Get relevant user behavior from RAG if enabled (synchronous)."""
        if not self.enable_rag:
            return "RAG disabled - using user context only"

        rag_start_time = time.time()
        logger.info("⏱️  Starting RAG retrieval for instruction improvement...")

        rag_agent = self._get_rag_module_sync()
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

    def _get_rag_module_sync(self) -> RAGAgent:
        """Get or initialize the RAG agent (synchronous)."""
        if self._rag_agent is None:
            logger.info("Initializing RAG agent...")
            try:
                loop = asyncio.get_running_loop()
                self._rag_agent = asyncio.run_coroutine_threadsafe(
                    create_rag_agent(
                        llm_model=self.llm_model,
                    ),
                    loop,
                ).result()
            except RuntimeError:
                # No event loop running
                self._rag_agent = asyncio.run(
                    create_rag_agent(
                        llm_model=self.llm_model,
                    )
                )
            logger.info("RAG agent initialized successfully")
        return self._rag_agent

    def _step(
        self,
        messages: List[Dict[str, Any]],
        response_model: type = ActionResponse,
        max_iterations: int = 3,
        preset_actions: Optional[List[Any]] = None,
    ) -> Any:
        """
        Generic workflow controller for action-based workflows.

        Args:
            messages: The formatted messages list
            response_model: Pydantic model for workflow steps (default: ActionResponse)
            max_iterations: Maximum workflow iterations
            preset_actions: Optional list of pre-set actions to execute before LLM decisions

        Returns:
            Result from the final action execution
        """

        logger.info(f"🤖 Starting workflow with {max_iterations} max iterations")
        logger.debug(f"Initial messages: {messages}")

        for iteration in range(max_iterations):
            logger.info(f"🔄 Workflow iteration {iteration + 1}/{max_iterations}")

            # Use preset actions first, then fall back to LLM
            if preset_actions:
                response = preset_actions.pop(0)  # Take first preset action
                logger.info(f"🎯 Using preset action: {response.action.value}")
            else:
                # Structured call with Pydantic model for intermediate steps
                response = self.llm_client.call_structured_messages(
                    messages=messages, output_type=response_model, temperature=0.1
                )

            if not response:
                logger.error("❌ No response from LLM")
                break

            # Type-safe access to structured response
            logger.log(CLI_DISPLAY_LEVEL, f"🧠 Agent reasoning: {response.reasoning}")
            logger.log(CLI_DISPLAY_LEVEL, f"⚡ Agent action: {response.action.value}")
            logger.log(CLI_DISPLAY_LEVEL, f"🔍 Action parameters: {response.parameters}")

            # Execute the action using ActionExecutor
            result = self.action_executor.execute_action(
                response.action, response.parameters
            )
            logger.log(CLI_DISPLAY_LEVEL, f"🔍 Action result: {result}")

            # Update conversation
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": f"Action: {response.action.value}, Reasoning: {response.reasoning}",
                    },
                    {"role": "user", "content": f"Tool result: {result}"},
                ]
            )

            # Check if workflow is complete
            if response.is_complete:
                logger.info("✅ Workflow completed successfully")
                break

        # Return the result from final action execution
        return result

    def sleeptime_compute(
        self,
        sessions_data: List[dict[str, Any]],
        user_id: str | None = "",
    ) -> None:
        """
        Process sessions through the three-tier memory system using workflow controller.

        Args:
            sessions_data: Raw session data to process
            user_id: User identifier
            file_store: OpenHands FileStore object (optional)
        """
        logger.info(
            f"🔄 Starting sleeptime_compute workflow for {len(sessions_data)} sessions"
        )
        if user_id is None:
            user_id = ""
        assert isinstance(
            user_id, str
        ), f"user_id must be a string, got {type(user_id)}"
        # Step 1: Pre-process sessions to get cleaned session files
        clean_session_stores = clean_sessions(sessions_data, self.file_store)

        # Save all cleaned sessions and collect their file paths
        async def _save_all(user_id: str) -> List[str]:
            await asyncio.gather(
                *(store.save(user_id) for store in clean_session_stores)
            )
            # Return list of file paths where sessions were saved
            return [
                get_cleaned_session_filename(store.clean_session.session_id, user_id)
                for store in clean_session_stores
            ]

        # Since this method is called from an async context, we can await directly
        cleaned_file_paths = asyncio.run(_save_all(user_id))
        logger.info(f"📁 Cleaned sessions saved to: {cleaned_file_paths}")

        # Step 2: Find unprocessed sessions (exist in cleaned but not in session models)
        try:
            cleaned_session_ids = [
                file_path.split("/")[-1].replace(".json", "")
                for file_path in self.file_store.list(get_cleaned_sessions_dir(user_id))
            ]
            # remove empty strings
            cleaned_session_ids = [x for x in cleaned_session_ids if x]

            # rank the cleaned_session_ids; the last id is the most recent session
            def get_last_updated(session_id: str) -> str:
                session_data = json.loads(
                    self.file_store.read(
                        get_cleaned_session_filename(session_id, user_id)
                    )
                )
                return str(session_data.get("last_updated", ""))

            cleaned_session_ids = sorted(cleaned_session_ids, key=get_last_updated)
        except FileNotFoundError:
            # No cleaned sessions directory exists yet
            cleaned_session_ids = []
            return
        # Find sessions that don't have corresponding model files
        unprocessed_sessions = [
            cleaned_session_ids[-1]
        ]  # always process the most recent session as we don't know whether such session is updated.
        for session_id in cleaned_session_ids[:-1]:
            model_file = get_session_model_filename(session_id, user_id)
            if not self.file_store.exists(model_file):
                unprocessed_sessions.append(session_id)
        logger.info(f"🔍 Unprocessed sessions: {unprocessed_sessions}")

        preset_actions = []
        if unprocessed_sessions:
            # Step 3: Create preset action for batch processing unprocessed sessions
            preset_actions += [
                ActionResponse(
                    action=ActionType.ANALYZE_SESSION,
                    parameters=AnalyzeSessionParams(
                        user_id=user_id,
                        session_batch=unprocessed_sessions,
                    ),
                    reasoning=f"Pre-configured batch processing of {len(unprocessed_sessions)} unprocessed sessions",
                    is_complete=False,
                )
            ]

        if (
            not self.file_store.exists(get_overall_user_model_filename(user_id))
            and unprocessed_sessions
        ):
            preset_actions += [
                ActionResponse(
                    action=ActionType.INITIALIZE_USER_PROFILE,
                    parameters=InitializeUserProfileParams(
                        user_id=user_id,
                    ),
                    reasoning="Pre-configured saving of updated user profile",
                    is_complete=True,
                ),
            ]

        user_model = load_user_model(user_id, self.file_store)
        # Step 4: Use workflow controller with preset actions
        messages = [
            {"role": "system", "content": PROMPTS["sleeptime_compute"]},
            {
                "role": "user",
                "content": f"Here is the content of the user model (`overall_user_model.json`): {user_model}\nI have {len(unprocessed_sessions)} unprocessed session files that need batch processing:\nSession IDs to process: {unprocessed_sessions}.",
            },
        ]
        final_result = self._step(
            messages=messages,
            preset_actions=preset_actions,
            max_iterations=3,
        )
        logger.info(f"🔄 Final result: {final_result}")


# Convenience function for quick access
def create_tom_agent(
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    enable_rag: bool = False,
    file_store: Optional[FileStore] = None,
    skip_memory_collection: bool = False,
) -> ToMAgent:
    """
    Create and initialize a ToM agent.

    Args:
        processed_data_dir: Directory containing processed user data
        user_model_dir: Directory containing user model data
        llm_model: LLM model to use (defaults to DEFAULT_LLM_MODEL)
        api_key: API key for LLM service (defaults to LITELLM_API_KEY env var)
        api_base: Base URL for LLM service (defaults to LITELLM_BASE_URL env var)
        enable_rag: Whether to enable RAG functionality
        **kwargs: Additional arguments for ToMAgent

    Returns:
        Initialized ToMAgent
    """
    config = ToMAgentConfig(
        file_store=file_store,
        llm_model=llm_model,
        api_key=api_key,
        api_base=api_base,
        enable_rag=enable_rag,
        skip_memory_collection=skip_memory_collection,
    )
    agent = ToMAgent(config=config)
    return agent
