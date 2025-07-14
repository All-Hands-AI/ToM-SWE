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
from .database import (
    InstructionImprovementResponse,
    InstructionRecommendation,
    NextActionsResponse,
    NextActionSuggestion,
    PersonalizedGuidance,
    UserContext,
)
from .generation_utils.generate import LLMConfig, call_llm_simple, call_llm_structured
from .rag_module import RAGAgent, create_rag_agent
from .tom_module import UserMentalStateAnalyzer

# Load environment variables
load_dotenv()

# Configure logging - use environment variable for level
log_level = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

# Configure litellm
litellm.set_verbose = False

# LLM configuration
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "litellm_proxy/claude-sonnet-4-20250514")
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
        logger.info(f"ToM Agent initialized with model: {self.llm_model}, RAG: {rag_status}")

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

    async def analyze_user_context(
        self, user_id: str, current_query: Optional[str] = None
    ) -> UserContext:
        """
        Analyze the current context for a user.

        Args:
            user_id: The user ID to analyze
            current_query: Optional current query/task the user is working on

        Returns:
            UserContext object with user's current state
        """
        logger.info(f"Analyzing user context for {user_id}")

        # Load user profile
        user_analysis = await self.tom_analyzer.load_existing_user_profile(
            user_id, self.user_model_dir
        )

        user_profile = None
        recent_sessions = []
        preferences = []
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

    async def propose_instructions(
        self,
        user_context: UserContext,
        original_instruction: str,
        user_msg_context: Optional[str] = None,
    ) -> List[InstructionRecommendation]:
        """
        Propose improved instructions based on user context.

        Args:
            user_context: The user's current context
            original_instruction: The original instruction to improve
            user_msg_context: Optional user message context

        Returns:
            List of instruction recommendations
        """
        logger.info(f"Proposing instructions for user {user_context.user_id}")

        # Get relevant user behavior from RAG (if enabled)
        relevant_behavior = await self._get_relevant_behavior(original_instruction)

        # Build prompt for instruction improvement
        prompt = self._build_instruction_prompt(
            user_context, original_instruction, relevant_behavior, user_msg_context
        )

        # DEBUG: Check prompt length before LLM call
        self._debug_large_prompt(prompt, user_context, relevant_behavior)

        result = await self._call_llm_structured(
            prompt,
            output_type=InstructionImprovementResponse,
            temperature=0.2,
        )
        logger.info(f"🔍 PROPOSE_INSTRUCTIONS RESULT: {result}")
        if not result:
            return []

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

        return [
            InstructionRecommendation(
                original_instruction=original_instruction,
                improved_instruction=final_instruction,
                reasoning=result.reasoning,
                confidence_score=result.confidence_score,
                clarity_score=result.clarity_score,
            )
        ]

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

Based on the conversation context and user patterns, here's a clearer interpretation of what the user likely means:

## Improved Instruction
{improved_instruction}

## Confidence in interpretation
Confidence in interpretation: {scores.confidence_score*100:.0f}%

## YOU MUST FOLLOW THIS INSTRUCTION
**IMPORTANT** (you HAVE to follow this instruction): First show user the improved instruction (just copy the improved instruction section above) and ask if they would like to continue with the plan (e.g., 'Would you like to continue with the plan? Or do you have any other plans in mind?'). And offer to explore the current environment a bit more first (e.g., 'Would you like to explore the current environment a bit more first?')"""

        return final_instruction

    async def _get_relevant_behavior(self, original_instruction: str) -> str:
        """Get relevant user behavior from RAG if enabled."""
        if not self.enable_rag:
            return "RAG disabled - using user context only"

        rag_start_time = time.time()
        logger.info("⏱️  Starting RAG retrieval for instruction improvement...")

        rag_agent = await self._get_rag_agent()
        rag_init_time = time.time()
        logger.info(f"⏱️  RAG agent initialization: {rag_init_time - rag_start_time:.2f}s")

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

            # Add surrounding context
            surrounding = metadata.get("surrounding_context", {})
            if surrounding.get("prev_agent_msg"):
                context_parts.append(f"Previous agent: {surrounding['prev_agent_msg']}")
            if surrounding.get("next_agent_msg"):
                context_parts.append(f"Next agent: {surrounding['next_agent_msg']}")
            if surrounding.get("prev_user_msg"):
                context_parts.append(f"Previous user: {surrounding['prev_user_msg']}")
            if surrounding.get("next_user_msg"):
                context_parts.append(f"Next user: {surrounding['next_user_msg']}")

            behavior_parts.append(f"Context {i+1}:\n" + "\n".join(context_parts))

        relevant_behavior = (
            "\n\n".join(behavior_parts) if behavior_parts else "No relevant patterns found"
        )

        total_rag_time = query_end_time - rag_start_time
        query_time = query_end_time - query_start_time
        logger.info(f"⏱️  RAG query execution: {query_time:.2f}s")
        logger.info(f"⏱️  Total RAG time for instructions: {total_rag_time:.2f}s")

        return relevant_behavior

    def _build_instruction_prompt(
        self,
        user_context: UserContext,
        original_instruction: str,
        relevant_behavior: str,
        user_msg_context: Optional[str],
    ) -> str:
        """Build the prompt for instruction improvement."""
        return f"""
Based on the following user context and behavior patterns, propose a plan to help the user achieve their goal (use markdown bullet points format).

User Context:
- User ID: {user_context.user_id}
- Mental State Summary: {user_context.mental_state_summary}
- Preferences: {', '.join(user_context.preferences or [])}
- Recent Session Count: {len(user_context.recent_sessions or [])}

Relevant User Behavior Patterns:
{relevant_behavior}

Original Instruction:
"{original_instruction}"

Context (interactions happening before the original instruction, if any):
{user_msg_context}

Please generate the plan that you think the user was actually trying to achieve (following the format below).
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

    async def suggest_next_actions(
        self, user_context: UserContext, current_task_context: Optional[str] = None
    ) -> List[NextActionSuggestion]:
        """
        Suggest next actions based on user context and current task.

        Args:
            user_context: The user's current context
            current_task_context: Optional context about the current task

        Returns:
            List of next action suggestions
        """
        logger.info(f"Suggesting next actions for user {user_context.user_id}")

        # Get relevant patterns from RAG (if enabled)
        workflow_patterns = ""
        if self.enable_rag:
            rag_start_time = time.time()
            logger.info("⏱️  Starting RAG retrieval for next actions...")

            rag_agent = await self._get_rag_agent()
            rag_init_time = time.time()
            logger.info(f"⏱️  RAG agent initialization: {rag_init_time - rag_start_time:.2f}s")

            rag_query = "User workflow patterns and next action preferences"
            if current_task_context:
                rag_query += f" for tasks like: {current_task_context}"

            query_start_time = time.time()
            retrieved_docs = rag_agent.retrieve(rag_query, k=3)
            query_end_time = time.time()

            # Extract workflow patterns from retrieved documents
            pattern_parts = []
            for i, doc in enumerate(retrieved_docs[:3]):  # Use all 3 for workflow patterns
                pattern_parts.append(
                    f"Pattern {i+1}: {doc.content[:400]}..."
                )  # Shorter for workflow patterns

            workflow_patterns = (
                "\n\n".join(pattern_parts) if pattern_parts else "No workflow patterns found"
            )

            total_rag_time = query_end_time - rag_start_time
            query_time = query_end_time - query_start_time
            logger.info(f"⏱️  RAG query execution: {query_time:.2f}s")
            logger.info(f"⏱️  Total RAG time for next actions: {total_rag_time:.2f}s")
        else:
            workflow_patterns = "RAG disabled - using user context only"

        # Analyze recent session patterns
        recent_intents: List[str] = []
        recent_emotions: List[str] = []
        recent_sessions = user_context.recent_sessions or []
        for session in recent_sessions[-3:]:  # Last 3 sessions
            recent_intents.extend(session.intent_distribution.keys())
            recent_emotions.extend(session.emotion_distribution.keys())

        prompt = f"""
Based on the user's context and behavior patterns, suggest 3-5 next actions they should consider.
Focus on actions that align with their preferences and current mental state.

User Context:
- Mental State: {user_context.mental_state_summary}
- Preferences: {', '.join(user_context.preferences or [])}
- Recent Intents: {', '.join(set(recent_intents))}
- Recent Emotions: {', '.join(set(recent_emotions))}

Workflow Patterns:
{workflow_patterns}

Current Task Context:
{current_task_context or "General development work"}

For each suggested action, provide:
1. Clear action description
2. Priority level (high/medium/low)
3. Reasoning for the suggestion
4. Expected outcome
5. Alignment with user preferences (0-1 score)
"""

        # DEBUG: Check prompt length before LLM call
        prompt_length = len(prompt)
        logger.info("🔍 SUGGEST_NEXT_ACTIONS DEBUG:")
        logger.info(f"  - Prompt length: {prompt_length:,} characters")
        logger.info(f"  - Estimated tokens: ~{prompt_length // 4:,} tokens")
        logger.info(f"  - Workflow patterns length: {len(workflow_patterns):,} characters")
        logger.info(
            f"  - Mental state summary length: {len(user_context.mental_state_summary or ''):,}"
        )

        if prompt_length > 50000:  # If prompt is suspiciously large
            logger.warning(f"⚠️  LARGE PROMPT DETECTED: {prompt_length:,} characters")
            logger.info(f"  - Recent intents: {len(recent_intents)}")
            logger.info(f"  - Recent emotions: {len(recent_emotions)}")
            logger.info(f"  - Workflow patterns snippet: {workflow_patterns[:200]}...")

            # Save full prompt to file for inspection
            with open("/tmp/large_prompt_next_actions_debug.txt", "w") as f:
                f.write(prompt)
            logger.info("  - Full prompt saved to /tmp/large_prompt_next_actions_debug.txt")

        result = await self._call_llm_structured(
            prompt,
            output_type=NextActionsResponse,
            temperature=0.3,
        )
        if not result:
            return []

        suggestions = []
        for llm_suggestion in result.suggestions:
            suggestions.append(
                NextActionSuggestion(
                    action_description=llm_suggestion.action_description,
                    priority=llm_suggestion.priority,
                    reasoning=llm_suggestion.reasoning,
                    expected_outcome=llm_suggestion.expected_outcome,
                    user_preference_alignment=llm_suggestion.user_preference_alignment,
                )
            )
        return suggestions

    async def get_personalized_guidance(
        self,
        user_id: str,
        instruction: Optional[str] = None,
        current_task: Optional[str] = None,
        user_msg_context: Optional[str] = None,
    ) -> PersonalizedGuidance:
        """
        Get comprehensive personalized guidance for a user.

        Args:
            user_id: The user ID
            instruction: Optional instruction to improve
            current_task: Optional current task context
            user_msg_context: Optional user message context

        Returns:
            PersonalizedGuidance object with complete recommendations
        """
        logger.info(f"Generating personalized guidance for user {user_id}")

        # Analyze user context
        user_context = await self.analyze_user_context(user_id, current_task)

        # Generate instruction recommendations if provided
        instruction_recommendations = []
        if instruction:
            instruction_recommendations = await self.propose_instructions(
                user_context, instruction, user_msg_context
            )

        # Generate next action suggestions
        next_action_suggestions = await self.suggest_next_actions(user_context, current_task)

        # Generate overall guidance
        overall_guidance = await self._generate_overall_guidance(
            user_context, instruction_recommendations, next_action_suggestions
        )

        # Calculate overall confidence
        confidence_scores = []
        if instruction_recommendations:
            confidence_scores.extend([rec.confidence_score for rec in instruction_recommendations])
        if next_action_suggestions:
            confidence_scores.extend(
                [sug.user_preference_alignment for sug in next_action_suggestions]
            )

        overall_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        )

        return PersonalizedGuidance(
            user_context=user_context,
            instruction_recommendations=instruction_recommendations,
            next_action_suggestions=next_action_suggestions,
            overall_guidance=overall_guidance,
            confidence_score=overall_confidence,
        )

    async def _generate_overall_guidance(
        self,
        user_context: UserContext,
        instruction_recommendations: List[InstructionRecommendation],
        next_action_suggestions: List[NextActionSuggestion],
    ) -> str:
        """Generate overall guidance summary."""

        prompt = f"""
Based on the user's context and the generated recommendations, provide a concise overall guidance summary.
This should be a friendly, personalized message that ties together the recommendations.

User Mental State: {user_context.mental_state_summary}
User Preferences: {', '.join(user_context.preferences or [])}

Instruction Improvements: {len(instruction_recommendations)} recommendations
Next Actions: {len(next_action_suggestions)} suggestions

Provide a 2-3 sentence summary that:
1. Acknowledges the user's current state/preferences
2. Highlights the key recommendations
3. Provides encouragement aligned with their mental state
"""

        response = await self._call_llm_simple(prompt, temperature=0.4, max_tokens=150)
        return (
            response
            or "Based on your preferences and working style, I've provided personalized recommendations to help you work more effectively."
        )


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
            processed_data_dir=processed_data_dir, user_model_dir=user_model_dir, **kwargs
        )
    )
    return agent


# Example usage
if __name__ == "__main__":

    async def main() -> None:
        # Create ToM agent
        agent = await create_tom_agent()

        # Example usage
        user_id = "example_user_123"
        instruction = "Debug the function that's causing errors"

        # Get personalized guidance
        guidance = await agent.get_personalized_guidance(
            user_id=user_id,
            instruction=instruction,
            current_task="Debugging Python application",
            user_msg_context="Web development",
        )

        print(f"Personalized guidance for user {user_id}:")
        print(f"Overall guidance: {guidance.overall_guidance}")
        print(f"Confidence score: {guidance.confidence_score:.2f}")

        if guidance.instruction_recommendations:
            print("\nInstruction improvements:")
            for rec in guidance.instruction_recommendations:
                print(f"- {rec.improved_instruction}")
                print(f"  Reasoning: {rec.reasoning}")

        if guidance.next_action_suggestions:
            print("\nNext action suggestions:")
            for sug in guidance.next_action_suggestions:
                print(f"- [{sug.priority.upper()}] {sug.action_description}")
                print(f"  Expected outcome: {sug.expected_outcome}")

    asyncio.run(main())
