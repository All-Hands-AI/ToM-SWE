import asyncio
from datetime import datetime
from typing import Any, Dict, List

from tom_swe.generation.generate import LLMClient
import json

# Third-party imports

from tom_swe.generation.dataclass import (
    UserAnalysis,
    SessionAnalysis,
    UserProfile,
    SessionAnalysisForLLM,
    SessionSummary,
)
from tom_swe.memory.locations import get_overall_user_model_filename
from tom_swe.memory.store import FileStore
from tom_swe.prompts.manager import render_prompt

# Get logger that properly integrates with parent applications like OpenHands
try:
    from tom_swe.logging_config import get_tom_swe_logger, CLI_DISPLAY_LEVEL

    logger = get_tom_swe_logger(__name__)
except ImportError:
    # Fallback for standalone use
    import logging

    logger = logging.getLogger(__name__)
    CLI_DISPLAY_LEVEL = 25
    logging.addLevelName(CLI_DISPLAY_LEVEL, "CLI_DISPLAY")


class ToMAnalyzer:
    def __init__(
        self,
        llm_client: LLMClient,
        file_store: FileStore,
        session_batch_size: int = 3,
        user_id: str = "",
    ) -> None:
        """Initialize the analyzer with configuration and validate setup."""
        self.session_batch_size = session_batch_size
        self.user_id = user_id
        self.llm_client = llm_client
        self.file_store = file_store

    async def analyze_session(self, session_data: Dict[str, Any]) -> SessionAnalysis:
        """
        Analyze a complete session and return a session summary.
        Uses important user messages as focus points with full session context.
        """
        session_id = session_data.get("session_id", "unknown")
        logger.log(
            CLI_DISPLAY_LEVEL, f"🔍 Tom: Starting session analysis for {session_id}"
        )

        if not session_data or "messages" not in session_data:
            logger.log(
                CLI_DISPLAY_LEVEL, f"⚠️ Tom: No session data available for {session_id}"
            )
            return SessionAnalysis(
                session_id=session_id,
                intent="",
                per_message_analysis=[],
                user_modeling_summary="No session data available",
                session_start="",
                session_end="",
                last_updated=datetime.now().isoformat(),
            )

        # Extract important user messages and full session context
        important_user_messages = []
        all_messages = []

        logger.log(
            CLI_DISPLAY_LEVEL,
            f'📊 Tom: Processing {len(session_data["messages"])} messages',
        )

        for index, message in enumerate(session_data["messages"]):
            if index == 0:
                continue
            # Build full session context (all messages)
            role = message.get("role", "unknown")
            content = message.get("content", "")
            all_messages.append(f"{role}: {content}")

            # Filter for important user messages
            if role == "user" and message.get("is_important", True):
                important_user_messages.append(content)

        # If no important messages marked, use all user messages
        if not important_user_messages:
            logger.log(
                CLI_DISPLAY_LEVEL,
                "🔄 Tom: No important messages marked, using all user messages",
            )
            for message in session_data["messages"]:
                if message.get("source") == "user":
                    important_user_messages.append(message.get("content", ""))

        logger.log(
            CLI_DISPLAY_LEVEL,
            f"📝 Tom: Found {len(important_user_messages)} important user messages",
        )

        # Create comprehensive session context with truncation to fit context window
        def truncate_text_to_tokens(text: str, max_tokens: int = 50000) -> str:
            """Truncate text to approximately fit within token limit."""
            # Rough estimate: 1 token ≈ 4 characters for English text
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text

            # Truncate from the middle to keep both beginning and end context
            half_chars = max_chars // 2
            return (
                text[:half_chars]
                + f"\n\n[... TRUNCATED {len(text) - max_chars} characters ...]\n\n"
                + text[-half_chars:]
            )

        full_session_context = truncate_text_to_tokens(
            "\n".join(all_messages), max_tokens=100000
        )
        key_user_messages = truncate_text_to_tokens(
            "\n".join(important_user_messages), max_tokens=30000
        )

        logger.log(CLI_DISPLAY_LEVEL, "🤖 Tom: Sending session to LLM for analysis")
        logger.log(
            CLI_DISPLAY_LEVEL,
            f"📏 Tom: Full context: {len(full_session_context)} chars, Key messages: {len(key_user_messages)} chars",
        )

        prompt = render_prompt(
            "session_analysis",
            full_session_context=full_session_context,
            key_user_messages=key_user_messages,
            session_id=session_id,
            total_messages=len(session_data["messages"]),
            important_user_messages=len(important_user_messages),
        )

        logger.log(CLI_DISPLAY_LEVEL, "🔄 Tom: Calling LLM for structured analysis...")
        result = await self.llm_client.call_structured_async(
            prompt=prompt,
            output_type=SessionAnalysisForLLM,
        )
        logger.log(CLI_DISPLAY_LEVEL, "✅ Tom: LLM analysis completed")

        session_analysis = SessionAnalysis(
            session_id=session_id,
            user_modeling_summary=result.user_modeling_summary,
            intent=result.intent,
            per_message_analysis=result.per_message_analysis,
            session_start=session_data.get("start_time") or "",
            session_end=session_data.get("end_time") or "",
            session_tldr=result.session_tldr,
            last_updated=datetime.now().isoformat(),
        )

        logger.log(
            CLI_DISPLAY_LEVEL,
            f"📋 Tom: Session analysis complete - Intent: {result.intent[:100]}...",
        )
        logger.log(
            CLI_DISPLAY_LEVEL,
            f"👤 Tom: User modeling summary: {result.user_modeling_summary[:100]}...",
        )

        # Auto-update overall_user_model if it exists
        user_model_path = get_overall_user_model_filename(self.user_id)
        if self.file_store.list(user_model_path):
            logger.log(CLI_DISPLAY_LEVEL, "🔄 Tom: Auto-updating overall user model")
            await self._auto_update_user_model(session_analysis)
            logger.log(CLI_DISPLAY_LEVEL, "✅ Tom: User model updated")
        else:
            logger.log(CLI_DISPLAY_LEVEL, "📭 Tom: No existing user model to update")

        return session_analysis

    async def _auto_update_user_model(self, session_analysis: SessionAnalysis) -> None:
        """Auto-update the overall user model with new session information."""
        try:
            # Use LocalFileStore for file operations
            user_model_path = get_overall_user_model_filename(self.user_id)
            user_model_content = self.file_store.read(user_model_path)
            user_model = json.loads(user_model_content)

            # Add new session summary to the model
            new_session_summary = {
                "session_id": session_analysis.session_id,
                "session_tldr": session_analysis.session_tldr,
            }

            # Update session summaries (keep latest 50)
            user_model["session_summaries"] = user_model.get("session_summaries", [])
            user_model["session_summaries"].append(new_session_summary)
            if len(user_model["session_summaries"]) > 50:
                user_model["session_summaries"] = user_model["session_summaries"][-50:]

            # Update timestamp
            user_model["last_updated"] = datetime.now().isoformat()

            # Save updated model
            updated_content = json.dumps(user_model, indent=2)
            self.file_store.write(user_model_path, updated_content)

        except Exception as e:
            # Log error but don't fail the session analysis
            print(f"Warning: Failed to auto-update user model: {e}")

    async def initialize_user_analysis(
        self, session_summaries: List[SessionAnalysis]
    ) -> UserAnalysis:
        """Initialize UserAnalysis from latest 50 session summaries using LLM."""
        # Take only the latest 50 sessions
        # Create prompt with session data
        sessions_text = [s.model_dump() for s in session_summaries]

        prompt = render_prompt(
            "user_analysis",
            user_id=self.user_id,
            num_sessions=len(session_summaries),
            sessions_text=sessions_text,
        )

        result = await self.llm_client.call_structured_async(
            prompt=prompt,
            output_type=UserProfile,
        )

        return UserAnalysis(
            user_profile=result
            or UserProfile(
                user_id=self.user_id,
                overall_description=["User analysis unavailable"],
                preference_summary=[],
            ),
            session_summaries=[
                SessionSummary(
                    session_id=s.session_id,
                    session_tldr=s.session_tldr,
                )
                for s in session_summaries
            ],
            last_updated=datetime.now().isoformat(),
        )

    async def process_session_batch(
        self,
        session_batch: List[Dict[str, Any]],
    ) -> List[SessionAnalysis]:
        """
        Process a batch of sessions concurrently.
        Returns list of successfully processed session summaries.
        """
        # Process all sessions in the batch concurrently
        tasks = [self.analyze_session(session_data) for session_data in session_batch]
        results = await asyncio.gather(*tasks)

        return results
