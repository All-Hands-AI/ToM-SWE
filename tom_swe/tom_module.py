import asyncio
from datetime import datetime
from typing import Any, Dict, List

from tom_swe.generation.generate import LLMClient

# Third-party imports

from tom_swe.generation.dataclass import (
    UserAnalysis,
    SessionAnalysis,
    UserProfile,
    SessionAnalysisForLLM,
    UserAnalysisForLLM,
)
from tom_swe.prompts import PROMPTS


class ToMAnalyzer:
    def __init__(
        self,
        llm_client: LLMClient,
        session_batch_size: int = 3,
        user_id: str = "",
    ) -> None:
        """Initialize the analyzer with configuration and validate setup."""
        self.session_batch_size = session_batch_size
        self.user_id = user_id
        self.llm_client = llm_client

    async def analyze_session(self, session_data: Dict[str, Any]) -> SessionAnalysis:
        """
        Analyze a complete session and return a session summary.
        Uses important user messages as focus points with full session context.
        """
        session_id = session_data.get("session_id", "unknown")

        if not session_data or "messages" not in session_data:
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

        for message in session_data["messages"]:
            # Build full session context (all messages)
            role = message.get("source", "unknown")
            content = message.get("content", "")
            all_messages.append(f"{role}: {content}")

            # Filter for important user messages
            if message.get("source") == "user" and message.get("is_important", True):
                important_user_messages.append(content)

        # If no important messages marked, use all user messages
        if not important_user_messages:
            for message in session_data["messages"]:
                if message.get("source") == "user":
                    important_user_messages.append(message.get("content", ""))

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

        prompt = PROMPTS["session_analysis"].format(
            full_session_context=full_session_context,
            key_user_messages=key_user_messages,
            session_id=session_id,
            total_messages=len(session_data["messages"]),
            important_user_messages=len(important_user_messages),
        )
        result = await self.llm_client.call_structured_async(
            prompt=prompt,
            output_type=SessionAnalysisForLLM,
        )

        return SessionAnalysis(
            session_id=session_id,
            user_modeling_summary=result.user_modeling_summary,
            intent=result.intent,
            per_message_analysis=result.per_message_analysis,
            session_start=session_data.get("start_time") or "",
            session_end=session_data.get("end_time") or "",
            last_updated=datetime.now().isoformat(),
        )

    async def initialize_user_analysis(
        self, session_summaries: List[SessionAnalysis]
    ) -> UserAnalysis:
        """Initialize UserAnalysis from latest 50 session summaries using LLM."""
        # Take only the latest 50 sessions
        recent_sessions = (
            session_summaries[-30:]
            if len(session_summaries) > 30
            else session_summaries
        )
        # Create prompt with session data
        sessions_text = [s.model_dump() for s in recent_sessions]

        prompt = PROMPTS["user_analysis"].format(
            user_id=self.user_id,
            num_sessions=len(recent_sessions),
            sessions_text=sessions_text,
        )

        result = await self.llm_client.call_structured_async(
            prompt=prompt,
            output_type=UserAnalysisForLLM,
        )

        return UserAnalysis(
            user_profile=result.user_profile
            or UserProfile(
                user_id=self.user_id,
                overall_description=["User analysis unavailable"],
                preference_summary=[],
            ),
            session_summaries=result.session_summaries or [],
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
