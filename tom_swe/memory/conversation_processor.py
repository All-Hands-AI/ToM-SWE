"""
Simple low-level conversation processor with sleeptime computation.
"""

import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable, TypeVar
from .store import UserModelStore
from .locations import get_cleaned_session_filename
from datetime import datetime

T = TypeVar("T")


@dataclass
class CleanMessage:
    """Clean message object."""

    role: str
    content: str
    is_important: bool = False


@dataclass
class CleanSession:
    """Clean session object similar to FileUserModelStore format."""

    session_id: str
    start_time: str
    end_time: str
    messages: List[CleanMessage]
    user_id: str = ""
    last_updated: str = ""


def _clean_user_message(content: str) -> str:
    """Remove system tags and templates from user message."""
    patterns = [
        r"<REPOSITORY_INFO>.*?</REPOSITORY_INFO>",
        r"<RUNTIME_INFORMATION>.*?</RUNTIME_INFORMATION>",
        r"<EXTRA_INFO>.*?</EXTRA_INFO>",
        r"<ENVIRONMENT>.*?</ENVIRONMENT>",
        r"<CONTEXT>.*?</CONTEXT>",
        r"<system-reminder>.*?</system-reminder>",
        # Remove phase-based template that starts with "Follow these phases to resolve the issue:"
        r"Follow these phases to resolve the issue:.*?(?=\n\n(?![0-9]+\.|Phase [0-9]+\.)|$)",
    ]

    cleaned = content
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    return cleaned.strip()


def _is_important_user_message(original: str, cleaned: str) -> bool:
    """Check if user message is important based on rag_module logic."""
    if not cleaned.strip():
        return False

    if len(cleaned) < 10:  # Too short
        return False

    if len(cleaned) < len(original) * 0.3:  # Mostly system content
        return False

    token_count = len(cleaned) // 4  # Simple token estimation
    if token_count > 3000:  # Too long
        return False

    return True


def clean_sessions(
    sessions_data: List[Dict[str, Any]], file_store: Optional[Any] = None
) -> List["CleanSessionStore"]:
    """
    Process sessions_data to CleanSessionStore objects.

    Args:
        sessions_data: List of session data dictionaries
        file_store: Optional OpenHands FileStore object

    Returns:
        List of CleanSessionStore objects
    """
    clean_session_stores = []

    for session_data in sessions_data:
        session_id = session_data.get("session_id", "unknown")
        conversation_messages = session_data.get("conversation_messages", [])

        clean_messages = []

        for msg in conversation_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if not content.strip():
                continue

            clean_msg = CleanMessage(role=role, content=content)

            # Check if user message is important
            if role == "user":
                cleaned_content = _clean_user_message(content)
                clean_msg.is_important = _is_important_user_message(
                    content, cleaned_content
                )
            else:
                clean_msg.is_important = False

            clean_messages.append(clean_msg)

        clean_session = CleanSession(
            session_id=session_id,
            start_time=session_data.get("start_time", ""),
            end_time=session_data.get("end_time", ""),
            messages=clean_messages,
            last_updated=datetime.now().isoformat(),
        )

        # Create CleanSessionStore for this session
        store = CleanSessionStore(
            file_store=file_store,
            clean_session=clean_session,
        )

        # only keep sessions with at least 2 important messages
        if sum(msg.is_important for msg in clean_messages) >= 1:
            clean_session_stores.append(store)

    return clean_session_stores


@dataclass
class CleanSessionStore(UserModelStore):
    """Store for clean sessions, following OpenHands FileStore pattern."""

    file_store: Any
    clean_session: CleanSession

    async def save(self, user_id: str = "") -> None:
        """Save this clean session."""
        # Build filename with correct parameter order: sid first, then optional user_id
        filename = get_cleaned_session_filename(self.clean_session.session_id, user_id)
        session_json = json.dumps(asdict(self.clean_session), indent=2, default=str)

        await self._call_sync_from_async(self.file_store.write, filename, session_json)
        print(f"📁 Saved clean session: {filename}")

    async def save_model(self, user_id: str, model_data: Any) -> None:
        """Save model (for UserModelStore interface)."""
        await self.save(user_id)

    async def get_model(self, user_id: str) -> CleanSession:
        """Load this session."""
        return self.clean_session

    async def delete_model(self, user_id: str) -> None:
        """Delete this session."""
        pass  # Placeholder

    async def exists(self, user_id: str) -> bool:
        """Check if this session exists."""
        return False  # Placeholder

    async def _call_sync_from_async(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        """Call synchronous function from async context."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, func, *args, **kwargs
        )

    @classmethod
    async def get_instance(
        cls, config: Any, user_id: str | None
    ) -> "CleanSessionStore":
        """Get a clean session store instance."""
        file_store = getattr(config, "file_store", None) if config else None
        # Need a clean_session - this would be provided differently
        clean_session = CleanSession(
            session_id="", start_time="", end_time="", messages=[]
        )
        return cls(file_store=file_store, clean_session=clean_session)
