"""
Simple low-level conversation processor with sleeptime computation.
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, TypeVar
from .store import UserModelStore
from .locations import get_cleaned_session_filename


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
    messages: List[CleanMessage]
    user_id: str = ""


def _clean_user_message(content: str) -> str:
    """Remove system tags from user message."""
    patterns = [
        r"<REPOSITORY_INFO>.*?</REPOSITORY_INFO>",
        r"<RUNTIME_INFORMATION>.*?</RUNTIME_INFORMATION>",
        r"<EXTRA_INFO>.*?</EXTRA_INFO>",
        r"<ENVIRONMENT>.*?</ENVIRONMENT>",
        r"<CONTEXT>.*?</CONTEXT>",
        r"<system-reminder>.*?</system-reminder>",
    ]

    cleaned = content
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    return cleaned.strip()


def _is_important_user_message(original: str, cleaned: str) -> bool:
    """Check if user message is important based on rag_module logic."""
    if not cleaned.strip():
        return False

    if len(cleaned) < 25:  # Too short
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

            clean_messages.append(clean_msg)

        clean_session = CleanSession(
            session_id=session_id,
            messages=clean_messages,
        )

        # Create CleanSessionStore for this session
        store = CleanSessionStore(
            file_store=file_store or LocalFileStore(), clean_session=clean_session
        )
        clean_session_stores.append(store)

    return clean_session_stores


class LocalFileStore:
    """Local FileStore implementation as fallback when OpenHands FileStore not available."""

    def write(self, path: str, content: str) -> None:
        """Write content to file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)

    def read(self, path: str) -> str:
        """Read content from file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(file_path, "r") as f:
            return f.read()


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
        if not file_store:
            file_store = LocalFileStore()
        # Need a clean_session - this would be provided differently
        clean_session = CleanSession("", [])
        return cls(file_store=file_store, clean_session=clean_session)


def sleeptime_compute(
    sessions_data: List[Dict[str, Any]],
    user_id: str = "",
    file_store: Optional[Any] = None,
) -> List[CleanSessionStore]:
    """
    Process sessions, automatically save them, and return CleanSessionStore objects.

    Args:
        sessions_data: Raw session data to process
        user_id: User identifier
        file_store: OpenHands FileStore object (optional)

    Returns:
        List of CleanSessionStore objects (already saved)
    """
    import asyncio

    clean_session_stores = clean_sessions(sessions_data, file_store)

    # Automatically save all sessions concurrently
    async def _save_all() -> None:
        await asyncio.gather(*(store.save(user_id) for store in clean_session_stores))

    asyncio.run(_save_all())

    return clean_session_stores
