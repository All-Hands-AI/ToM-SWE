"""
File-based implementation of UserModelStore following OpenHands FileConversationStore pattern.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from .store import UserModelStore

logger = logging.getLogger(__name__)

# Base directory for user model files
USER_MODEL_BASE_DIR = "data/user_model/user_model_overall"

T = TypeVar("T")


def get_user_model_filename(user_id: str) -> str:
    """Get the filename for a user's model file."""
    return f"{USER_MODEL_BASE_DIR}/{user_id}.json"


@dataclass
class FileUserModelStore(UserModelStore):
    """File-based user model storage following OpenHands pattern."""

    file_store: Any  # FileStore - will be injected

    async def save_model(self, user_id: str, model_data: Any) -> None:
        """Store user model to file."""
        json_str = json.dumps(model_data, indent=2, default=str)
        path = self.get_user_model_filename(user_id)
        # Using call_sync_from_async pattern like OpenHands
        await self._call_sync_from_async(self.file_store.write, path, json_str)

    async def get_model(self, user_id: str) -> Any:
        """Load user model from file."""
        path = self.get_user_model_filename(user_id)
        json_str = await self._call_sync_from_async(self.file_store.read, path)

        # Validate the JSON
        json_obj = json.loads(json_str)
        if "user_profile" not in json_obj:
            raise FileNotFoundError(f"Invalid user model format: {path}")

        return json_obj

    async def delete_model(self, user_id: str) -> None:
        """Delete user model file."""
        path = self.get_user_model_filename(user_id)
        await self._call_sync_from_async(self.file_store.delete, path)

    async def exists(self, user_id: str) -> bool:
        """Check if user model file exists."""
        path = self.get_user_model_filename(user_id)
        try:
            await self._call_sync_from_async(self.file_store.read, path)
            return True
        except FileNotFoundError:
            return False

    def get_user_model_dir(self) -> str:
        """Get the directory containing user model files."""
        return USER_MODEL_BASE_DIR

    def get_user_model_filename(self, user_id: str) -> str:
        """Get the filename for a user's model file."""
        return get_user_model_filename(user_id)

    async def _call_sync_from_async(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        """Call synchronous function from async context - placeholder for now."""
        # TODO: Import actual call_sync_from_async from OpenHands
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, func, *args, **kwargs
        )

    @classmethod
    async def get_instance(
        cls, config: Any, user_id: str | None
    ) -> "FileUserModelStore":
        """Get a file-based user model store instance."""
        # TODO: Use actual get_file_store from OpenHands when available
        # For now, create a simple file store placeholder
        file_store = SimpleFileStore()
        return FileUserModelStore(file_store)


class SimpleFileStore:
    """Simple file store implementation as placeholder."""

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

    def delete(self, path: str) -> None:
        """Delete file."""
        file_path = Path(path)
        if file_path.exists():
            file_path.unlink()
