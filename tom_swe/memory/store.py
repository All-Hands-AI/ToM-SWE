"""
User Model Store interface following OpenHands ConversationStore pattern.
"""

from abc import ABC, abstractmethod
from typing import Any

from .locations import get_overall_user_model_filename


class FileStore:
    @abstractmethod
    def write(self, path: str, contents: str | bytes) -> None:
        pass

    @abstractmethod
    def read(self, path: str) -> str:
        pass

    @abstractmethod
    def list(self, path: str) -> list[str]:
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        pass


class UserModelStore(ABC):
    """Abstract base class for user model storage.

    This is an extension point in ToM-SWE that allows applications to customize how
    user models are stored. Applications can substitute their own implementation by:
    1. Creating a class that inherits from UserModelStore
    2. Implementing all required methods
    3. Setting the configuration to use the fully qualified name of the class

    The class is instantiated via get_instance().
    """

    @abstractmethod
    async def save_model(self, user_id: str, model_data: Any) -> None:
        """Store user model."""
        pass

    @abstractmethod
    async def get_model(self, user_id: str) -> Any:
        """Load user model."""
        pass

    async def validate_model(self, user_id: str, requesting_user_id: str) -> bool:
        """Validate that model belongs to the current user."""
        # Default implementation - users can only access their own model
        return user_id == requesting_user_id

    @abstractmethod
    async def delete_model(self, user_id: str) -> None:
        """Delete user model."""
        pass

    @classmethod
    @abstractmethod
    async def get_instance(cls, config: Any, user_id: str | None) -> "UserModelStore":
        """Get a store for the user represented by the configuration."""
        pass


def load_user_model(user_id: str, file_store: FileStore) -> str:
    """Load user model from file store."""
    user_model_filename = get_overall_user_model_filename(user_id)
    if file_store.list(user_model_filename):
        user_model = file_store.read(user_model_filename)
    else:
        user_model = "none"
    return user_model
