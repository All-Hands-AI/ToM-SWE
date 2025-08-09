"""
File location utilities for user modeling, following OpenHands patterns.
"""

from pathlib import Path

# Base directory
USER_MODEL_BASE_DIR = "~/.openhands/usermodeling"


def get_cleaned_sessions_dir(user_id: str | None = None) -> str:
    """Get the directory for cleaned sessions."""
    base = str(Path(USER_MODEL_BASE_DIR).expanduser())
    if user_id:
        return f"{base}/users/{user_id}/cleaned_sessions"
    else:
        return f"{base}/cleaned_sessions"


def get_cleaned_session_filename(sid: str, user_id: str | None = None) -> str:
    """Get the filename for a cleaned session."""
    return f"{get_cleaned_sessions_dir(user_id)}/{sid}.json"
