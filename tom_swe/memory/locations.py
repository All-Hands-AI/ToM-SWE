"""
File location utilities for user modeling, following OpenHands patterns.
"""

from pathlib import Path

# Base directory
USER_MODEL_BASE_DIR = "usermodeling"


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


def get_session_models_dir(user_id: str | None = None) -> str:
    """Get the directory for session models."""
    base = str(Path(USER_MODEL_BASE_DIR).expanduser())
    if user_id:
        return f"{base}/users/{user_id}/session_models"
    else:
        return f"{base}/session_models"


def get_session_model_filename(sid: str, user_id: str | None = None) -> str:
    """Get the filename for a session model."""
    return f"{get_session_models_dir(user_id)}/{sid}.json"


def get_overall_user_model_filename(user_id: str | None = None) -> str:
    """Get the filename for the overall user model."""
    base = str(Path(USER_MODEL_BASE_DIR).expanduser())
    if user_id:
        return f"{base}/users/{user_id}/overall_user_model.json"
    else:
        return f"{base}/overall_user_model.json"


def get_usermodeling_dir(user_id: str | None = None) -> str:
    """Get the directory for the overall user model."""
    base = str(Path(USER_MODEL_BASE_DIR).expanduser())
    if user_id:
        return f"{base}/users/{user_id}"
    else:
        return f"{base}"
