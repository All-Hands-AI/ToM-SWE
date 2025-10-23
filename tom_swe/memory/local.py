import logging
import os
import shutil
from typing import TypeVar

from .store import FileStore

try:
    from tom_swe.logging_config import get_tom_swe_logger

    logger = get_tom_swe_logger(__name__)
except ImportError:
    # Fallback for standalone use
    logger = logging.getLogger(__name__)

# Base directory for user model files
USER_MODEL_BASE_DIR = "data/user_model/user_model_overall"

T = TypeVar("T")


class LocalFileStore(FileStore):
    root: str

    def __init__(self, root: str):
        if root.startswith("~"):
            root = os.path.expanduser(root)
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def get_full_path(self, path: str) -> str:
        if path.startswith("/"):
            path = path[1:]
        return os.path.join(self.root, path)

    def write(self, path: str, contents: str | bytes) -> None:
        full_path = self.get_full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        mode = "w" if isinstance(contents, str) else "wb"
        with open(full_path, mode) as f:
            f.write(contents)

    def read(self, path: str) -> str:
        full_path = self.get_full_path(path)
        with open(full_path, "r") as f:
            return f.read()

    def list(self, path: str) -> list[str]:
        full_path = self.get_full_path(path)
        if not os.path.exists(full_path):
            return []

        # If path is a file, return the file itself (S3-consistent behavior)
        if os.path.isfile(full_path):
            return [path]

        # Otherwise it's a directory, return its contents
        files = [os.path.join(path, f) for f in os.listdir(full_path)]
        files = [f + "/" if os.path.isdir(self.get_full_path(f)) else f for f in files]
        return files

    def delete(self, path: str) -> None:
        try:
            full_path = self.get_full_path(path)
            if not os.path.exists(full_path):
                logger.debug(f"Local path does not exist: {full_path}")
                return
            if os.path.isfile(full_path):
                os.remove(full_path)
                logger.debug(f"Removed local file: {full_path}")
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
                logger.debug(f"Removed local directory: {full_path}")
        except Exception as e:
            logger.error(f"Error clearing local file store: {str(e)}")
