"""Data utilities for ToM-SWE."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Dictionary containing the loaded data.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return {}
    
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return {}


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """Save data to a JSON file.
    
    Args:
        data: Data to save.
        file_path: Path to the JSON file.
        
    Returns:
        True if successful, False otherwise.
    """
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    os.makedirs(file_path.parent, exist_ok=True)
    
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Data saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def get_user_data_files(data_dir: Union[str, Path] = "data") -> List[Path]:
    """Get all user data files in the data directory.
    
    Args:
        data_dir: Path to the data directory.
        
    Returns:
        List of paths to user data files.
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return []
    
    return list(data_dir.glob("user_*.json"))


def get_analysis_files(data_dir: Union[str, Path] = "data") -> List[Path]:
    """Get all analysis files in the data directory.
    
    Args:
        data_dir: Path to the data directory.
        
    Returns:
        List of paths to analysis files.
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return []
    
    return list(data_dir.glob("analysis_*.json"))


def extract_user_id(file_path: Union[str, Path]) -> str:
    """Extract user ID from a file path.
    
    Args:
        file_path: Path to a user data or analysis file.
        
    Returns:
        User ID.
    """
    file_path = Path(file_path)
    file_name = file_path.stem
    
    if file_name.startswith("user_"):
        return file_name[5:]
    elif file_name.startswith("analysis_"):
        return file_name[9:]
    else:
        return ""