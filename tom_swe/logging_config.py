"""
Logging configuration for tom-swe integration with OpenHands.

This module provides proper logger configuration for tom-swe when used
as a dependency in OpenHands or other applications.
"""

import logging

# Custom log level for CLI display messages
CLI_DISPLAY_LEVEL = 25
logging.addLevelName(CLI_DISPLAY_LEVEL, "CLI_DISPLAY")


def configure_tom_swe_logging(parent_logger_name: str = "openhands") -> None:
    """
    Configure tom-swe loggers to inherit from the specified parent logger.

    This ensures that tom-swe logging output appears correctly when used
    as a dependency in applications like OpenHands.

    Args:
        parent_logger_name: Name of the parent logger to inherit from (default: "openhands")
    """
    # List of all tom-swe modules that use logging
    tom_swe_modules = [
        "tom_swe.api.main",
        "tom_swe.memory.local",
        "tom_swe.tom_agent",
        "tom_swe.generation.generate",
        "tom_swe.generation.action",
        "tom_swe.utils",
        "tom_swe.generation.output_parsers",
        "tom_swe.rag_module",
    ]

    # Get the parent logger
    parent_logger = logging.getLogger(parent_logger_name)

    for module_name in tom_swe_modules:
        module_logger = logging.getLogger(module_name)

        # Clear any existing handlers
        module_logger.handlers.clear()

        # Set the parent
        module_logger.parent = parent_logger

        # Ensure propagation is enabled so messages flow to parent
        module_logger.propagate = True

        # Don't set a specific level - inherit from parent
        module_logger.setLevel(logging.NOTSET)


def get_tom_swe_logger(
    name: str, parent_logger_name: str = "openhands"
) -> logging.Logger:
    """
    Get a properly configured logger for tom-swe modules.

    Args:
        name: The logger name (typically __name__)
        parent_logger_name: Name of the parent logger to inherit from

    Returns:
        Configured logger instance
    """
    # Create logger as child of parent
    if name.startswith("tom_swe."):
        logger_name = name
    else:
        logger_name = f"tom_swe.{name}"

    logger = logging.getLogger(logger_name)

    # Set parent relationship
    parent_logger = logging.getLogger(parent_logger_name)
    logger.parent = parent_logger
    logger.propagate = True
    logger.setLevel(logging.NOTSET)

    return logger


def is_openhands_logging_available() -> bool:
    """
    Check if OpenHands logging is available and configured.

    Returns:
        True if OpenHands logger exists and has handlers
    """
    try:
        openhands_logger = logging.getLogger("openhands")
        return bool(openhands_logger.handlers)
    except Exception:
        return False


def auto_configure_logging() -> None:
    """
    Automatically configure tom-swe logging based on available parent loggers.

    This function attempts to detect the environment and configure logging accordingly:
    - If OpenHands logger is available, use it as parent
    - Otherwise, fall back to root logger
    """
    if is_openhands_logging_available():
        configure_tom_swe_logging("openhands")
    else:
        # Fall back to ensuring root logger has basic configuration
        if not logging.root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s:%(levelname)s - %(message)s",
            )
