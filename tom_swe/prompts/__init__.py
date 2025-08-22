"""
Centralized prompt management for ToM Agent workflows.

This module provides a single source of truth for all system prompts
used across different ToM Agent workflows.
"""

from .registry import PROMPTS

__all__ = ["PROMPTS"]
