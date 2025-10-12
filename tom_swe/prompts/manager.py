"""
Template manager for ToM-SWE prompts using Jinja2.

This module provides a centralized way to manage and render prompt templates
using Jinja2 templating engine, replacing the previous string.format() approach.
"""

from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader


class PromptManager:
    """Manager for Jinja2-based prompt templates."""

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the prompt manager.

        Args:
            templates_dir: Directory containing template files.
                          Defaults to 'templates' in the same directory as this file.
        """
        if templates_dir is None:
            self.templates_dir = Path(__file__).parent / "templates"
        else:
            self.templates_dir = Path(templates_dir)

        # Create Jinja2 environment with FileSystemLoader
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            # Enable auto-escaping for safety (though we're not using HTML)
            autoescape=False,
            # Strip whitespace for cleaner prompts
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add custom filters if needed
        self.env.filters["length"] = len

    def render(self, template_name: str, **kwargs: Any) -> str:
        """
        Render a template with the given variables.

        Args:
            template_name: Name of the template file (without .jinja2 extension)
            **kwargs: Variables to pass to the template

        Returns:
            Rendered template as a string

        Raises:
            FileNotFoundError: If template file doesn't exist
            jinja2.TemplateError: If there's an error in template rendering
        """
        template_file = f"{template_name}.jinja2"

        try:
            template = self.env.get_template(template_file)
            result: str = template.render(**kwargs)
            return result
        except Exception as e:
            raise RuntimeError(
                f"Failed to render template '{template_file}': {e}"
            ) from e

    def render_from_string(self, template_string: str, **kwargs: Any) -> str:
        """
        Render a template from a string (for backward compatibility).

        Args:
            template_string: Template content as a string
            **kwargs: Variables to pass to the template

        Returns:
            Rendered template as a string
        """
        template = self.env.from_string(template_string)
        result: str = template.render(**kwargs)
        return result

    def list_templates(self) -> list[str]:
        """
        List all available template files.

        Returns:
            List of template names (without .jinja2 extension)
        """
        if not self.templates_dir.exists():
            return []

        templates = []
        for file_path in self.templates_dir.glob("*.jinja2"):
            templates.append(file_path.stem)

        return sorted(templates)


# Global instance for easy access
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def render_prompt(template_name: str, **kwargs: Any) -> str:
    """
    Convenience function to render a prompt template.

    Args:
        template_name: Name of the template file (without .jinja2 extension)
        **kwargs: Variables to pass to the template

    Returns:
        Rendered template as a string
    """
    return get_prompt_manager().render(template_name, **kwargs)
