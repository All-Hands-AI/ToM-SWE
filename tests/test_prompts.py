"""
Tests for the Jinja2-based prompt template system.
"""

import pytest
from tom_swe.prompts.manager import PromptManager, render_prompt


class TestPromptManager:
    """Test cases for PromptManager class."""

    def setup_method(self) -> None:
        """Setup method called before each test."""
        # Use the default templates directory
        self.manager = PromptManager()

    def test_manager_initialization(self) -> None:
        """Test that PromptManager initializes correctly."""
        assert self.manager is not None
        assert self.manager.templates_dir.exists()
        assert self.manager.env is not None

    def test_list_templates(self) -> None:
        """Test that list_templates returns expected templates."""
        templates = self.manager.list_templates()

        expected_templates = [
            "message_condensation",
            "propose_instructions",
            "session_analysis",
            "sleeptime_compute",
            "user_analysis",
        ]

        for template in expected_templates:
            assert template in templates

    def test_render_session_analysis(self) -> None:
        """Test rendering session_analysis template."""
        result = self.manager.render(
            "session_analysis",
            full_session_context="Test session context",
            key_user_messages="Test user messages",
            session_id="test-123",
            total_messages=5,
            important_user_messages=2,
        )

        assert "Test session context" in result
        assert "Test user messages" in result
        assert "test-123" in result
        assert "5" in result
        assert "2" in result
        assert "Analyze this coding session" in result

    def test_render_user_analysis(self) -> None:
        """Test rendering user_analysis template."""
        result = self.manager.render(
            "user_analysis",
            user_id="user-456",
            num_sessions=3,
            sessions_text=["session1", "session2", "session3"],
        )

        assert "user-456" in result
        assert "3 sessions" in result
        assert "session1" in result
        assert "Analyze these recent coding sessions" in result

    def test_render_message_condensation(self) -> None:
        """Test rendering message_condensation template."""
        result = self.manager.render(
            "message_condensation",
            max_tokens=100,
            content="This is a long message that needs to be condensed for better processing.",
        )

        assert "100 tokens" in result
        assert "This is a long message" in result
        assert "Please condense" in result

    def test_render_static_templates(self) -> None:
        """Test rendering templates that don't require variables."""
        # Test sleeptime_compute
        sleeptime_result = self.manager.render("sleeptime_compute")
        assert "user modeling expert" in sleeptime_result
        assert "UPDATE_JSON_FIELD" in sleeptime_result

        # Test propose_instructions
        propose_result = self.manager.render("propose_instructions")
        assert "ToM (theory of mind) Agent" in propose_result
        assert "expert in modeling user's mental state" in propose_result

    def test_render_nonexistent_template(self) -> None:
        """Test that rendering non-existent template raises error."""
        with pytest.raises(RuntimeError, match="Failed to render template"):
            self.manager.render("nonexistent_template")

    def test_render_from_string(self) -> None:
        """Test rendering template from string."""
        template_string = "Hello {{ name }}, you have {{ count }} messages."
        result = self.manager.render_from_string(template_string, name="Alice", count=5)

        assert result == "Hello Alice, you have 5 messages."

    def test_custom_filters(self) -> None:
        """Test that custom filters are available."""
        # Test length filter
        template_string = "You have {{ items | length }} items."
        result = self.manager.render_from_string(template_string, items=["a", "b", "c"])

        assert result == "You have 3 items."


class TestRenderPromptFunction:
    """Test cases for the convenience render_prompt function."""

    def test_render_prompt_function(self) -> None:
        """Test that render_prompt function works correctly."""
        result = render_prompt(
            "session_analysis",
            full_session_context="Function test context",
            key_user_messages="Function test messages",
            session_id="func-test-123",
            total_messages=10,
            important_user_messages=4,
        )

        assert "Function test context" in result
        assert "func-test-123" in result
        assert "10" in result

    def test_render_prompt_with_missing_variables(self) -> None:
        """Test rendering with missing required variables."""
        # This should still work, rendering variables as empty strings
        result = render_prompt("session_analysis")

        # Jinja2 renders missing variables as empty strings
        assert "Session ID: " in result  # The label is there but variable is empty
        assert "Total messages: " in result  # Same for other variables


class TestTemplateConsistency:
    """Test template content consistency."""

    def test_templates_contain_expected_content(self) -> None:
        """Test that templates contain expected content."""
        # Test session_analysis contains expected sections
        result = render_prompt(
            "session_analysis",
            full_session_context="test context",
            key_user_messages="test messages",
            session_id="test-id",
            total_messages=5,
            important_user_messages=2,
        )

        # Verify expected content is present
        assert "test context" in result
        assert "test messages" in result
        assert "test-id" in result
        assert "5" in result
        assert "2" in result
        assert "Full Session Context:" in result
        assert "Key User Messages" in result
        assert "Session Metadata:" in result


if __name__ == "__main__":
    pytest.main([__file__])
