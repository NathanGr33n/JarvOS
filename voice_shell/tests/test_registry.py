from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice_shell.src.actions.registry import ActionRegistry, ActionResult, _action_cat, _action_cd, _action_date, _action_launch_app, _action_ls, _action_time


class TestActionRegistry:
    """Tests for the ActionRegistry."""

    def test_default_registry_has_basic_actions(self):
        """Verify that the default registry contains basic shell actions."""
        registry = ActionRegistry()
        assert registry.is_allowed("ls")
        assert registry.is_allowed("cat")
        assert registry.is_allowed("cd")
        assert registry.is_allowed("date")
        assert registry.is_allowed("time")
        assert registry.is_allowed("app:firefox")

    def test_custom_registry_allows_only_specified(self):
        """Verify that a custom registry only allows specified commands."""
        registry = ActionRegistry(
            allowed_shell_commands=["ls"],
            allowed_apps=["code"],
        )
        assert registry.is_allowed("ls")
        assert registry.is_allowed("app:code")
        assert not registry.is_allowed("cat")
        assert not registry.is_allowed("app:firefox")

    def test_get_returns_callable(self):
        """Verify that get returns a callable for allowed actions."""
        registry = ActionRegistry()
        handler = registry.get("ls")
        assert callable(handler)

    def test_get_returns_none_for_unknown(self):
        """Verify that get returns None for disallowed actions."""
        registry = ActionRegistry(allowed_shell_commands=[])
        assert registry.get("ls") is None

    def test_list_actions_sorted(self):
        """Verify that list_actions returns a sorted list."""
        registry = ActionRegistry(
            allowed_shell_commands=["cat", "ls"],
            allowed_apps=["firefox"],
        )
        actions = registry.list_actions()
        assert actions == ["app:firefox", "cat", "ls"]

    def test_list_actions_empty(self):
        """Verify that an empty registry returns an empty list."""
        registry = ActionRegistry(allowed_shell_commands=[], allowed_apps=[])
        assert registry.list_actions() == []


class TestBuiltInActions:
    """Tests for the built-in action functions."""

    def test_action_ls(self):
        """Verify that ls executes and returns a non-empty result."""
        result = _action_ls()
        assert result.error is None
        assert result.returncode == 0
        assert len(result.stdout) > 0

    def test_action_cat_existing_file(self, tmp_path: Path):
        """Verify that cat reads an existing file correctly."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        result = _action_cat(str(test_file))
        assert result.error is None
        assert result.stdout == "hello world"

    def test_action_cat_missing_file(self):
        """Verify that cat returns an error for a missing file."""
        result = _action_cat("/nonexistent/path/file.txt")
        assert result.error is not None
        assert "File not found" in result.error

    def test_action_cd_valid(self, tmp_path: Path):
        """Verify that cd changes to a valid directory."""
        original = Path.cwd()
        result = _action_cd(str(tmp_path))
        assert result.error is None
        assert result.stdout == str(tmp_path)
        # Restore
        import os
        os.chdir(original)

    def test_action_cd_invalid(self):
        """Verify that cd returns an error for an invalid directory."""
        result = _action_cd("/nonexistent/directory/12345")
        assert result.error is not None
        assert "Path not found" in result.error

    def test_action_time(self):
        """Verify that time returns a formatted string."""
        result = _action_time()
        assert result.error is None
        assert len(result.stdout) > 0
        # Should be in 12-hour format like "12:34 PM"
        assert "M" in result.stdout  # AM or PM

    def test_action_date(self):
        """Verify that date returns a formatted string."""
        result = _action_date()
        assert result.error is None
        assert len(result.stdout) > 0
        # Should contain the current year
        import datetime
        assert str(datetime.datetime.now().year) in result.stdout

    def test_action_launch_app(self):
        """Verify that app launch returns an error for a non-existent app."""
        result = _action_launch_app("nonexistent_application_12345")
        assert result.error is not None
        assert "Cannot launch" in result.error
