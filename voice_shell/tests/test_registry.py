from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice_shell.src.actions.registry import ActionRegistry, ActionResult, _action_cat, _action_cd, _action_date, _action_get_battery_status, _action_launch_app, _action_ls, _action_search_files, _action_time


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
        assert "Path not found" in result.error

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


class TestPhase2Actions:
    """Tests for Phase 2 registry expansions."""

    def test_default_registry_has_phase2_actions(self):
        registry = ActionRegistry()
        assert registry.is_allowed("list_directory")
        assert registry.is_allowed("read_file")
        assert registry.is_allowed("search_files")
        assert registry.is_allowed("get_battery_status")

    def test_list_directory_alias(self, tmp_path: Path):
        registry = ActionRegistry()
        handler = registry.get("list_directory")
        assert handler is not None
        result = handler(str(tmp_path))
        assert result.error is None
        assert result.returncode == 0

    def test_read_file_alias(self, tmp_path: Path):
        path = tmp_path / "note.txt"
        path.write_text("phase2")
        registry = ActionRegistry()
        handler = registry.get("read_file")
        assert handler is not None
        result = handler(str(path))
        assert result.error is None
        assert result.stdout == "phase2"

    def test_search_files_finds_match(self, tmp_path: Path):
        target = tmp_path / "hello_world.txt"
        target.write_text("x")
        result = _action_search_files(f"hello|{tmp_path}")
        assert result.error is None
        assert str(target) in result.stdout

    def test_search_files_requires_query(self):
        result = _action_search_files("")
        assert result.error is not None

    def test_get_battery_status_returns_status_or_error(self):
        result = _action_get_battery_status()
        assert result.stdout or result.error
