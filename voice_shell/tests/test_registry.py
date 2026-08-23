from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice_shell.src.actions.registry import (
    ActionRegistry,
    ActionResult,
    _action_cat,
    _action_cd,
    _action_date,
    _action_get_battery_status,
    _action_get_system_status,
    _action_launch_app,
    _action_ls,
    _action_move_file,
    _action_search_files,
    _action_set_brightness,
    _action_set_volume,
    _action_time,
    _action_write_file,
)


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

    def test_default_registry_has_settings_and_write_tools(self):
        registry = ActionRegistry()
        for name in (
            "write_file",
            "move_file",
            "set_volume",
            "set_brightness",
            "get_system_status",
        ):
            assert registry.is_allowed(name)

    def test_write_file_creates_content(self, tmp_path: Path):
        target = tmp_path / "note.txt"
        result = _action_write_file(f"{target}|hello phase2")
        assert result.error is None
        assert target.read_text(encoding="utf-8") == "hello phase2"
        assert "Wrote" in result.stdout

    def test_write_file_requires_path_and_content(self):
        result = _action_write_file("only-path")
        assert result.error is not None

    def test_move_file_renames(self, tmp_path: Path):
        src = tmp_path / "a.txt"
        dest = tmp_path / "b.txt"
        src.write_text("moved", encoding="utf-8")
        result = _action_move_file(f"{src}|{dest}")
        assert result.error is None
        assert not src.exists()
        assert dest.read_text(encoding="utf-8") == "moved"

    def test_move_file_rejects_existing_dest(self, tmp_path: Path):
        src = tmp_path / "a.txt"
        dest = tmp_path / "b.txt"
        src.write_text("x", encoding="utf-8")
        dest.write_text("y", encoding="utf-8")
        result = _action_move_file(f"{src}|{dest}")
        assert result.error is not None
        assert "already exists" in result.error

    def test_set_volume_invalid_level(self):
        result = _action_set_volume("150")
        assert result.error is not None
        assert "out of range" in result.error

    def test_set_volume_with_pactl(self):
        with patch("voice_shell.src.actions.registry.shutil.which", side_effect=lambda c: "/usr/bin/pactl" if c == "pactl" else None), \
             patch("voice_shell.src.actions.registry.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
            result = _action_set_volume("40")
        assert result.error is None
        assert result.stdout == "Volume set to 40%"
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0][:3] == ["pactl", "set-sink-volume", "@DEFAULT_SINK@"]

    def test_set_brightness_invalid_level(self):
        result = _action_set_brightness("abc")
        assert result.error is not None

    def test_set_brightness_with_brightnessctl(self):
        with patch("voice_shell.src.actions.registry.shutil.which", return_value="/usr/bin/brightnessctl"), \
             patch("voice_shell.src.actions.registry.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
            result = _action_set_brightness("55")
        assert result.error is None
        assert result.stdout == "Brightness set to 55%"

    def test_get_system_status_includes_cwd(self):
        result = _action_get_system_status()
        assert result.error is None
        assert "cwd=" in result.stdout
        assert "time=" in result.stdout
