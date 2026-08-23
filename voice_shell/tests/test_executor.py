from voice_shell.src.actions.executor import ActionExecutor, ParsedAction
from voice_shell.src.actions.registry import ActionRegistry, ActionResult


class TestActionExecutor:
    """Tests for the ActionExecutor."""

    def test_parse_shell_action_no_arg(self):
        """Verify parsing of a shell action without argument."""
        executor = ActionExecutor()
        actions = executor.parse_actions("[EXEC:shell:ls]")
        assert len(actions) == 1
        assert actions[0] == ParsedAction("ls", "")

    def test_parse_shell_action_with_arg(self):
        """Verify parsing of a shell action with an argument."""
        executor = ActionExecutor()
        actions = executor.parse_actions("[EXEC:shell:cat /home/user/file.txt]")
        assert len(actions) == 1
        assert actions[0] == ParsedAction("cat", "/home/user/file.txt")

    def test_parse_app_action(self):
        """Verify parsing of an application launch action."""
        executor = ActionExecutor()
        actions = executor.parse_actions("[EXEC:app:firefox]")
        assert len(actions) == 1
        assert actions[0] == ParsedAction("app:firefox", "")

    def test_parse_standalone_time(self):
        """Verify parsing of a standalone time action."""
        executor = ActionExecutor()
        actions = executor.parse_actions("[EXEC:time]")
        assert len(actions) == 1
        assert actions[0] == ParsedAction("time", "")

    def test_parse_standalone_date(self):
        """Verify parsing of a standalone date action."""
        executor = ActionExecutor()
        actions = executor.parse_actions("[EXEC:date]")
        assert len(actions) == 1
        assert actions[0] == ParsedAction("date", "")

    def test_parse_multiple_actions(self):
        """Verify parsing of multiple actions in one response."""
        text = "[EXEC:shell:ls] Then [EXEC:app:firefox] and [EXEC:time]"
        executor = ActionExecutor()
        actions = executor.parse_actions(text)
        assert len(actions) == 3
        assert actions[0] == ParsedAction("ls", "")
        assert actions[1] == ParsedAction("app:firefox", "")
        assert actions[2] == ParsedAction("time", "")

    def test_parse_no_actions(self):
        """Verify that a text with no action tags returns an empty list."""
        executor = ActionExecutor()
        actions = executor.parse_actions("Hello, how are you?")
        assert actions == []

    def test_strip_tags(self):
        """Verify that action tags are removed from the response text."""
        executor = ActionExecutor()
        text = "[EXEC:shell:ls] Your files are ready."
        cleaned = executor._strip_tags(text)
        assert "[EXEC:shell:ls]" not in cleaned
        assert "Your files are ready." in cleaned

    def test_parse_and_execute_with_mock_registry(self):
        """Verify parse_and_execute delegates to the registry and returns cleaned text."""
        registry = ActionRegistry()
        executor = ActionExecutor(registry)
        result = executor.parse_and_execute("[EXEC:time] The current time is shown.")
        assert "[EXEC:time]" not in result.cleaned_response
        assert "The current time is shown." in result.cleaned_response
        assert "time" in result.action_result

    def test_parse_and_execute_no_actions(self):
        """Verify parse_and_execute with no action tags returns the original text."""
        executor = ActionExecutor()
        result = executor.parse_and_execute("Hello, I am Nova.")
        assert result.cleaned_response == "Hello, I am Nova."
        assert result.action_result == ""

    def test_parse_structured_actions_json_object(self):
        """Verify JSON payload actions are parsed with response text."""
        executor = ActionExecutor()
        payload = """
{
  "response": "Listing your files now.",
  "actions": [
    {"name": "ls", "arg": "."},
    {"name": "time"}
  ]
}
"""
        actions, response = executor.parse_structured_actions(payload)
        assert response == "Listing your files now."
        assert actions == [ParsedAction("ls", "."), ParsedAction("time", "")]

    def test_parse_and_execute_structured_payload(self):
        """Verify structured payload is executed and cleaned response comes from response field."""
        executor = ActionExecutor()
        payload = """
{
  "response": "Checking current folder.",
  "actions": [{"name": "pwd"}]
}
"""
        result = executor.parse_and_execute(payload)
        assert result.cleaned_response == "Checking current folder."
        assert "pwd" in result.action_result

    def test_execute_allowed_action(self):
        """Verify that executing an allowed action returns a result."""
        executor = ActionExecutor()
        result = executor.execute(ParsedAction("time", ""))
        assert result.error is None
        assert len(result.stdout) > 0

    def test_execute_disallowed_action(self):
        """Verify that executing a disallowed action returns an error."""
        executor = ActionExecutor(ActionRegistry(allowed_shell_commands=[]))
        result = executor.execute(ParsedAction("ls", ""))
        assert result.error is not None
        assert "not allowed" in result.error

    def test_execute_app_action(self):
        """Verify that an app action is handled correctly."""
        executor = ActionExecutor()
        # nonexistent app should return an error from the registry handler
        result = executor.execute(ParsedAction("app:nonexistent_app_12345", ""))
        assert result.error is not None

    def test_cleaned_text_whitespace_normalized(self):
        """Verify that extra whitespace is normalized after tag removal."""
        executor = ActionExecutor()
        text = "[EXEC:shell:ls]  [EXEC:time]   Done."
        cleaned = executor._strip_tags(text)
        assert "  " not in cleaned
        assert "Done." in cleaned

class TestConfirmationAndSchema:
    """Phase 2 confirmation and schema validation behavior."""

    def test_confirmation_blocks_app_launch(self):
        executor = ActionExecutor(require_confirmation=True)
        result = executor.execute(ParsedAction("app:firefox", ""))
        assert result.error is not None
        assert "confirmation" in result.error

    def test_confirmation_allows_after_confirm(self):
        executor = ActionExecutor(require_confirmation=True)
        executor.confirm_action("app:firefox")
        # May fail to launch binary, but must not fail for confirmation.
        result = executor.execute(ParsedAction("app:firefox", ""))
        assert result.error is None or "Cannot launch" in (result.error or "")

    def test_schema_rejects_missing_required_arg(self):
        executor = ActionExecutor()
        result = executor.execute(ParsedAction("cat", ""))
        assert result.error is not None
        assert "requires argument" in result.error

    def test_non_sensitive_action_without_confirmation(self):
        executor = ActionExecutor(require_confirmation=True)
        result = executor.execute(ParsedAction("time", ""))
        assert result.error is None

    def test_confirmation_blocks_write_file(self):
        executor = ActionExecutor(require_confirmation=True)
        result = executor.execute(ParsedAction("write_file", "/tmp/x|y"))
        assert result.error is not None
        assert "confirmation" in result.error

    def test_confirmation_blocks_set_volume(self):
        executor = ActionExecutor(require_confirmation=True)
        result = executor.execute(ParsedAction("set_volume", "20"))
        assert result.error is not None
        assert "confirmation" in result.error
