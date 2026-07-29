import time
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

    def test_parse_structured_tool_call(self):
        """Verify parsing of a structured JSON tool call."""
        executor = ActionExecutor()
        actions = executor.parse_actions('{"tool":"cat","argument":"/tmp/a.txt"}')
        assert len(actions) == 1
        assert actions[0] == ParsedAction("cat", "/tmp/a.txt")

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
        assert len(result.action_reports) == 1
        assert result.action_reports[0].action_name == "time"
        assert result.action_reports[0].status == "ok"

    def test_parse_and_execute_no_actions(self):
        """Verify parse_and_execute with no action tags returns the original text."""
        executor = ActionExecutor()
        result = executor.parse_and_execute("Hello, I am Nova.")
        assert result.cleaned_response == "Hello, I am Nova."
        assert result.action_result == ""

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

    def test_execute_timeout_returns_structured_error(self):
        """Verify execute returns a timeout error when dispatch exceeds timeout."""
        executor = ActionExecutor(execution_timeout_seconds=0.01)

        def _slow_dispatch(_):
            time.sleep(0.05)
            return ActionResult(stdout="done")

        executor.dispatcher.dispatch = _slow_dispatch
        result = executor.execute(ParsedAction("time", ""))
        assert result.error is not None
        assert "timed out" in result.error

    def test_execute_cancellation_returns_structured_error(self):
        """Verify execute handles cancellation-like interrupts with a clear error."""
        executor = ActionExecutor()

        def _cancel_dispatch(_):
            raise KeyboardInterrupt

        executor.dispatcher.dispatch = _cancel_dispatch
        result = executor.execute(ParsedAction("time", ""))
        assert result.error is not None
        assert "cancelled" in result.error.lower()

    def test_execute_exception_returns_structured_error(self):
        """Verify execute maps unexpected dispatcher errors into ActionResult.error."""
        executor = ActionExecutor()

        def _error_dispatch(_):
            raise RuntimeError("boom")

        executor.dispatcher.dispatch = _error_dispatch
        result = executor.execute(ParsedAction("time", ""))
        assert result.error is not None
        assert "failed: boom" in result.error

    def test_parse_and_execute_error_report_status(self):
        """Verify parse_and_execute sets timeout/cancellation statuses in reports."""
        executor = ActionExecutor()
        sequence = iter(
            [
                ActionResult(error="Action 'time' timed out after 0.10s."),
                ActionResult(error="Action 'date' cancelled before completion."),
                ActionResult(error="Action 'ls' failed: boom"),
            ]
        )

        executor.execute = lambda _: next(sequence)
        result = executor.parse_and_execute(
            "[EXEC:time] [EXEC:date] [EXEC:shell:ls]"
        )

        assert [report.status for report in result.action_reports] == [
            "timeout",
            "cancelled",
            "error",
        ]

    def test_cleaned_text_whitespace_normalized(self):
        """Verify that extra whitespace is normalized after tag removal."""
        executor = ActionExecutor()
        text = "[EXEC:shell:ls]  [EXEC:time]   Done."
        cleaned = executor._strip_tags(text)
        assert "  " not in cleaned
        assert "Done." in cleaned