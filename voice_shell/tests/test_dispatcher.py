from voice_shell.src.actions.dispatcher import DispatchPolicy, ToolDispatcher
from voice_shell.src.actions.registry import ActionRegistry, ActionResult
from voice_shell.src.actions.schema import ToolCall


class TestToolDispatcher:
    def test_dispatch_success_for_safe_tool(self):
        dispatcher = ToolDispatcher(registry=ActionRegistry())
        result = dispatcher.dispatch(ToolCall(name="time", argument=""))
        assert result.error is None
        assert result.stdout

    def test_dispatch_rejects_invalid_call(self):
        dispatcher = ToolDispatcher(registry=ActionRegistry())
        result = dispatcher.dispatch(ToolCall(name="cat", argument=""))
        assert result.error is not None
        assert "requires an argument" in result.error

    def test_dispatch_blocks_launch_when_policy_disallows(self):
        dispatcher = ToolDispatcher(
            registry=ActionRegistry(),
            policy=DispatchPolicy(allow_launch_only=False),
        )
        result = dispatcher.dispatch(ToolCall(name="app:firefox", argument=""))
        assert result.error is not None
        assert "blocked by policy" in result.error

    def test_dispatch_rejects_unknown_tool(self):
        dispatcher = ToolDispatcher(registry=ActionRegistry())
        result = dispatcher.dispatch(ToolCall(name="unknown", argument=""))
        assert result.error is not None
        assert "Unknown tool" in result.error

    def test_dispatch_blocks_tool_in_blocklist(self):
        dispatcher = ToolDispatcher(
            registry=ActionRegistry(),
            policy=DispatchPolicy(blocked_tools={"time"}),
        )
        result = dispatcher.dispatch(ToolCall(name="time", argument=""))
        assert result.error is not None
        assert "blocked by policy list" in result.error

    def test_dispatch_routes_filesystem_calls(self):
        dispatcher = ToolDispatcher(registry=ActionRegistry())
        called = {"filesystem": False}

        def _filesystem_route(handler, call):
            called["filesystem"] = True
            return handler(call.argument)

        dispatcher._category_dispatchers[dispatcher.schema.get_definition("pwd").category] = _filesystem_route
        result = dispatcher.dispatch(ToolCall(name="pwd", argument=""))
        assert result.error is None
        assert called["filesystem"] is True

    def test_dispatch_routes_application_calls(self):
        class _Registry:
            @staticmethod
            def get(name: str):
                if name == "app:firefox":
                    return lambda _: ActionResult(stdout="launched")
                return None

        dispatcher = ToolDispatcher(registry=_Registry())
        called = {"application": False}

        def _application_route(handler, call):
            called["application"] = True
            return handler(call.argument)

        dispatcher._category_dispatchers[dispatcher.schema.get_definition("app:firefox").category] = _application_route
        result = dispatcher.dispatch(ToolCall(name="app:firefox", argument=""))
        assert result.error is None
        assert result.stdout == "launched"
        assert called["application"] is True

    def test_dispatch_retries_once_for_transient_error(self):
        attempts = {"count": 0}

        def _flaky_handler(_: str):
            attempts["count"] += 1
            if attempts["count"] == 1:
                return ActionResult(error="temporarily unavailable", returncode=1)
            return ActionResult(stdout="ok")

        class _Registry:
            @staticmethod
            def get(name: str):
                return _flaky_handler if name == "time" else None

        dispatcher = ToolDispatcher(
            registry=_Registry(),
            policy=DispatchPolicy(max_transient_retries=1),
        )
        result = dispatcher.dispatch(ToolCall(name="time", argument=""))
        assert result.error is None
        assert result.stdout == "ok"
        assert attempts["count"] == 2

    def test_dispatch_does_not_retry_non_transient_error(self):
        attempts = {"count": 0}

        def _failing_handler(_: str):
            attempts["count"] += 1
            return ActionResult(error="permission denied", returncode=1)

        class _Registry:
            @staticmethod
            def get(name: str):
                return _failing_handler if name == "time" else None

        dispatcher = ToolDispatcher(
            registry=_Registry(),
            policy=DispatchPolicy(max_transient_retries=2),
        )
        result = dispatcher.dispatch(ToolCall(name="time", argument=""))
        assert result.error == "permission denied"
        assert attempts["count"] == 1

    def test_dispatch_returns_last_error_when_retries_exhausted(self):
        attempts = {"count": 0}

        def _always_transient(_: str):
            attempts["count"] += 1
            return ActionResult(error="timeout while executing", returncode=1)

        class _Registry:
            @staticmethod
            def get(name: str):
                return _always_transient if name == "time" else None

        dispatcher = ToolDispatcher(
            registry=_Registry(),
            policy=DispatchPolicy(max_transient_retries=2),
        )
        result = dispatcher.dispatch(ToolCall(name="time", argument=""))
        assert result.error == "timeout while executing"
        assert attempts["count"] == 3

    def test_dispatch_retries_on_transient_oserror(self):
        attempts = {"count": 0}

        def _flaky_os_handler(_: str):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise OSError("resource temporarily unavailable")
            return ActionResult(stdout="ok")

        class _Registry:
            @staticmethod
            def get(name: str):
                return _flaky_os_handler if name == "time" else None

        dispatcher = ToolDispatcher(
            registry=_Registry(),
            policy=DispatchPolicy(max_transient_retries=1),
        )
        result = dispatcher.dispatch(ToolCall(name="time", argument=""))
        assert result.error is None
        assert result.stdout == "ok"
        assert attempts["count"] == 2
