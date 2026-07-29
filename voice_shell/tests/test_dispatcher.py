from voice_shell.src.actions.dispatcher import DispatchPolicy, ToolDispatcher
from voice_shell.src.actions.registry import ActionRegistry
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
