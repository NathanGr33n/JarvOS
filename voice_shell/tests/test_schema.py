from voice_shell.src.actions.schema import (
    ToolCall,
    ToolSchema,
    parse_structured_tool_calls,
    serialize_tool_call,
    serialize_tool_calls,
)


class TestToolSchema:
    def test_parse_structured_single_call(self):
        calls = parse_structured_tool_calls('{"tool":"cat","argument":"/tmp/test.txt"}')
        assert calls == [ToolCall(name="cat", argument="/tmp/test.txt")]

    def test_parse_structured_list(self):
        calls = parse_structured_tool_calls('[{"tool":"time"},{"tool":"date"}]')
        assert calls == [
            ToolCall(name="time", argument=""),
            ToolCall(name="date", argument=""),
        ]

    def test_parse_structured_wrapped_tool_calls(self):
        calls = parse_structured_tool_calls(
            '{"tool_calls":[{"tool":"cat","arguments":{"path":"/tmp/a.txt"}}]}'
        )
        assert calls == [ToolCall(name="cat", argument="/tmp/a.txt")]

    def test_parse_structured_ignores_non_json(self):
        calls = parse_structured_tool_calls("hello world")
        assert calls == []

    def test_parse_structured_from_json_code_fence(self):
        calls = parse_structured_tool_calls(
            "Result:\n```json\n{\"tool\":\"time\"}\n```"
        )
        assert calls == [ToolCall(name="time", argument="")]

    def test_parse_structured_normalizes_app_tool(self):
        calls = parse_structured_tool_calls(
            '{"tool":"app","arguments":{"app":"firefox"}}'
        )
        assert calls == [ToolCall(name="app:firefox", argument="")]

    def test_parse_structured_serializes_non_string_argument(self):
        calls = parse_structured_tool_calls('{"tool":"cat","argument":123}')
        assert calls == [ToolCall(name="cat", argument="123")]

    def test_parse_structured_serializes_dict_argument_fallback(self):
        calls = parse_structured_tool_calls(
            '{"tool":"cat","arguments":{"unknown":{"k":"v"}}}'
        )
        assert calls == [ToolCall(name="cat", argument='{"k": "v"}')]

    def test_serialize_single_tool_call(self):
        serialized = serialize_tool_call(ToolCall(name="cat", argument="/tmp/a.txt"))
        assert serialized == {"tool": "cat", "argument": "/tmp/a.txt"}

    def test_serialize_tool_calls_wrapper(self):
        serialized = serialize_tool_calls(
            [
                ToolCall(name="time", argument=""),
                ToolCall(name="app:firefox", argument=""),
            ]
        )
        assert (
            serialized
            == '{"tool_calls": [{"tool": "time", "argument": ""}, {"tool": "app:firefox", "argument": ""}]}'
        )

    def test_validate_known_safe_tool(self):
        schema = ToolSchema()
        error = schema.validate(ToolCall(name="time", argument=""))
        assert error is None

    def test_validate_rejects_unknown_tool(self):
        schema = ToolSchema()
        error = schema.validate(ToolCall(name="unknown", argument=""))
        assert error is not None
        assert "Unknown tool" in error

    def test_validate_requires_argument(self):
        schema = ToolSchema()
        error = schema.validate(ToolCall(name="cat", argument=""))
        assert error is not None
        assert "requires an argument" in error

    def test_validate_rejects_extra_argument(self):
        schema = ToolSchema()
        error = schema.validate(ToolCall(name="time", argument="now"))
        assert error is not None
        assert "does not accept arguments" in error

    def test_validate_app_tool_name(self):
        schema = ToolSchema()
        error = schema.validate(ToolCall(name="app:firefox", argument=""))
        assert error is None

    def test_validate_rejects_invalid_app_tool_name(self):
        schema = ToolSchema()
        error = schema.validate(ToolCall(name="app:bad name", argument=""))
        assert error is not None
        assert "Invalid app tool name" in error

    def test_validate_cat_requires_existing_file_path(self, tmp_path):
        schema = ToolSchema()
        test_file = tmp_path / "a.txt"
        test_file.write_text("hello")
        error = schema.validate(ToolCall(name="cat", argument=str(test_file)))
        assert error is None

    def test_validate_cat_rejects_directory_path(self, tmp_path):
        schema = ToolSchema()
        error = schema.validate(ToolCall(name="cat", argument=str(tmp_path)))
        assert error is not None
        assert "requires a file path" in error

    def test_validate_cd_requires_directory_path(self, tmp_path):
        schema = ToolSchema()
        test_file = tmp_path / "file.txt"
        test_file.write_text("x")
        error = schema.validate(ToolCall(name="cd", argument=str(test_file)))
        assert error is not None
        assert "requires a directory path" in error
