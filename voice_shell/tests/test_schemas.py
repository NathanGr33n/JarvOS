from voice_shell.src.actions.schemas import (
    action_requires_confirmation,
    get_tool_schema,
    list_tool_schemas,
    render_tool_schema_prompt,
    tool_schemas_as_json,
    validate_action,
)


class TestToolSchemas:
    def test_list_tool_schemas_default_non_empty(self):
        schemas = list_tool_schemas()
        assert len(schemas) > 0
        names = {s.name for s in schemas}
        assert "ls" in names
        assert "search_files" in names
        assert "get_battery_status" in names

    def test_list_tool_schemas_filtered(self):
        schemas = list_tool_schemas(["pwd", "missing", "time"])
        assert [s.name for s in schemas] == ["pwd", "time"]

    def test_get_tool_schema(self):
        schema = get_tool_schema("read_file")
        assert schema is not None
        assert schema.category == "filesystem"
        assert any(p.required for p in schema.parameters)

    def test_tool_schemas_as_json(self):
        payload = tool_schemas_as_json(["pwd"])
        assert payload == [
            {
                "name": "pwd",
                "description": "Return the current working directory.",
                "category": "filesystem",
                "requires_confirmation": False,
                "parameters": [],
            }
        ]

    def test_render_tool_schema_prompt_includes_catalog(self):
        text = render_tool_schema_prompt(["ls", "app:firefox"])
        assert "Tool catalog:" in text
        assert "- ls:" in text
        assert "app:firefox" in text
        assert "needs confirmation" in text

    def test_validate_action_unknown(self):
        assert validate_action("rm") is not None

    def test_validate_action_required_arg(self):
        assert validate_action("cat", "") is not None
        assert validate_action("cat", "/tmp/file.txt") is None

    def test_validate_action_dynamic_app_allowed_by_schema(self):
        assert validate_action("app:custom") is None

    def test_action_requires_confirmation(self):
        assert action_requires_confirmation("app:firefox") is True
        assert action_requires_confirmation("ls") is False
        assert action_requires_confirmation("app:custom") is True
        assert action_requires_confirmation("write_file") is True
        assert action_requires_confirmation("move_file") is True
        assert action_requires_confirmation("set_volume") is True
        assert action_requires_confirmation("set_brightness") is True

    def test_new_settings_and_fs_tools_present(self):
        names = {s.name for s in list_tool_schemas()}
        assert "write_file" in names
        assert "move_file" in names
        assert "set_volume" in names
        assert "set_brightness" in names
        assert "get_system_status" in names

    def test_validate_write_file_requires_arg(self):
        assert validate_action("write_file", "") is not None
        assert validate_action("write_file", "/tmp/a.txt|hi") is None
