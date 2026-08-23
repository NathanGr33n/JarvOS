"""JSON-serializable tool schemas for the Phase 2 action layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ToolParameter:
    """A single tool argument definition."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False


@dataclass(frozen=True)
class ToolSchema:
    """Schema describing one OS tool the LLM may invoke."""

    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    requires_confirmation: bool = False
    category: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "requires_confirmation": self.requires_confirmation,
            "parameters": [asdict(p) for p in self.parameters],
        }


# Canonical tool definitions used for prompt injection and validation.
TOOL_SCHEMAS: Dict[str, ToolSchema] = {
    "ls": ToolSchema(
        name="ls",
        description="List files in a directory (alias of list_directory).",
        parameters=[
            ToolParameter("arg", description="Directory path (default: current directory)."),
        ],
        category="filesystem",
    ),
    "list_directory": ToolSchema(
        name="list_directory",
        description="List files in a directory.",
        parameters=[
            ToolParameter("arg", description="Directory path (default: current directory)."),
        ],
        category="filesystem",
    ),
    "cat": ToolSchema(
        name="cat",
        description="Read a text file (alias of read_file).",
        parameters=[
            ToolParameter("arg", description="Absolute or relative file path.", required=True),
        ],
        category="filesystem",
    ),
    "read_file": ToolSchema(
        name="read_file",
        description="Read the contents of a text file.",
        parameters=[
            ToolParameter("arg", description="Absolute or relative file path.", required=True),
        ],
        category="filesystem",
    ),
    "search_files": ToolSchema(
        name="search_files",
        description="Search for files by name under a directory. Arg format: 'query' or 'query|/path'.",
        parameters=[
            ToolParameter(
                "arg",
                description="Search query, optionally with root path as 'query|/path'.",
                required=True,
            ),
        ],
        category="filesystem",
    ),
    "cd": ToolSchema(
        name="cd",
        description="Change the current working directory.",
        parameters=[
            ToolParameter("arg", description="Directory path.", required=True),
        ],
        category="filesystem",
    ),
    "pwd": ToolSchema(
        name="pwd",
        description="Return the current working directory.",
        category="filesystem",
    ),
    "time": ToolSchema(
        name="time",
        description="Return the current local time.",
        category="system",
    ),
    "date": ToolSchema(
        name="date",
        description="Return the current local date.",
        category="system",
    ),
    "get_battery_status": ToolSchema(
        name="get_battery_status",
        description="Return battery charge percentage and charging state when available.",
        category="system",
    ),
    "write_file": ToolSchema(
        name="write_file",
        description="Write text content to a file. Arg format: 'path|content'.",
        parameters=[
            ToolParameter(
                "arg",
                description="Destination path and content as 'path|content'.",
                required=True,
            ),
        ],
        requires_confirmation=True,
        category="filesystem",
    ),
    "move_file": ToolSchema(
        name="move_file",
        description="Move or rename a file/directory. Arg format: 'src|dest'.",
        parameters=[
            ToolParameter(
                "arg",
                description="Source and destination as 'src|dest'.",
                required=True,
            ),
        ],
        requires_confirmation=True,
        category="filesystem",
    ),
    "set_volume": ToolSchema(
        name="set_volume",
        description="Set system output volume to a percentage (0-100).",
        parameters=[
            ToolParameter("arg", description="Volume level 0-100.", required=True),
        ],
        requires_confirmation=True,
        category="settings",
    ),
    "set_brightness": ToolSchema(
        name="set_brightness",
        description="Set display brightness to a percentage (0-100).",
        parameters=[
            ToolParameter("arg", description="Brightness level 0-100.", required=True),
        ],
        requires_confirmation=True,
        category="settings",
    ),
    "get_system_status": ToolSchema(
        name="get_system_status",
        description="Return local time, cwd, load, memory, and battery summary.",
        category="system",
    ),
    "app:firefox": ToolSchema(
        name="app:firefox",
        description="Launch the Firefox browser.",
        requires_confirmation=True,
        category="application",
    ),
    "app:nautilus": ToolSchema(
        name="app:nautilus",
        description="Launch the Nautilus file manager.",
        requires_confirmation=True,
        category="application",
    ),
    "app:code": ToolSchema(
        name="app:code",
        description="Launch Visual Studio Code.",
        requires_confirmation=True,
        category="application",
    ),
    "app:terminal": ToolSchema(
        name="app:terminal",
        description="Launch a terminal emulator.",
        requires_confirmation=True,
        category="application",
    ),
}


def list_tool_schemas(names: Optional[Iterable[str]] = None) -> List[ToolSchema]:
    """Return tool schemas, optionally filtered to the given names."""
    if names is None:
        return [TOOL_SCHEMAS[k] for k in sorted(TOOL_SCHEMAS)]
    result: List[ToolSchema] = []
    for name in names:
        schema = TOOL_SCHEMAS.get(name)
        if schema is not None:
            result.append(schema)
    return result


def get_tool_schema(name: str) -> Optional[ToolSchema]:
    """Look up a single tool schema by name."""
    return TOOL_SCHEMAS.get(name)


def tool_schemas_as_json(names: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Return tool schemas as plain JSON-serializable dictionaries."""
    return [schema.to_dict() for schema in list_tool_schemas(names)]


def render_tool_schema_prompt(names: Optional[Iterable[str]] = None) -> str:
    """Render a compact tool-schema block for injection into the system prompt."""
    schemas = list_tool_schemas(names)
    if not schemas:
        return "No tools are currently available."

    lines = [
        "Available tools (prefer JSON actions):",
        'Respond with JSON: {"response": "...", "actions": [{"name": "...", "arg": "..."}]}',
        "Tool catalog:",
    ]
    for schema in schemas:
        params = ", ".join(
            f"{p.name}{'*' if p.required else ''}: {p.description or p.type}"
            for p in schema.parameters
        )
        confirm = " [needs confirmation]" if schema.requires_confirmation else ""
        param_part = f" args=[{params}]" if params else " args=[]"
        lines.append(f"- {schema.name}: {schema.description}{param_part}{confirm}")
    return "\n".join(lines)


def validate_action(name: str, arg: str = "") -> Optional[str]:
    """Validate an action name and argument against the schema catalog.

    Returns:
        An error message if invalid, otherwise ``None``.
    """
    schema = TOOL_SCHEMAS.get(name)
    if schema is None:
        # Allow dynamic app: names not pre-listed; registry whitelist is authoritative.
        if name.startswith("app:") and len(name) > 4:
            return None
        return f"Unknown action '{name}'."

    required = [p for p in schema.parameters if p.required]
    if required and not (arg or "").strip():
        req_names = ", ".join(p.name for p in required)
        return f"Action '{name}' requires argument(s): {req_names}."
    return None


def action_requires_confirmation(name: str) -> bool:
    """Return whether the named action is marked as requiring confirmation."""
    schema = TOOL_SCHEMAS.get(name)
    if schema is not None:
        return schema.requires_confirmation
    # Unlisted app launches are treated as sensitive by default.
    return name.startswith("app:")
