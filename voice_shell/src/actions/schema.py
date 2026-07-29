import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ToolCategory(Enum):
    """High-level tool category for dispatch routing."""

    FILESYSTEM = "filesystem"
    APPLICATION = "application"
    SYSTEM = "system"


class SafetyClass(Enum):
    """Safety classification for a tool call."""

    SAFE_READ = "safe_read"
    LAUNCH_ONLY = "launch_only"
    CONFIRM_REQUIRED = "confirm_required"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ToolCall:
    """Normalized structured representation of a tool call."""

    name: str
    argument: str = ""


@dataclass(frozen=True)
class ToolDefinition:
    """Schema entry for a known tool."""

    name: str
    category: ToolCategory
    safety_class: SafetyClass
    requires_argument: bool
    allows_argument: bool = True


class ToolSchema:
    """Tool schema and validators for Phase 2 structured action calls."""

    _APP_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")

    def __init__(self):
        self._definitions = {
            "ls": ToolDefinition(
                name="ls",
                category=ToolCategory.FILESYSTEM,
                safety_class=SafetyClass.SAFE_READ,
                requires_argument=False,
                allows_argument=False,
            ),
            "cat": ToolDefinition(
                name="cat",
                category=ToolCategory.FILESYSTEM,
                safety_class=SafetyClass.SAFE_READ,
                requires_argument=True,
            ),
            "cd": ToolDefinition(
                name="cd",
                category=ToolCategory.FILESYSTEM,
                safety_class=SafetyClass.SAFE_READ,
                requires_argument=True,
            ),
            "pwd": ToolDefinition(
                name="pwd",
                category=ToolCategory.FILESYSTEM,
                safety_class=SafetyClass.SAFE_READ,
                requires_argument=False,
                allows_argument=False,
            ),
            "time": ToolDefinition(
                name="time",
                category=ToolCategory.SYSTEM,
                safety_class=SafetyClass.SAFE_READ,
                requires_argument=False,
                allows_argument=False,
            ),
            "date": ToolDefinition(
                name="date",
                category=ToolCategory.SYSTEM,
                safety_class=SafetyClass.SAFE_READ,
                requires_argument=False,
                allows_argument=False,
            ),
        }

    def get_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        if tool_name.startswith("app:"):
            return ToolDefinition(
                name=tool_name,
                category=ToolCategory.APPLICATION,
                safety_class=SafetyClass.LAUNCH_ONLY,
                requires_argument=False,
                allows_argument=False,
            )
        return self._definitions.get(tool_name)

    def validate(self, call: ToolCall) -> Optional[str]:
        definition = self.get_definition(call.name)
        if definition is None:
            return f"Unknown tool '{call.name}'."

        argument = call.argument.strip()

        if definition.requires_argument and not argument:
            return f"Tool '{call.name}' requires an argument."
        if not definition.allows_argument and argument:
            return f"Tool '{call.name}' does not accept arguments."
        if "\x00" in argument:
            return f"Tool '{call.name}' argument contains invalid null byte."

        if definition.category == ToolCategory.APPLICATION:
            return self._validate_app_tool_name(call.name)
        if definition.category == ToolCategory.FILESYSTEM:
            return self._validate_filesystem_call(call.name, argument)

        return None

    def _validate_app_tool_name(self, tool_name: str) -> Optional[str]:
        app_name = tool_name.split(":", 1)[1]
        if not app_name or not self._APP_NAME_PATTERN.match(app_name):
            return f"Invalid app tool name '{tool_name}'."
        return None

    def _validate_filesystem_call(self, tool_name: str, argument: str) -> Optional[str]:
        if tool_name not in {"cat", "cd"}:
            return None
        target = Path(argument).expanduser()
        if not target.exists():
            return f"Path not found: {argument}"
        if tool_name == "cat" and target.is_dir():
            return f"Tool '{tool_name}' requires a file path, got directory: {argument}"
        if tool_name == "cd" and not target.is_dir():
            return f"Tool '{tool_name}' requires a directory path: {argument}"
        return None


def parse_structured_tool_calls(text: str) -> list[ToolCall]:
    """Parse structured JSON tool call payloads from LLM output.

    Supports payloads:
    - {"tool": "cat", "argument": "/tmp/a.txt"}
    - {"tool": "cat", "arguments": {"path": "/tmp/a.txt"}}
    - [{"tool": "time"}, {"tool": "app:firefox"}]
    - {"tool_calls": [{...}, {...}]}
    """

    raw = text.strip()
    if not raw:
        return []
    if not raw.startswith("{") and not raw.startswith("["):
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, dict) and "tool_calls" in payload:
        entries = payload.get("tool_calls") or []
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = [payload]

    tool_calls: list[ToolCall] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        tool_name = entry.get("tool") or entry.get("name") or entry.get("action")
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        tool_name = tool_name.strip()

        arg = _extract_argument(entry)
        tool_calls.append(ToolCall(name=tool_name, argument=arg))

    return tool_calls


def _extract_argument(entry: dict[str, Any]) -> str:
    argument = entry.get("argument")
    if isinstance(argument, str):
        return argument

    arguments = entry.get("arguments") or entry.get("args")
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict):
        for key in ("path", "app", "value", "text"):
            value = arguments.get(key)
            if isinstance(value, str):
                return value

    return ""