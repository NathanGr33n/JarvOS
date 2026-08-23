import re
import json
from typing import NamedTuple, Optional, Set

from .registry import ActionRegistry, ActionResult
from .schemas import action_requires_confirmation, validate_action


class ParsedAction(NamedTuple):
    """Result of parsing an action tag from an LLM response."""
    action_name: str
    argument: str = ""


class ExecutionResult(NamedTuple):
    """Result of executing an action and cleaning the LLM response."""
    cleaned_response: str
    action_result: str = ""


class ParsedResponse(NamedTuple):
    """Parsed spoken text plus actions extracted from an LLM response."""
    cleaned_response: str
    actions: list[ParsedAction]


# Affirmative / negative phrases accepted during voice confirmation.
_AFFIRMATIVE_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|confirm|confirmed|do it|go ahead|proceed|affirmative|please do)\b",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"\b(no|nope|nah|cancel|stop|don't|do not|negative|abort|nevermind|never mind|reject)\b",
    re.IGNORECASE,
)


class ActionExecutor:
    """Parses action tags from LLM responses and executes whitelisted actions.

    Action tags follow the pattern ``[EXEC:<type>:<action>[:<arg>]]``.
    Examples:
    - ``[EXEC:shell:ls]`` → execute ``ls -la``
    - ``[EXEC:shell:cat /path/to/file]`` → read file
    - ``[EXEC:app:firefox]`` → launch application
    - ``[EXEC:time]`` → return current time

    Tags are stripped from the spoken response so the user does not hear them.
    Structured JSON payloads are preferred in Phase 2.
    """

    # Regex pattern for action tags: [EXEC:category:action[:argument]]
    _ACTION_TAG_PATTERN = re.compile(
        r"\[EXEC:(?P<category>shell|app):(?P<action>[^\]: ]+)(?:[ :](?P<arg>[^\]]*))?\]"
    )

    # Standalone commands (no arguments)
    _STANDALONE_PATTERN = re.compile(r"\[EXEC:(?P<action>time|date)\]")

    def __init__(
        self,
        registry: Optional[ActionRegistry] = None,
        require_confirmation: bool = False,
        confirmed_actions: Optional[Set[str]] = None,
    ):
        """Initialize the executor with an action registry.

        Args:
            registry: The ``ActionRegistry`` to use for looking up and executing
                actions. If ``None``, a default registry is created.
            require_confirmation: When True, sensitive actions are blocked unless
                present in ``confirmed_actions``.
            confirmed_actions: Action names the user has already confirmed.
        """
        self.registry = registry or ActionRegistry()
        self.require_confirmation = require_confirmation
        self.confirmed_actions: Set[str] = set(confirmed_actions or set())

    def confirm_action(self, action_name: str) -> None:
        """Mark an action name as user-confirmed for this session."""
        self.confirmed_actions.add(action_name)

    def clear_confirmed_actions(self) -> None:
        """Clear session-level confirmation grants."""
        self.confirmed_actions.clear()

    def needs_confirmation(self, action: ParsedAction) -> bool:
        """Return True when the action is gated and not yet session-confirmed."""
        if not self.require_confirmation:
            return False
        if not action_requires_confirmation(action.action_name):
            return False
        return action.action_name not in self.confirmed_actions

    @staticmethod
    def is_affirmative(transcript: str) -> bool:
        """Return True if the transcript clearly affirms confirmation."""
        text = (transcript or "").strip()
        if not text:
            return False
        # Prefer explicit negatives when both appear ("yes, no wait").
        if _NEGATIVE_RE.search(text) and not _AFFIRMATIVE_RE.search(text):
            return False
        if _NEGATIVE_RE.search(text) and _AFFIRMATIVE_RE.search(text):
            # Ambiguous mixed answer — treat as not affirmative.
            return False
        return bool(_AFFIRMATIVE_RE.search(text))

    @staticmethod
    def is_negative(transcript: str) -> bool:
        """Return True if the transcript clearly declines confirmation."""
        text = (transcript or "").strip()
        if not text:
            return False
        if _AFFIRMATIVE_RE.search(text) and not _NEGATIVE_RE.search(text):
            return False
        return bool(_NEGATIVE_RE.search(text))

    @staticmethod
    def describe_action(action: ParsedAction) -> str:
        """Return a short speakable description of an action."""
        name = action.action_name
        arg = (action.argument or "").strip()
        if name.startswith("app:"):
            app = name.split(":", 1)[1]
            return f"launch {app}"
        if name == "write_file":
            path = arg.split("|", 1)[0].strip() if arg else "a file"
            return f"write file {path or 'a file'}"
        if name == "move_file":
            if "|" in arg:
                src, dest = arg.split("|", 1)
                return f"move {src.strip()} to {dest.strip()}"
            return "move a file"
        if name == "set_volume":
            return f"set volume to {arg or 'the requested level'} percent"
        if name == "set_brightness":
            return f"set brightness to {arg or 'the requested level'} percent"
        if arg:
            return f"{name} {arg}"
        return name

    def parse_response(self, text: str) -> ParsedResponse:
        """Parse structured JSON or legacy tags without executing actions."""
        actions, structured_response = self.parse_structured_actions(text)
        if actions:
            cleaned = (structured_response or "").strip()
        else:
            actions = self.parse_actions(text)
            cleaned = self._strip_tags(text)
        return ParsedResponse(cleaned_response=cleaned.strip(), actions=actions)

    def parse_actions(self, text: str) -> list[ParsedAction]:
        """Extract all action tags from the LLM response text.

        Args:
            text: The raw LLM response containing action tags.

        Returns:
            A list of ``ParsedAction`` tuples.
        """
        actions = []

        for match in self._ACTION_TAG_PATTERN.finditer(text):
            category = match.group("category")
            action = match.group("action")
            arg = match.group("arg") or ""
            name = f"{category}:{action}" if category == "app" else action
            actions.append(ParsedAction(name, arg))

        for match in self._STANDALONE_PATTERN.finditer(text):
            actions.append(ParsedAction(match.group("action"), ""))

        return actions

    def parse_structured_actions(self, text: str) -> tuple[list[ParsedAction], Optional[str]]:
        """Parse JSON action payloads from model output.

        Supported payload shape:
        {
          "response": "Human-friendly response text",
          "actions": [
            {"name": "ls", "arg": "/tmp"},
            {"name": "app:firefox"}
          ]
        }
        """
        payload = self._extract_action_payload(text)
        if payload is None:
            return [], None

        raw_actions = payload.get("actions", [])
        if not isinstance(raw_actions, list):
            return [], payload.get("response")

        parsed_actions: list[ParsedAction] = []
        for item in raw_actions:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            arg = item.get("arg", "")
            if not isinstance(name, str):
                continue
            if not isinstance(arg, str):
                arg = str(arg)
            parsed_actions.append(ParsedAction(name.strip(), arg.strip()))

        response = payload.get("response")
        if response is not None and not isinstance(response, str):
            response = str(response)
        return parsed_actions, response

    def execute(self, parsed: ParsedAction, *, force: bool = False) -> ActionResult:
        """Execute a single parsed action if it is whitelisted.

        Args:
            parsed: The parsed action to execute.
            force: When True, skip confirmation gates (caller already confirmed).

        Returns:
            An ``ActionResult`` with ``stdout`` or ``error`` set.
        """
        schema_error = validate_action(parsed.action_name, parsed.argument)
        if schema_error:
            return ActionResult(error=schema_error)

        handler = self.registry.get(parsed.action_name)
        if handler is None:
            return ActionResult(error=f"Action '{parsed.action_name}' is not allowed.")

        if (
            not force
            and self.require_confirmation
            and action_requires_confirmation(parsed.action_name)
            and parsed.action_name not in self.confirmed_actions
        ):
            return ActionResult(
                error=(
                    f"Action '{parsed.action_name}' requires confirmation. "
                    "Ask the user to confirm, then retry."
                )
            )

        return handler(parsed.argument)

    def format_action_result(self, action: ParsedAction, result: ActionResult) -> str:
        """Format one action result line for HUD/TTS."""
        if result.error:
            return f"[{action.action_name}] Error: {result.error}"
        if result.stdout:
            return f"[{action.action_name}] {result.stdout}"
        return f"[{action.action_name}] done"

    def parse_and_execute(self, text: str) -> ExecutionResult:
        """Parse action tags from the text, execute them, and return a cleaned response.

        Actions that still need confirmation are reported as errors rather than run.
        Prefer the orchestrator voice-confirmation path when gates are enabled.

        Args:
            text: The raw LLM response text.

        Returns:
            An ``ExecutionResult`` with the cleaned response (action tags removed)
            and a concatenated string of action results.
        """
        parsed = self.parse_response(text)
        results = []
        for action in parsed.actions:
            result = self.execute(action)
            results.append(self.format_action_result(action, result))

        return ExecutionResult(
            cleaned_response=parsed.cleaned_response,
            action_result="\n".join(results),
        )

    def _extract_action_payload(self, text: str) -> Optional[dict]:
        """Extract and decode a JSON action payload if present."""
        text = text.strip()
        if not text:
            return None

        candidates: list[str] = [text]
        fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced_match:
            candidates.insert(0, fenced_match.group(1))

        for candidate in candidates:
            try:
                decoded = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict) and "actions" in decoded:
                return decoded
        return None

    def _strip_tags(self, text: str) -> str:
        """Remove all action tags from the text."""
        text = self._ACTION_TAG_PATTERN.sub("", text)
        text = self._STANDALONE_PATTERN.sub("", text)
        # Clean up extra whitespace left behind
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()
