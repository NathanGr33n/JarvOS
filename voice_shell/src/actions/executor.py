import re
from typing import NamedTuple, Optional

from .registry import ActionRegistry, ActionResult


class ParsedAction(NamedTuple):
    """Result of parsing an action tag from an LLM response."""
    action_name: str
    argument: str = ""


class ExecutionResult(NamedTuple):
    """Result of executing an action and cleaning the LLM response."""
    cleaned_response: str
    action_result: str = ""


class ActionExecutor:
    """Parses action tags from LLM responses and executes whitelisted actions.

    Action tags follow the pattern ``[EXEC:<type>:<action>[:<arg>]]``.
    Examples:
    - ``[EXEC:shell:ls]`` → execute ``ls -la``
    - ``[EXEC:shell:cat /path/to/file]`` → read file
    - ``[EXEC:app:firefox]`` → launch application
    - ``[EXEC:time]`` → return current time

    Tags are stripped from the spoken response so the user does not hear them.
    """

    # Regex pattern for action tags: [EXEC:category:action[:argument]]
    _ACTION_TAG_PATTERN = re.compile(
        r"\[EXEC:(?P<category>shell|app):(?P<action>[^\]:]+)(?::(?P<arg>[^\]]*))?\]"
    )

    # Standalone commands (no arguments)
    _STANDALONE_PATTERN = re.compile(r"\[EXEC:(?P<action>time|date)\]")

    def __init__(self, registry: Optional[ActionRegistry] = None):
        """Initialize the executor with an action registry.

        Args:
            registry: The ``ActionRegistry`` to use for looking up and executing
                actions. If ``None``, a default registry is created.
        """
        self.registry = registry or ActionRegistry()

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

    def execute(self, parsed: ParsedAction) -> ActionResult:
        """Execute a single parsed action if it is whitelisted.

        Args:
            parsed: The parsed action to execute.

        Returns:
            An ``ActionResult`` with ``stdout`` or ``error`` set.
        """
        handler = self.registry.get(parsed.action_name)
        if handler is None:
            return ActionResult(error=f"Action '{parsed.action_name}' is not allowed.")
        return handler(parsed.argument)

    def parse_and_execute(self, text: str) -> ExecutionResult:
        """Parse action tags from the text, execute them, and return a cleaned response.

        Args:
            text: The raw LLM response text.

        Returns:
            An ``ExecutionResult`` with the cleaned response (action tags removed)
            and a concatenated string of action results.
        """
        actions = self.parse_actions(text)
        cleaned = self._strip_tags(text)

        results = []
        for action in actions:
            result = self.execute(action)
            if result.error:
                results.append(f"[{action.action_name}] Error: {result.error}")
            elif result.stdout:
                results.append(f"[{action.action_name}] {result.stdout}")

        return ExecutionResult(
            cleaned_response=cleaned.strip(),
            action_result="\n".join(results),
        )

    def _strip_tags(self, text: str) -> str:
        """Remove all action tags from the text."""
        text = self._ACTION_TAG_PATTERN.sub("", text)
        text = self._STANDALONE_PATTERN.sub("", text)
        # Clean up extra whitespace left behind
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()
