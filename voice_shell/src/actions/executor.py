import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Literal, NamedTuple, Optional

from .dispatcher import ToolDispatcher
from .registry import ActionRegistry, ActionResult
from .schema import ToolCall, ToolSchema, parse_structured_tool_calls


class ParsedAction(NamedTuple):
    """Result of parsing an action tag or structured tool call."""

    action_name: str
    argument: str = ""


class ExecutionResult(NamedTuple):
    """Result of executing an action and cleaning the LLM response."""

    cleaned_response: str
    action_result: str = ""
    action_reports: tuple["ActionExecutionReport", ...] = ()


class ActionExecutionReport(NamedTuple):
    """Structured per-action execution summary for dispatcher-facing consumers."""

    action_name: str
    status: Literal["ok", "error", "timeout", "cancelled"]
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    error: Optional[str] = None


class ActionExecutor:
    """Parses action calls from LLM responses and executes whitelisted actions.

    Phase 2 foundation:
    - Prefer structured JSON tool calls (schema-validated).
    - Keep legacy ``[EXEC:...]`` tags as migration compatibility.
    """

    # Legacy regex pattern for action tags: [EXEC:category:action[:argument]]
    _ACTION_TAG_PATTERN = re.compile(
        r"\[EXEC:(?P<category>shell|app):(?P<action>[^\]: ]+)(?:[ :](?P<arg>[^\]]*))?\]"
    )

    # Legacy standalone commands (no arguments)
    _STANDALONE_PATTERN = re.compile(r"\[EXEC:(?P<action>time|date)\]")

    def __init__(
        self,
        registry: Optional[ActionRegistry] = None,
        execution_timeout_seconds: float = 3.0,
    ):
        self.registry = registry or ActionRegistry()
        self.schema = ToolSchema()
        self.execution_timeout_seconds = execution_timeout_seconds
        self.dispatcher = ToolDispatcher(
            registry=self.registry,
            schema=self.schema,
        )

    def parse_actions(self, text: str) -> list[ParsedAction]:
        """Extract actions from JSON tool payloads or legacy action tags."""

        actions: list[ParsedAction] = []

        structured_calls = parse_structured_tool_calls(text)
        if structured_calls:
            for call in structured_calls:
                actions.append(ParsedAction(call.name, call.argument))
            return actions

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
        """Execute a parsed action through the policy-aware dispatcher."""

        call = ToolCall(name=parsed.action_name, argument=parsed.argument)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.dispatcher.dispatch, call)
            try:
                return future.result(timeout=self.execution_timeout_seconds)
            except FuturesTimeoutError:
                return ActionResult(
                    error=(
                        f"Action '{parsed.action_name}' timed out after "
                        f"{self.execution_timeout_seconds:.2f}s."
                    )
                )
            except KeyboardInterrupt:
                return ActionResult(
                    error=f"Action '{parsed.action_name}' cancelled before completion."
                )
            except Exception as exc:
                return ActionResult(error=f"Action '{parsed.action_name}' failed: {exc}")

    def parse_and_execute(self, text: str) -> ExecutionResult:
        """Parse actions from text, execute them, and return cleaned response."""

        actions = self.parse_actions(text)
        cleaned = self._strip_tags(text)

        results: list[str] = []
        reports: list[ActionExecutionReport] = []
        for action in actions:
            result = self.execute(action)
            if result.error:
                status: Literal["error", "timeout", "cancelled"] = "error"
                lower_error = result.error.lower()
                if "timed out" in lower_error:
                    status = "timeout"
                elif "cancelled" in lower_error:
                    status = "cancelled"
                results.append(f"[{action.action_name}] Error: {result.error}")
                reports.append(
                    ActionExecutionReport(
                        action_name=action.action_name,
                        status=status,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        returncode=result.returncode,
                        error=result.error,
                    )
                )
            elif result.stdout:
                results.append(f"[{action.action_name}] {result.stdout}")
                reports.append(
                    ActionExecutionReport(
                        action_name=action.action_name,
                        status="ok",
                        stdout=result.stdout,
                        stderr=result.stderr,
                        returncode=result.returncode,
                    )
                )
            else:
                reports.append(
                    ActionExecutionReport(
                        action_name=action.action_name,
                        status="ok",
                        stdout=result.stdout,
                        stderr=result.stderr,
                        returncode=result.returncode,
                    )
                )

        return ExecutionResult(
            cleaned_response=cleaned.strip(),
            action_result="\n".join(results),
            action_reports=tuple(reports),
        )

    def _strip_tags(self, text: str) -> str:
        """Remove legacy action tags from text.

        Structured JSON payloads are not altered by this method.
        """

        text = self._ACTION_TAG_PATTERN.sub("", text)
        text = self._STANDALONE_PATTERN.sub("", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()