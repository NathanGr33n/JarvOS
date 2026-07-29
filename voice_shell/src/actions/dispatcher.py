from dataclasses import dataclass, field
from typing import Optional

from .registry import ActionRegistry, ActionResult
from .schema import SafetyClass, ToolCall, ToolCategory, ToolSchema


@dataclass(frozen=True)
class DispatchPolicy:
    """Controls which safety classes may execute immediately."""

    allow_safe_read: bool = True
    allow_launch_only: bool = True
    allow_confirm_required: bool = False
    allow_blocked: bool = False
    blocked_tools: set[str] = field(default_factory=set)
    max_transient_retries: int = 1


class ToolDispatcher:
    """Dispatches validated tool calls to registry handlers."""

    def __init__(
        self,
        registry: Optional[ActionRegistry] = None,
        schema: Optional[ToolSchema] = None,
        policy: Optional[DispatchPolicy] = None,
    ):
        self.registry = registry or ActionRegistry()
        self.schema = schema or ToolSchema()
        self.policy = policy or DispatchPolicy()
        self._category_dispatchers = {
            ToolCategory.FILESYSTEM: self._dispatch_filesystem,
            ToolCategory.APPLICATION: self._dispatch_application,
            ToolCategory.SYSTEM: self._dispatch_system,
        }

    def dispatch(self, call: ToolCall) -> ActionResult:
        validation_error = self.schema.validate(call)
        if validation_error:
            return ActionResult(error=validation_error)

        definition = self.schema.get_definition(call.name)
        if definition is None:
            return ActionResult(error=f"Unknown tool '{call.name}'.")

        if not self._is_allowed_by_policy(definition.safety_class):
            return ActionResult(
                error=(
                    f"Tool '{call.name}' is blocked by policy "
                    f"({definition.safety_class.value})."
                )
            )
        if call.name in self.policy.blocked_tools:
            return ActionResult(error=f"Tool '{call.name}' is blocked by policy list.")

        handler = self.registry.get(call.name)
        if handler is None:
            return ActionResult(error=f"Action '{call.name}' is not allowed.")
        category_dispatcher = self._category_dispatchers.get(definition.category)
        if category_dispatcher is None:
            return ActionResult(error=f"Unsupported tool category for '{call.name}'.")
        return self._dispatch_with_transient_retries(category_dispatcher, handler, call)

    def _dispatch_filesystem(self, handler, call: ToolCall) -> ActionResult:
        return handler(call.argument)

    def _dispatch_application(self, handler, call: ToolCall) -> ActionResult:
        return handler(call.argument)

    def _dispatch_system(self, handler, call: ToolCall) -> ActionResult:
        return handler(call.argument)

    def _dispatch_with_transient_retries(self, dispatcher, handler, call: ToolCall) -> ActionResult:
        retries = max(0, self.policy.max_transient_retries)
        last_error: Optional[str] = None

        for attempt in range(retries + 1):
            try:
                result = dispatcher(handler, call)
            except OSError as exc:
                last_error = f"Transient execution failure for '{call.name}': {exc}"
                if attempt < retries and self._is_transient_error(last_error):
                    continue
                return ActionResult(error=last_error)

            if result.error and attempt < retries and self._is_transient_error(result.error):
                last_error = result.error
                continue
            return result

        return ActionResult(error=last_error or f"Transient retries exhausted for '{call.name}'.")

    @staticmethod
    def _is_transient_error(error: str) -> bool:
        normalized = error.lower()
        transient_markers = (
            "temporar",
            "transient",
            "timeout",
            "timed out",
            "try again",
            "resource busy",
            "eagain",
            "unavailable",
        )
        return any(marker in normalized for marker in transient_markers)

    def _is_allowed_by_policy(self, safety_class: SafetyClass) -> bool:
        if safety_class == SafetyClass.SAFE_READ:
            return self.policy.allow_safe_read
        if safety_class == SafetyClass.LAUNCH_ONLY:
            return self.policy.allow_launch_only
        if safety_class == SafetyClass.CONFIRM_REQUIRED:
            return self.policy.allow_confirm_required
        if safety_class == SafetyClass.BLOCKED:
            return self.policy.allow_blocked
        return False
