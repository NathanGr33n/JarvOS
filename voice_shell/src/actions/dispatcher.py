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
        if definition.category == ToolCategory.FILESYSTEM:
            return self._dispatch_filesystem(handler, call)
        if definition.category == ToolCategory.APPLICATION:
            return self._dispatch_application(handler, call)
        if definition.category == ToolCategory.SYSTEM:
            return self._dispatch_system(handler, call)

        return ActionResult(error=f"Unsupported tool category for '{call.name}'.")

    def _dispatch_filesystem(self, handler, call: ToolCall) -> ActionResult:
        return handler(call.argument)

    def _dispatch_application(self, handler, call: ToolCall) -> ActionResult:
        return handler(call.argument)

    def _dispatch_system(self, handler, call: ToolCall) -> ActionResult:
        return handler(call.argument)

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
