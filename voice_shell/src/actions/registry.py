import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class ActionResult:
    """Result of executing an action."""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    error: Optional[str] = None


def _resolve_existing_path(path: str) -> tuple[Optional[str], Optional[str]]:
    """Return an absolute normalized path if it exists, otherwise an error."""
    candidate = path.strip() if path else "."
    expanded = os.path.expanduser(candidate)
    normalized = os.path.abspath(expanded)
    if not os.path.exists(normalized):
        return None, f"Path not found: {candidate}"
    return normalized, None


def _action_ls(path: str = "") -> ActionResult:
    """Execute ``ls -la`` in the current or requested directory."""
    target, error = _resolve_existing_path(path)
    if error:
        return ActionResult(error=error)
    if target is None:
        return ActionResult(error="Path validation failed for ls.")
    if not os.path.isdir(target):
        return ActionResult(error=f"Not a directory: {target}")
    result = subprocess.run(
        ["ls", "-la", target],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = result.stdout.strip()
    if len(stdout) > 4000:
        stdout = f"{stdout[:4000]}\n...[truncated]"
    return ActionResult(
        stdout=stdout,
        stderr=result.stderr.strip(),
        returncode=result.returncode,
    )


def _action_cat(filepath: str) -> ActionResult:
    """Read the contents of a file using ``cat``."""
    target, error = _resolve_existing_path(filepath)
    if error:
        return ActionResult(error=error)
    if target is None:
        return ActionResult(error="Path validation failed for cat.")
    if not os.path.isfile(target):
        return ActionResult(error=f"Not a file: {target}")
    result = subprocess.run(
        ["cat", target],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = result.stdout.strip()
    if len(stdout) > 4000:
        stdout = f"{stdout[:4000]}\n...[truncated]"
    return ActionResult(
        stdout=stdout,
        stderr=result.stderr.strip(),
        returncode=result.returncode,
    )


def _action_cd(path: str) -> ActionResult:
    """Change the current working directory."""
    target, error = _resolve_existing_path(path)
    if error:
        return ActionResult(error=error)
    if target is None:
        return ActionResult(error="Path validation failed for cd.")
    if not os.path.isdir(target):
        return ActionResult(error=f"Not a directory: {target}")
    try:
        os.chdir(target)
        return ActionResult(stdout=os.getcwd())
    except OSError as exc:
        return ActionResult(error=f"Cannot change directory to {target}: {exc}")

def _action_pwd(_: str = "") -> ActionResult:
    """Return the current working directory."""
    return ActionResult(stdout=os.getcwd())


def _action_time(_: str = "") -> ActionResult:
    """Return the current time."""
    import datetime
    return ActionResult(stdout=datetime.datetime.now().strftime("%I:%M %p"))


def _action_date(_: str = "") -> ActionResult:
    """Return the current date."""
    import datetime
    return ActionResult(
        stdout=datetime.datetime.now().strftime("%A, %B %d, %Y")
    )


def _action_launch_app(name: str) -> ActionResult:
    """Launch an application by name."""
    try:
        subprocess.Popen([name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ActionResult(stdout=f"Launched {name}")
    except OSError as exc:
        return ActionResult(error=f"Cannot launch {name}: {exc}")


class ActionRegistry:
    """Registry of whitelisted, safe actions that can be executed by the orchestrator.

    Each action is a callable that receives a single string argument and returns an
    ``ActionResult``. The registry validates the action name before execution.
    """

    def __init__(
        self,
        allowed_shell_commands: Optional[List[str]] = None,
        allowed_apps: Optional[List[str]] = None,
    ):
        """Initialize the action registry.

        Args:
            allowed_shell_commands: List of permitted shell command names (e.g., ``["ls", "cat"]``).
            allowed_apps: List of permitted application names (e.g., ``["firefox"]``).
        """
        self.allowed_shell_commands = set(
            allowed_shell_commands if allowed_shell_commands is not None else ["ls", "cat", "pwd", "date", "cd", "time"]
        )
        self.allowed_apps = set(
            allowed_apps if allowed_apps is not None else ["firefox", "nautilus", "code", "terminal"]
        )

        self._registry: Dict[str, Callable[[str], ActionResult]] = {}
        self._build_registry()

    def _build_registry(self) -> None:
        """Populate the internal registry with safe, whitelisted actions."""
        if "ls" in self.allowed_shell_commands:
            self._registry["ls"] = _action_ls
        if "cat" in self.allowed_shell_commands:
            self._registry["cat"] = _action_cat
        if "cd" in self.allowed_shell_commands:
            self._registry["cd"] = _action_cd
        if "pwd" in self.allowed_shell_commands:
            self._registry["pwd"] = _action_pwd
        if "date" in self.allowed_shell_commands:
            self._registry["date"] = _action_date
        if "time" in self.allowed_shell_commands:
            self._registry["time"] = _action_time

        for app in self.allowed_apps:
            self._registry[f"app:{app}"] = self._make_app_launcher(app)

    def get(self, name: str) -> Optional[Callable[[str], ActionResult]]:
        """Retrieve an action by name if it is whitelisted.

        Args:
            name: The action name (e.g., ``"ls"``, ``"cat"``, ``"app:firefox"``).

        Returns:
            The action callable, or ``None`` if the action is not whitelisted.
        """
        return self._registry.get(name)

    def is_allowed(self, name: str) -> bool:
        """Check whether an action name is in the whitelist."""
        return name in self._registry

    @staticmethod
    def _make_app_launcher(app_name: str):
        """Create a closure that launches a specific app, ignoring any argument."""
        def _launcher(_: str = "") -> ActionResult:
            return _action_launch_app(app_name)
        return _launcher

    def list_actions(self) -> List[str]:
        """Return a sorted list of all registered action names."""
        return sorted(self._registry.keys())
