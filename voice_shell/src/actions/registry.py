import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
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


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[truncated]"


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
    return ActionResult(
        stdout=_truncate(result.stdout.strip()),
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
    return ActionResult(
        stdout=_truncate(result.stdout.strip()),
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


def _action_search_files(argument: str) -> ActionResult:
    """Search for files by name under a root directory.

    Argument formats:
    - ``query``
    - ``query|/path/to/root``
    """
    raw = (argument or "").strip()
    if not raw:
        return ActionResult(error="search_files requires a query.")

    if "|" in raw:
        query, root = raw.split("|", 1)
    else:
        query, root = raw, "."

    query = query.strip()
    root = root.strip() or "."
    if not query:
        return ActionResult(error="search_files requires a non-empty query.")

    target, error = _resolve_existing_path(root)
    if error:
        return ActionResult(error=error)
    if target is None:
        return ActionResult(error="Path validation failed for search_files.")
    if not os.path.isdir(target):
        return ActionResult(error=f"Not a directory: {target}")

    matches: List[str] = []
    root_path = Path(target)
    query_lower = query.lower()
    try:
        for path in root_path.rglob("*"):
            if query_lower in path.name.lower():
                matches.append(str(path))
                if len(matches) >= 50:
                    break
    except OSError as exc:
        return ActionResult(error=f"Search failed under {target}: {exc}")

    if not matches:
        return ActionResult(stdout=f"No matches for '{query}' under {target}")
    return ActionResult(stdout=_truncate("\n".join(matches)))


def _action_get_battery_status(_: str = "") -> ActionResult:
    """Return battery status from sysfs when available."""
    power_supply = Path("/sys/class/power_supply")
    if not power_supply.exists():
        return ActionResult(error="Battery information is not available on this system.")

    batteries = sorted(
        entry for entry in power_supply.iterdir()
        if entry.is_dir() and (
            (entry / "type").exists() and (entry / "type").read_text(encoding="utf-8").strip() == "Battery"
            or entry.name.lower().startswith("bat")
        )
    )
    if not batteries:
        return ActionResult(error="No battery device found.")

    battery = batteries[0]
    capacity_path = battery / "capacity"
    status_path = battery / "status"

    parts: List[str] = []
    try:
        if capacity_path.exists():
            capacity = capacity_path.read_text(encoding="utf-8").strip()
            parts.append(f"{capacity}%")
        if status_path.exists():
            status = status_path.read_text(encoding="utf-8").strip()
            parts.append(status)
    except OSError as exc:
        return ActionResult(error=f"Unable to read battery status: {exc}")

    if not parts:
        return ActionResult(error="Battery status files are incomplete.")
    return ActionResult(stdout=" ".join(parts))


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

    DEFAULT_SHELL_COMMANDS = [
        "ls",
        "cat",
        "pwd",
        "date",
        "cd",
        "time",
        "list_directory",
        "read_file",
        "search_files",
        "get_battery_status",
    ]

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
            allowed_shell_commands
            if allowed_shell_commands is not None
            else list(self.DEFAULT_SHELL_COMMANDS)
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
        if "list_directory" in self.allowed_shell_commands:
            self._registry["list_directory"] = _action_ls
        if "cat" in self.allowed_shell_commands:
            self._registry["cat"] = _action_cat
        if "read_file" in self.allowed_shell_commands:
            self._registry["read_file"] = _action_cat
        if "search_files" in self.allowed_shell_commands:
            self._registry["search_files"] = _action_search_files
        if "cd" in self.allowed_shell_commands:
            self._registry["cd"] = _action_cd
        if "pwd" in self.allowed_shell_commands:
            self._registry["pwd"] = _action_pwd
        if "date" in self.allowed_shell_commands:
            self._registry["date"] = _action_date
        if "time" in self.allowed_shell_commands:
            self._registry["time"] = _action_time
        if "get_battery_status" in self.allowed_shell_commands:
            self._registry["get_battery_status"] = _action_get_battery_status

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
