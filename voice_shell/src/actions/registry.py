import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class ActionResult:
    """Result of executing an action."""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    error: Optional[str] = None


def _normalize_path(path: str, default: str = ".") -> str:
    """Expand and absolutize a user-supplied path without requiring existence."""
    candidate = path.strip() if path else default
    return os.path.abspath(os.path.expanduser(candidate))


def _resolve_existing_path(path: str) -> tuple[Optional[str], Optional[str]]:
    """Return an absolute normalized path if it exists, otherwise an error."""
    candidate = path.strip() if path else "."
    normalized = _normalize_path(candidate)
    if not os.path.exists(normalized):
        return None, f"Path not found: {candidate}"
    return normalized, None


def _split_two_args(argument: str, separator: str = "|") -> Tuple[str, str]:
    """Split ``left|right`` style arguments; empty parts become empty strings."""
    raw = (argument or "").strip()
    if separator not in raw:
        return raw, ""
    left, right = raw.split(separator, 1)
    return left.strip(), right.strip()


def _parse_level(argument: str) -> tuple[Optional[int], Optional[str]]:
    """Parse a 0-100 integer level from a bare number or ``N%`` string."""
    raw = (argument or "").strip().rstrip("%")
    if not raw:
        return None, "Level is required (0-100)."
    try:
        value = int(raw)
    except ValueError:
        return None, f"Invalid level '{argument}'; expected an integer 0-100."
    if value < 0 or value > 100:
        return None, f"Level out of range: {value}. Expected 0-100."
    return value, None


def _run_command(cmd: List[str], timeout: float = 5.0) -> ActionResult:
    """Run an argv list without a shell and return a truncated result."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError:
        return ActionResult(error=f"Command not found: {cmd[0]}")
    except subprocess.TimeoutExpired:
        return ActionResult(error=f"Command timed out: {' '.join(cmd)}")
    except OSError as exc:
        return ActionResult(error=f"Failed to run {cmd[0]}: {exc}")
    return ActionResult(
        stdout=_truncate(result.stdout.strip()),
        stderr=result.stderr.strip(),
        returncode=result.returncode,
    )


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


def _action_write_file(argument: str) -> ActionResult:
    """Write text content to a file.

    Argument format: ``path|content``
    """
    path_part, content = _split_two_args(argument)
    if not path_part:
        return ActionResult(error="write_file requires 'path|content'.")
    if content == "" and "|" not in (argument or ""):
        return ActionResult(error="write_file requires 'path|content'.")

    target = _normalize_path(path_part)
    parent = os.path.dirname(target) or "."
    if not os.path.isdir(parent):
        return ActionResult(error=f"Parent directory not found: {parent}")
    if os.path.isdir(target):
        return ActionResult(error=f"Refusing to overwrite directory: {target}")

    try:
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        return ActionResult(error=f"Cannot write {target}: {exc}")
    return ActionResult(stdout=f"Wrote {len(content)} bytes to {target}")


def _action_move_file(argument: str) -> ActionResult:
    """Move or rename a file/directory.

    Argument format: ``src|dest``
    """
    src_part, dest_part = _split_two_args(argument)
    if not src_part or not dest_part:
        return ActionResult(error="move_file requires 'src|dest'.")

    src = _normalize_path(src_part)
    dest = _normalize_path(dest_part)
    if not os.path.exists(src):
        return ActionResult(error=f"Path not found: {src_part}")
    if os.path.exists(dest):
        return ActionResult(error=f"Destination already exists: {dest}")
    parent = os.path.dirname(dest) or "."
    if not os.path.isdir(parent):
        return ActionResult(error=f"Destination parent not found: {parent}")

    try:
        shutil.move(src, dest)
    except OSError as exc:
        return ActionResult(error=f"Cannot move {src} to {dest}: {exc}")
    return ActionResult(stdout=f"Moved {src} -> {dest}")


def _action_set_volume(argument: str) -> ActionResult:
    """Set the default audio sink volume to 0-100 percent via pactl or amixer."""
    level, error = _parse_level(argument)
    if error:
        return ActionResult(error=error)
    assert level is not None

    if shutil.which("pactl"):
        result = _run_command(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"])
        if result.error or result.returncode != 0:
            detail = result.error or result.stderr or "pactl failed"
            return ActionResult(error=f"Failed to set volume: {detail}", returncode=result.returncode)
        return ActionResult(stdout=f"Volume set to {level}%")

    if shutil.which("amixer"):
        result = _run_command(["amixer", "-q", "sset", "Master", f"{level}%"])
        if result.error or result.returncode != 0:
            detail = result.error or result.stderr or "amixer failed"
            return ActionResult(error=f"Failed to set volume: {detail}", returncode=result.returncode)
        return ActionResult(stdout=f"Volume set to {level}%")

    return ActionResult(error="No supported volume control found (pactl/amixer).")


def _action_set_brightness(argument: str) -> ActionResult:
    """Set display brightness to 0-100 percent via brightnessctl or sysfs."""
    level, error = _parse_level(argument)
    if error:
        return ActionResult(error=error)
    assert level is not None

    if shutil.which("brightnessctl"):
        result = _run_command(["brightnessctl", "set", f"{level}%"])
        if result.error or result.returncode != 0:
            detail = result.error or result.stderr or "brightnessctl failed"
            return ActionResult(error=f"Failed to set brightness: {detail}", returncode=result.returncode)
        return ActionResult(stdout=f"Brightness set to {level}%")

    backlight_root = Path("/sys/class/backlight")
    if backlight_root.exists():
        devices = sorted(entry for entry in backlight_root.iterdir() if entry.is_dir())
        if devices:
            device = devices[0]
            max_path = device / "max_brightness"
            cur_path = device / "brightness"
            try:
                max_value = int(max_path.read_text(encoding="utf-8").strip())
                if max_value <= 0:
                    return ActionResult(error="Invalid max brightness value.")
                target = max(0, min(max_value, round(max_value * (level / 100.0))))
                cur_path.write_text(str(target), encoding="utf-8")
                return ActionResult(stdout=f"Brightness set to {level}% ({target}/{max_value})")
            except OSError as exc:
                return ActionResult(error=f"Failed to set brightness via sysfs: {exc}")
            except ValueError as exc:
                return ActionResult(error=f"Invalid brightness sysfs value: {exc}")

    return ActionResult(error="No supported brightness control found (brightnessctl/sysfs).")


def _read_load_average() -> Optional[str]:
    try:
        with open("/proc/loadavg", "r", encoding="utf-8") as handle:
            parts = handle.read().strip().split()
        if len(parts) >= 3:
            return f"{parts[0]} {parts[1]} {parts[2]}"
    except OSError:
        return None
    return None


def _read_mem_available_mb() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    match = re.search(r"(\d+)", line)
                    if match:
                        return int(match.group(1)) // 1024
                if line.startswith("MemFree:"):
                    # Fallback if MemAvailable is missing.
                    match = re.search(r"(\d+)", line)
                    if match:
                        return int(match.group(1)) // 1024
    except OSError:
        return None
    return None


def _action_get_system_status(_: str = "") -> ActionResult:
    """Return a short local system status summary for prompt/context use."""
    import datetime

    parts = [
        f"time={datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"cwd={os.getcwd()}",
    ]
    load = _read_load_average()
    if load:
        parts.append(f"load={load}")
    mem = _read_mem_available_mb()
    if mem is not None:
        parts.append(f"mem_available_mb={mem}")

    battery = _action_get_battery_status()
    if battery.stdout:
        parts.append(f"battery={battery.stdout}")

    return ActionResult(stdout="; ".join(parts))


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
        "write_file",
        "move_file",
        "set_volume",
        "set_brightness",
        "get_system_status",
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
        if "write_file" in self.allowed_shell_commands:
            self._registry["write_file"] = _action_write_file
        if "move_file" in self.allowed_shell_commands:
            self._registry["move_file"] = _action_move_file
        if "set_volume" in self.allowed_shell_commands:
            self._registry["set_volume"] = _action_set_volume
        if "set_brightness" in self.allowed_shell_commands:
            self._registry["set_brightness"] = _action_set_brightness
        if "get_system_status" in self.allowed_shell_commands:
            self._registry["get_system_status"] = _action_get_system_status

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
