from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional
from pathlib import Path


try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required for configuration loading. "
        "Install it with: pip install pyyaml"
    ) from exc


@dataclass
class AudioConfig:
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    sample_rate_input: int = 16000
    sample_rate_output: int = 22050
    chunk_size: int = 512
    max_recording_duration: float = 30.0


@dataclass
class WakeWordConfig:
    engine: str = "hotkey"
    model_path: Optional[str] = None
    keyword: str = "Hey Nova"
    sensitivity: float = 0.5


@dataclass
class STTConfig:
    engine: str = "whisper.cpp"
    base_url: str = "http://127.0.0.1:8081"
    language: str = "en"
    translate: bool = False


@dataclass
class LLMConfig:
    engine: str = "llama.cpp"
    base_url: str = "http://127.0.0.1:8082"
    max_tokens: int = 512
    temperature: float = 0.7
    system_prompt: Optional[str] = None


@dataclass
class TTSConfig:
    engine: str = "piper"
    model_path: str = "models/piper/en_US-lessac-medium.onnx"
    speaker_id: int = 0
    binary_path: str = "piper"


@dataclass
class ActionsConfig:
    allowed_shell_commands: list = field(
        default_factory=lambda: [
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
    )
    allowed_apps: list = field(default_factory=lambda: ["firefox", "nautilus", "code", "terminal"])
    # When True, gated tools wait for a spoken yes/no before running.
    require_confirmation: bool = True
    # Max seconds to listen for a confirmation reply after prompting.
    confirmation_timeout: float = 8.0
    # When True, a yes for one gated tool also grants other pending gated tools
    # in the same turn. When False, each gated tool is confirmed separately.
    confirm_batch: bool = True
    # Append-only SQLite log of executed/cancelled actions.
    audit_enabled: bool = True
    audit_db_path: str = "data/action_audit.db"

@dataclass
class HUDConfig:
    enabled: bool = True
    show_timestamps: bool = True
    # text | floating | both
    mode: str = "both"
    width: int = 480
    height: int = 140
    anchor: str = "top-center"
    margin: int = 24
    opacity: float = 0.92


@dataclass
class MemoryConfig:
    enabled: bool = True
    db_path: str = "data/memory.db"
    history_limit: int = 3


@dataclass
class HealthConfig:
    """Startup and pre-listen engine readiness gates."""
    enabled: bool = True
    require_stt: bool = True
    require_llm: bool = True
    require_tts: bool = True
    startup_timeout: float = 60.0
    poll_interval: float = 1.0
    # If True, refuse IDLE->LISTENING while required engines are down.
    block_listening: bool = True
    # If True, stop the orchestrator when startup health wait times out.
    fail_fast: bool = False


@dataclass
class ServicesConfig:
    """Optional local process supervision for engine binaries."""
    manage_whisper: bool = False
    manage_llama: bool = False
    ready_timeout: float = 30.0
    whisper_command: list = field(default_factory=list)
    llama_command: list = field(default_factory=list)
    whisper_health_url: Optional[str] = None
    llama_health_url: Optional[str] = None
    autostart: bool = False


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    hud: HUDConfig = field(default_factory=HUDConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    health: HealthConfig = field(default_factory=HealthConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load a configuration from a YAML file.

        Supports nested sections (e.g., ``audio: {sample_rate_input: 16000}``).
        Unknown keys are ignored for forward compatibility.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A populated ``Config`` instance.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        def _from_dict(data: Dict[str, Any], section_class: type) -> Any:
            known = {f.name for f in fields(section_class)}
            filtered = {k: v for k, v in data.items() if k in known}
            return section_class(**filtered)

        return cls(
            audio=_from_dict(raw.get("audio", {}), AudioConfig),
            wake_word=_from_dict(raw.get("wake_word", {}), WakeWordConfig),
            stt=_from_dict(raw.get("stt", {}), STTConfig),
            llm=_from_dict(raw.get("llm", {}), LLMConfig),
            tts=_from_dict(raw.get("tts", {}), TTSConfig),
            actions=_from_dict(raw.get("actions", {}), ActionsConfig),
            hud=_from_dict(raw.get("hud", {}), HUDConfig),
            memory=_from_dict(raw.get("memory", {}), MemoryConfig),
            services=_from_dict(raw.get("services", {}), ServicesConfig),
            health=_from_dict(raw.get("health", {}), HealthConfig),
        )
