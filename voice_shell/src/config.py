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
        ]
    )
    allowed_apps: list = field(default_factory=lambda: ["firefox", "nautilus", "code", "terminal"])
    require_confirmation: bool = False

@dataclass
class HUDConfig:
    enabled: bool = True
    show_timestamps: bool = True


@dataclass
class MemoryConfig:
    enabled: bool = True
    db_path: str = "data/memory.db"
    history_limit: int = 3


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
        )
