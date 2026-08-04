"""CLI/dashboard helpers for engine and service health status."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .engines.llm import LLMClient
from .engines.stt import STTClient
from .engines.tts import TTSClient
from .health import HealthReport, build_default_health_gate
from .services import ServiceManager, ServiceStatus


@dataclass
class ServiceSnapshot:
    name: str
    status: str
    health_url: Optional[str] = None
    command: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "health_url": self.health_url,
            "command": list(self.command),
        }


@dataclass
class DiagnosticReport:
    """Full status snapshot for operators."""

    generated_at: str
    config_path: Optional[str]
    health_enabled: bool
    engines: HealthReport
    services: List[ServiceSnapshot] = field(default_factory=list)
    endpoints: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        if not self.health_enabled:
            return True
        return self.engines.ready

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "config_path": self.config_path,
            "health_enabled": self.health_enabled,
            "ready": self.ready,
            "endpoints": dict(self.endpoints),
            "engines": self.engines.to_dict(),
            "services": [s.to_dict() for s in self.services],
            "notes": list(self.notes),
        }

    def to_text(self) -> str:
        lines = [
            "JarvOS Engine Health Status",
            f"Generated: {self.generated_at}",
        ]
        if self.config_path:
            lines.append(f"Config:    {self.config_path}")
        lines.append(f"Gates:     {'enabled' if self.health_enabled else 'disabled'}")
        lines.append(f"Overall:   {'READY' if self.ready else 'NOT READY'}")
        lines.append("")
        lines.append("Engines:")
        if not self.engines.engines:
            lines.append("  (none)")
        for engine in self.engines.engines:
            mark = "OK " if engine.healthy else "DOWN"
            req = "required" if engine.required else "optional"
            detail = f" - {engine.detail}" if engine.detail else ""
            lines.append(f"  [{mark}] {engine.name:4} ({req}){detail}")
        lines.append("")
        lines.append("Endpoints:")
        if not self.endpoints:
            lines.append("  (none)")
        for key, value in self.endpoints.items():
            lines.append(f"  - {key}: {value}")
        lines.append("")
        lines.append("Supervised services:")
        if not self.services:
            lines.append("  (none configured)")
        for svc in self.services:
            cmd = " ".join(svc.command) if svc.command else "(no command)"
            url = f" health={svc.health_url}" if svc.health_url else ""
            lines.append(f"  - {svc.name}: {svc.status}{url}")
            lines.append(f"      cmd: {cmd}")
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


def _load_config(config_path: Optional[Path]) -> tuple[Config, Optional[str]]:
    if config_path is None:
        return Config(), None
    path = Path(config_path)
    if path.exists():
        return Config.from_yaml(path), str(path)
    return Config(), str(path)


async def collect_diagnostic_report(config_path: Optional[Path] = None) -> DiagnosticReport:
    """Build engine clients from config and collect a one-shot health snapshot."""
    config, resolved = _load_config(config_path)
    health_cfg = config.health

    stt = STTClient(base_url=config.stt.base_url, language=config.stt.language, translate=config.stt.translate)
    llm = LLMClient(
        base_url=config.llm.base_url,
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
    )
    tts = TTSClient(
        model_path=Path(config.tts.model_path),
        speaker_id=config.tts.speaker_id,
        binary_path=Path(config.tts.binary_path),
    )

    notes: List[str] = []
    try:
        gate = build_default_health_gate(
            stt,
            llm,
            tts,
            require_stt=health_cfg.require_stt,
            require_llm=health_cfg.require_llm,
            require_tts=health_cfg.require_tts,
            timeout=0.0,
            poll_interval=health_cfg.poll_interval,
        )
        engines = await gate.check_once()
    finally:
        await stt.close()
        await llm.close()

    if not health_cfg.enabled:
        notes.append("Health gates are disabled in config; overall READY ignores engine failures.")

    manager = ServiceManager.from_config(config)
    services: List[ServiceSnapshot] = []
    for name, definition in manager.services.items():
        try:
            status = manager.get_status(name).value
        except Exception:
            status = ServiceStatus.STOPPED.value
        services.append(
            ServiceSnapshot(
                name=name,
                status=status,
                health_url=definition.health_url,
                command=list(definition.command),
            )
        )
    if not services:
        notes.append("No managed services configured (prefer systemd units in production).")

    endpoints = {
        "stt": config.stt.base_url,
        "llm": config.llm.base_url,
        "tts_model": str(config.tts.model_path),
        "tts_binary": str(config.tts.binary_path),
    }

    return DiagnosticReport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        config_path=resolved,
        health_enabled=bool(health_cfg.enabled),
        engines=engines,
        services=services,
        endpoints=endpoints,
        notes=notes,
    )


def format_report(report: DiagnosticReport, fmt: str = "text") -> str:
    """Render a diagnostic report as text or JSON."""
    kind = (fmt or "text").strip().lower()
    if kind == "json":
        return json.dumps(report.to_dict(), indent=2, sort_keys=True)
    return report.to_text()
