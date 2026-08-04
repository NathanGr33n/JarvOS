"""Engine readiness checks used before the voice loop accepts input."""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

HealthCheck = Callable[[], Awaitable[bool]]


@dataclass
class EngineHealth:
    """Snapshot of one engine's readiness."""

    name: str
    healthy: bool
    required: bool = True
    detail: str = ""


@dataclass
class HealthReport:
    """Aggregate readiness across engines."""

    engines: List[EngineHealth] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(e.healthy for e in self.engines if e.required)

    @property
    def required_names(self) -> List[str]:
        return [e.name for e in self.engines if e.required]

    def unhealthy_required(self) -> List[EngineHealth]:
        return [e for e in self.engines if e.required and not e.healthy]

    def summary(self) -> str:
        parts = []
        for engine in self.engines:
            flag = "ok" if engine.healthy else "down"
            req = "required" if engine.required else "optional"
            detail = f" ({engine.detail})" if engine.detail else ""
            parts.append(f"{engine.name}={flag}/{req}{detail}")
        return ", ".join(parts) if parts else "no engines configured"

    def to_dict(self) -> dict:
        """Serialize the report for JSON/CLI dashboards."""
        return {
            "ready": self.ready,
            "engines": [
                {
                    "name": e.name,
                    "healthy": e.healthy,
                    "required": e.required,
                    "detail": e.detail,
                    "status": "ok" if e.healthy else "down",
                }
                for e in self.engines
            ],
            "unhealthy_required": [e.name for e in self.unhealthy_required()],
            "summary": self.summary(),
        }


class EngineHealthGate:
    """Poll STT/LLM/TTS readiness until healthy or timeout."""

    def __init__(
        self,
        checks: Dict[str, HealthCheck],
        required: Optional[Sequence[str]] = None,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
    ):
        self.checks = dict(checks)
        self.required = set(required if required is not None else checks.keys())
        self.timeout = max(0.0, float(timeout))
        self.poll_interval = max(0.05, float(poll_interval))

    async def check_once(self) -> HealthReport:
        """Run all registered checks once."""
        engines: List[EngineHealth] = []
        for name, check in self.checks.items():
            detail = ""
            try:
                healthy = bool(await check())
            except Exception as exc:  # pragma: no cover - defensive
                healthy = False
                detail = str(exc)
            engines.append(
                EngineHealth(
                    name=name,
                    healthy=healthy,
                    required=name in self.required,
                    detail=detail,
                )
            )
        return HealthReport(engines=engines)

    async def wait_until_ready(self) -> HealthReport:
        """Poll until required engines are healthy or timeout expires.

        Returns the last health report (ready or not).
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        last = await self.check_once()
        if last.ready or self.timeout <= 0:
            return last

        while loop.time() < deadline:
            await asyncio.sleep(self.poll_interval)
            last = await self.check_once()
            if last.ready:
                return last
            logger.info("Waiting for engines: %s", last.summary())
        return last


def build_default_health_gate(
    stt,
    llm,
    tts,
    *,
    require_stt: bool = True,
    require_llm: bool = True,
    require_tts: bool = True,
    timeout: float = 60.0,
    poll_interval: float = 1.0,
) -> EngineHealthGate:
    """Construct a gate for the standard voice_shell engine clients."""

    async def _tts_check() -> bool:
        checker = getattr(tts, "health_check", None)
        if checker is None:
            binary = Path(getattr(tts, "binary_path", "piper"))
            model = Path(getattr(tts, "model_path", ""))
            binary_ok = binary.exists() or shutil.which(str(binary)) is not None
            model_ok = model.exists() if str(model) else False
            return bool(binary_ok and model_ok)
        result = checker()
        if asyncio.iscoroutine(result):
            return bool(await result)
        return bool(result)

    checks: Dict[str, HealthCheck] = {
        "stt": stt.health_check,
        "llm": llm.health_check,
        "tts": _tts_check,
    }
    required = []
    if require_stt:
        required.append("stt")
    if require_llm:
        required.append("llm")
    if require_tts:
        required.append("tts")
    return EngineHealthGate(
        checks=checks,
        required=required,
        timeout=timeout,
        poll_interval=poll_interval,
    )
