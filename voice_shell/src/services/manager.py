"""Process supervision helpers for local JarvOS engine services."""

from __future__ import annotations

import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Lifecycle status for a supervised service."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"


@dataclass
class ServiceDefinition:
    """Definition of a supervised local process."""

    name: str
    command: List[str]
    health_url: Optional[str] = None
    ready_timeout: float = 30.0
    poll_interval: float = 0.5
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class ManagedService:
    """Runtime state for one supervised service."""

    definition: ServiceDefinition
    process: Optional[subprocess.Popen] = None
    status: ServiceStatus = ServiceStatus.STOPPED
    last_error: Optional[str] = None


class ServiceManager:
    """Start, stop, and health-check configured local engine processes.

    Commands are launched with ``subprocess.Popen`` and never through a shell.
    """

    def __init__(self, services: Optional[List[ServiceDefinition]] = None):
        self._services: Dict[str, ManagedService] = {}
        for definition in services or []:
            self.register(definition)

    @classmethod
    def from_config(cls, config: "Config") -> "ServiceManager":
        """Build a manager from application config."""
        services_cfg = config.services
        definitions: List[ServiceDefinition] = []

        if services_cfg.manage_whisper and services_cfg.whisper_command:
            definitions.append(
                ServiceDefinition(
                    name="whisper",
                    command=list(services_cfg.whisper_command),
                    health_url=services_cfg.whisper_health_url or f"{config.stt.base_url.rstrip('/')}/",
                    ready_timeout=services_cfg.ready_timeout,
                )
            )

        if services_cfg.manage_llama and services_cfg.llama_command:
            definitions.append(
                ServiceDefinition(
                    name="llama",
                    command=list(services_cfg.llama_command),
                    health_url=services_cfg.llama_health_url or f"{config.llm.base_url.rstrip('/')}/health",
                    ready_timeout=services_cfg.ready_timeout,
                )
            )

        return cls(definitions)

    @property
    def services(self) -> Dict[str, ServiceDefinition]:
        """Return registered service definitions keyed by name."""
        return {name: managed.definition for name, managed in self._services.items()}

    def register(self, definition: ServiceDefinition) -> None:
        """Register or replace a service definition."""
        if not definition.command:
            raise ValueError(f"Service '{definition.name}' requires a non-empty command.")
        if definition.name in self._services and self.is_running(definition.name):
            raise RuntimeError(f"Service '{definition.name}' is running; stop it before re-registering.")
        self._services[definition.name] = ManagedService(definition=definition)

    def get_status(self, name: str) -> ServiceStatus:
        managed = self._require(name)
        if managed.process is not None and managed.process.poll() is not None:
            if managed.status == ServiceStatus.RUNNING:
                managed.status = ServiceStatus.FAILED
                managed.last_error = f"Process exited with code {managed.process.returncode}"
            managed.process = None
        return managed.status

    def is_running(self, name: str) -> bool:
        return self.get_status(name) in {ServiceStatus.RUNNING, ServiceStatus.STARTING, ServiceStatus.UNHEALTHY}

    def start(self, name: str, wait_healthy: bool = True) -> ServiceStatus:
        """Start a registered service."""
        managed = self._require(name)
        if managed.process is not None and managed.process.poll() is None:
            return managed.status

        definition = managed.definition
        try:
            managed.process = subprocess.Popen(
                definition.command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=None if not definition.env else {**os.environ, **definition.env},
            )
        except OSError as exc:
            managed.status = ServiceStatus.FAILED
            managed.last_error = str(exc)
            logger.error("Failed to start service %s: %s", name, exc)
            return managed.status

        managed.status = ServiceStatus.STARTING
        managed.last_error = None
        logger.info("Started service %s (pid=%s)", name, managed.process.pid)

        if wait_healthy:
            return self.wait_until_healthy(name)
        return managed.status

    def stop(self, name: str, timeout: float = 5.0) -> ServiceStatus:
        """Stop a supervised service if it is running."""
        managed = self._require(name)
        process = managed.process
        if process is None:
            managed.status = ServiceStatus.STOPPED
            return managed.status

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)

        managed.process = None
        managed.status = ServiceStatus.STOPPED
        managed.last_error = None
        logger.info("Stopped service %s", name)
        return managed.status

    def start_all(self, wait_healthy: bool = True) -> Dict[str, ServiceStatus]:
        return {name: self.start(name, wait_healthy=wait_healthy) for name in self._services}

    def stop_all(self) -> Dict[str, ServiceStatus]:
        return {name: self.stop(name) for name in list(self._services.keys())}

    def check_health(self, name: str) -> bool:
        """Return True when the service process is alive and optional health URL succeeds."""
        managed = self._require(name)
        process = managed.process
        if process is not None and process.poll() is not None:
            managed.status = ServiceStatus.FAILED
            managed.last_error = f"Process exited with code {process.returncode}"
            managed.process = None
            return False

        health_url = managed.definition.health_url
        if not health_url:
            healthy = process is not None and process.poll() is None
            managed.status = ServiceStatus.RUNNING if healthy else ServiceStatus.STOPPED
            return healthy

        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                healthy = 200 <= getattr(response, "status", 200) < 500
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            managed.last_error = str(exc)
            healthy = False

        if healthy:
            managed.status = ServiceStatus.RUNNING
            managed.last_error = None
        elif process is not None and process.poll() is None:
            managed.status = ServiceStatus.UNHEALTHY
        else:
            managed.status = ServiceStatus.STOPPED
        return healthy

    def wait_until_healthy(self, name: str) -> ServiceStatus:
        """Poll health until ready timeout expires."""
        managed = self._require(name)
        deadline = time.monotonic() + managed.definition.ready_timeout
        while time.monotonic() < deadline:
            if self.check_health(name):
                return managed.status
            if managed.process is not None and managed.process.poll() is not None:
                managed.status = ServiceStatus.FAILED
                managed.last_error = f"Process exited with code {managed.process.returncode}"
                managed.process = None
                return managed.status
            time.sleep(managed.definition.poll_interval)

        if managed.process is not None and managed.process.poll() is None:
            managed.status = ServiceStatus.UNHEALTHY
            managed.last_error = managed.last_error or "Timed out waiting for healthy status"
        else:
            managed.status = ServiceStatus.FAILED
            managed.last_error = managed.last_error or "Service failed during startup"
        return managed.status

    def _require(self, name: str) -> ManagedService:
        if name not in self._services:
            raise KeyError(f"Unknown service '{name}'")
        return self._services[name]
