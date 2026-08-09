"""Curated catalog of downloadable JarvOS models.

Only URLs declared here may be fetched by the downloader, keeping the local
"app store" restricted to known, official model sources (Hugging Face /
upstream project releases) instead of allowing arbitrary user-supplied URLs.

Checksum note: this catalog intentionally does NOT ship pre-filled SHA-256
values. Hardcoding a checksum here without independently verifying it
against the live upstream artifact would be worse than no checksum at all,
since it would create false confidence in an unverified value. Instead, the
downloader uses a trust-on-first-download (TOFU) model: it records the
SHA-256 of the first successful download in a local lockfile and verifies
against that pinned value on every subsequent download or status check, so
silent corruption or tampering *after* the first trusted fetch is still
detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelEntry:
    """A single downloadable model artifact."""

    name: str
    category: str  # "stt" | "llm" | "tts"
    description: str
    url: str
    dest_path: str  # relative path under the models root directory
    size_bytes: Optional[int] = None


# Catalog mirrors the hardware-profile tables in
# ResearchDesign/AI_First_Voice_OS.md and the file layout in
# ResearchDesign/Voice_Shell_PoC_Technical_Spec.md (Section 6).
MODEL_CATALOG: Dict[str, ModelEntry] = {
    "whisper-tiny": ModelEntry(
        name="whisper-tiny",
        category="stt",
        description="whisper.cpp tiny model (39M) - Raspberry Pi 5 tier.",
        url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        dest_path="whisper/ggml-tiny.bin",
    ),
    "whisper-base": ModelEntry(
        name="whisper-base",
        category="stt",
        description="whisper.cpp base model (74M) - budget laptop tier.",
        url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        dest_path="whisper/ggml-base.bin",
    ),
    "piper-lessac-medium": ModelEntry(
        name="piper-lessac-medium",
        category="tts",
        description="Piper en_US-lessac-medium voice (~50MB).",
        url=(
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        ),
        dest_path="piper/en_US-lessac-medium.onnx",
    ),
}


def get_model(name: str) -> Optional[ModelEntry]:
    """Look up a catalog entry by name."""
    return MODEL_CATALOG.get(name)


def list_models(category: Optional[str] = None) -> list[ModelEntry]:
    """Return catalog entries, optionally filtered by category."""
    entries = sorted(MODEL_CATALOG.values(), key=lambda e: e.name)
    if category is None:
        return entries
    return [e for e in entries if e.category == category]
