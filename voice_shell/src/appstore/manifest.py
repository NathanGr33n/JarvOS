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
    # LLM GGUF weights for llama.cpp (hardware tiers from AI_First_Voice_OS.md).
    # Official Qwen 7B Q4_K_M is split across two shards; the bartowski single-file
    # build is catalogued instead so the existing one-URL downloader can fetch it.
    "qwen2.5-1.5b-instruct-q4": ModelEntry(
        name="qwen2.5-1.5b-instruct-q4",
        category="llm",
        description="Qwen2.5-1.5B-Instruct Q4_K_M GGUF (~1.0GB) - Raspberry Pi 5 tier.",
        url=(
            "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/"
            "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        ),
        dest_path="llm/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        size_bytes=1_117_320_736,
    ),
    "qwen2.5-3b-instruct-q4": ModelEntry(
        name="qwen2.5-3b-instruct-q4",
        category="llm",
        description="Qwen2.5-3B-Instruct Q4_K_M GGUF (~2.0GB) - light laptop tier.",
        url=(
            "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
            "qwen2.5-3b-instruct-q4_k_m.gguf"
        ),
        dest_path="llm/qwen2.5-3b-instruct-q4_k_m.gguf",
        size_bytes=2_104_932_768,
    ),
    "llama-3.2-3b-instruct-q4": ModelEntry(
        name="llama-3.2-3b-instruct-q4",
        category="llm",
        description="Llama-3.2-3B-Instruct Q4_K_M GGUF (~1.9GB) - budget laptop tier.",
        url=(
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/"
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        ),
        dest_path="llm/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        size_bytes=2_019_377_696,
    ),
    "qwen2.5-7b-instruct-q4": ModelEntry(
        name="qwen2.5-7b-instruct-q4",
        category="llm",
        description=(
            "Qwen2.5-7B-Instruct Q4_K_M GGUF (~4.4GB) - budget laptop tier "
            "(single-file bartowski build; official Qwen repo ships split shards)."
        ),
        url=(
            "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/"
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        ),
        dest_path="llm/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        size_bytes=4_683_074_240,
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
