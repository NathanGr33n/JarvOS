"""Local "app store" downloader for pre-quantized JarvOS models.

Downloads are restricted to URLs declared in :mod:`manifest`. Integrity is
enforced with a trust-on-first-download (TOFU) model: the SHA-256 of the
first successful download is recorded in a local lockfile
(``<root>/.checksums.json``) and every subsequent download or status check
is verified against that pinned value, so silent corruption or tampering
after the first trusted fetch is detected.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .manifest import ModelEntry, get_model, list_models

_CHECKSUM_FILE = ".checksums.json"
_CHUNK_SIZE = 1024 * 1024


class DownloadError(Exception):
    """Raised when a model download or verification fails."""


@dataclass(frozen=True)
class ModelStatus:
    """Local status of a catalog entry."""

    entry: ModelEntry
    downloaded: bool
    verified: bool
    path: Path
    detail: str = ""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_lockfile(root: Path) -> Dict[str, str]:
    lock_path = root / _CHECKSUM_FILE
    if not lock_path.exists():
        return {}
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_lockfile(root: Path, data: Dict[str, str]) -> None:
    lock_path = root / _CHECKSUM_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def get_status(name: str, root: Path) -> ModelStatus:
    """Return the local download/verification status for a catalog entry."""
    entry = get_model(name)
    if entry is None:
        raise DownloadError(f"Unknown model '{name}'.")

    path = root / entry.dest_path
    if not path.exists():
        return ModelStatus(entry=entry, downloaded=False, verified=False, path=path)

    pinned = _load_lockfile(root).get(entry.name)
    if pinned is None:
        return ModelStatus(
            entry=entry,
            downloaded=True,
            verified=False,
            path=path,
            detail="File present but no pinned checksum recorded yet.",
        )

    actual = _hash_file(path)
    if actual == pinned:
        return ModelStatus(entry=entry, downloaded=True, verified=True, path=path)
    return ModelStatus(
        entry=entry,
        downloaded=True,
        verified=False,
        path=path,
        detail="Checksum mismatch against pinned value; file may be corrupted or tampered with.",
    )


def download(name: str, root: Path, force: bool = False) -> ModelStatus:
    """Download (or verify) a catalog entry into ``root``.

    On first successful download, the SHA-256 is pinned in the local
    lockfile. On subsequent calls, the existing file is verified against the
    pinned checksum instead of being re-downloaded, unless ``force`` is set.
    """
    entry = get_model(name)
    if entry is None:
        raise DownloadError(f"Unknown model '{name}'.")

    root = Path(root)
    dest = root / entry.dest_path
    lockfile = _load_lockfile(root)

    if dest.exists() and not force:
        status = get_status(name, root)
        if status.verified:
            return status
        if status.downloaded and status.detail:
            raise DownloadError(
                f"Refusing to use existing file for '{name}': {status.detail} "
                "Re-run with force=True to re-download."
            )

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(entry.url, timeout=30) as response:
            fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=f".{entry.name}.", suffix=".part")
            tmp_path = Path(tmp_name)
            try:
                with open(fd, "wb") as tmp_file:
                    while True:
                        chunk = response.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise DownloadError(f"Failed to download '{name}': {exc}") from exc

    digest = _hash_file(tmp_path)
    pinned = lockfile.get(entry.name)
    if pinned is not None and digest != pinned:
        tmp_path.unlink(missing_ok=True)
        raise DownloadError(
            f"Checksum mismatch for '{name}': expected pinned {pinned}, got {digest}. "
            "Download aborted; upstream file may have changed or been tampered with."
        )

    tmp_path.replace(dest)
    lockfile[entry.name] = digest
    _save_lockfile(root, lockfile)

    return ModelStatus(entry=entry, downloaded=True, verified=True, path=dest)


def status_all(root: Path, category: Optional[str] = None) -> List[ModelStatus]:
    """Return status for every catalog entry, optionally filtered by category."""
    return [get_status(entry.name, root) for entry in list_models(category)]


def clear_pin(name: str, root: Path) -> None:
    """Remove a model's pinned checksum, allowing a new value to be pinned.

    This is a deliberate, explicit action distinct from ``force`` re-download,
    so an untrusted mirror or a single ``force=True`` call can never silently
    replace a previously trusted checksum.
    """
    entry = get_model(name)
    if entry is None:
        raise DownloadError(f"Unknown model '{name}'.")
    root = Path(root)
    lockfile = _load_lockfile(root)
    if entry.name in lockfile:
        del lockfile[entry.name]
        _save_lockfile(root, lockfile)
