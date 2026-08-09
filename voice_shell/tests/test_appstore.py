import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from voice_shell.src.appstore import downloader, manifest


class FakeResponse:
    """Minimal context-manager stand-in for urllib's response object."""

    def __init__(self, payload: bytes):
        self._chunks = [payload]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def read(self, size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class TestManifest:
    def test_list_models_returns_sorted_entries(self):
        entries = manifest.list_models()
        names = [e.name for e in entries]
        assert names == sorted(names)
        assert len(entries) >= 1

    def test_list_models_filters_by_category(self):
        stt_entries = manifest.list_models(category="stt")
        assert all(e.category == "stt" for e in stt_entries)

    def test_get_model_unknown_returns_none(self):
        assert manifest.get_model("does-not-exist") is None

    def test_no_fabricated_checksums_shipped(self):
        """The catalog must never ship pre-filled checksums (see manifest.py docstring)."""
        for entry in manifest.MODEL_CATALOG.values():
            assert not hasattr(entry, "sha256")


class TestDownloader:
    MODEL_NAME = "whisper-tiny"

    def test_status_missing_file(self, tmp_path: Path):
        status = downloader.get_status(self.MODEL_NAME, tmp_path)
        assert status.downloaded is False
        assert status.verified is False

    def test_status_unknown_model_raises(self, tmp_path: Path):
        with pytest.raises(downloader.DownloadError):
            downloader.get_status("nonexistent-model", tmp_path)

    def test_download_pins_checksum_on_first_fetch(self, tmp_path: Path):
        payload = b"fake-model-bytes"
        with patch("voice_shell.src.appstore.downloader.urllib.request.urlopen") as urlopen:
            urlopen.return_value = FakeResponse(payload)
            status = downloader.download(self.MODEL_NAME, tmp_path)

        assert status.downloaded is True
        assert status.verified is True
        assert status.path.read_bytes() == payload

        lockfile = json.loads((tmp_path / ".checksums.json").read_text())
        assert lockfile[self.MODEL_NAME] == hashlib.sha256(payload).hexdigest()

    def test_second_download_call_reuses_verified_file_without_network(self, tmp_path: Path):
        payload = b"fake-model-bytes"
        with patch("voice_shell.src.appstore.downloader.urllib.request.urlopen") as urlopen:
            urlopen.return_value = FakeResponse(payload)
            downloader.download(self.MODEL_NAME, tmp_path)

        with patch("voice_shell.src.appstore.downloader.urllib.request.urlopen") as urlopen:
            status = downloader.download(self.MODEL_NAME, tmp_path)
            urlopen.assert_not_called()
        assert status.verified is True

    def test_download_detects_tampering_against_pinned_checksum(self, tmp_path: Path):
        payload = b"fake-model-bytes"
        with patch("voice_shell.src.appstore.downloader.urllib.request.urlopen") as urlopen:
            urlopen.return_value = FakeResponse(payload)
            downloader.download(self.MODEL_NAME, tmp_path)

        # Simulate the file being corrupted/tampered with on disk.
        entry = manifest.get_model(self.MODEL_NAME)
        (tmp_path / entry.dest_path).write_bytes(b"tampered-bytes")

        status = downloader.get_status(self.MODEL_NAME, tmp_path)
        assert status.downloaded is True
        assert status.verified is False
        assert "mismatch" in status.detail.lower()

        with pytest.raises(downloader.DownloadError):
            downloader.download(self.MODEL_NAME, tmp_path)

    def test_force_redownload_rejects_mismatched_pinned_checksum(self, tmp_path: Path):
        with patch("voice_shell.src.appstore.downloader.urllib.request.urlopen") as urlopen:
            urlopen.return_value = FakeResponse(b"original-bytes")
            downloader.download(self.MODEL_NAME, tmp_path)

        with patch("voice_shell.src.appstore.downloader.urllib.request.urlopen") as urlopen:
            urlopen.return_value = FakeResponse(b"different-bytes")
            with pytest.raises(downloader.DownloadError):
                downloader.download(self.MODEL_NAME, tmp_path, force=True)

    def test_clear_pin_allows_new_checksum_to_be_trusted(self, tmp_path: Path):
        with patch("voice_shell.src.appstore.downloader.urllib.request.urlopen") as urlopen:
            urlopen.return_value = FakeResponse(b"original-bytes")
            downloader.download(self.MODEL_NAME, tmp_path)

        downloader.clear_pin(self.MODEL_NAME, tmp_path)

        with patch("voice_shell.src.appstore.downloader.urllib.request.urlopen") as urlopen:
            urlopen.return_value = FakeResponse(b"different-bytes")
            status = downloader.download(self.MODEL_NAME, tmp_path, force=True)
        assert status.verified is True
        assert status.path.read_bytes() == b"different-bytes"

    def test_download_network_failure_raises_download_error(self, tmp_path: Path):
        with patch("voice_shell.src.appstore.downloader.urllib.request.urlopen") as urlopen:
            urlopen.side_effect = OSError("network unreachable")
            with pytest.raises(downloader.DownloadError):
                downloader.download(self.MODEL_NAME, tmp_path)
        # No partial file should remain in the destination directory.
        entry = manifest.get_model(self.MODEL_NAME)
        assert not (tmp_path / entry.dest_path).exists()

    def test_status_all_covers_full_catalog(self, tmp_path: Path):
        statuses = downloader.status_all(tmp_path)
        assert len(statuses) == len(manifest.MODEL_CATALOG)
        assert all(not s.downloaded for s in statuses)
