import threading
import time
from pathlib import Path
from typing import Optional


class WakeWordDetectorError(Exception):
    """Raised when the wake word detector fails to initialize or process audio."""


class WakeWordDetector:
    """Multi-backend wake word detector supporting Porcupine, OpenWakeWord, and hotkey fallback.

    The interface is uniform regardless of the backend: ``process_chunk`` accepts
    raw 16-bit PCM audio and returns ``True`` when the wake word is detected.

    Backends (in order of preference):
    - **porcupine**: Picovoice Porcupine engine (fast, accurate, requires model file).
    - **openwakeword**: Open-source ONNX-based engine (free, slightly more CPU).
    - **hotkey**: Keyboard hotkey fallback for development (no microphone needed).
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        keyword: Optional[str] = None,
        sensitivity: float = 0.5,
        backend: Optional[str] = None,
        hotkey: str = "ctrl+shift+space",
        access_key: Optional[str] = None,
    ):
        """Initialize the wake word detector with the first available backend.

        Args:
            model_path: Path to the wake word model file (``.ppn`` for Porcupine,
                ``.onnx`` for OpenWakeWord). Ignored for ``hotkey`` backend.
            keyword: Human-readable keyword name (e.g., "Hey Nova"). Used for logging.
            sensitivity: Detection sensitivity (0.0–1.0). Higher = more triggers.
            backend: Explicit backend name (``porcupine``, ``openwakeword``, ``hotkey``).
                If ``None``, auto-detects in the order above.
            hotkey: Global hotkey combination for the ``hotkey`` backend.
            access_key: Picovoice access key (required for commercial use; free tier
                does not need a key for local, non-commercial use in some versions).

        Raises:
            WakeWordDetectorError: If no backend can be initialized.
        """
        self.keyword = keyword or "wake word"
        self._backend_name = backend
        self._backend: Optional[object] = None
        self._hotkey_flag = False
        self._hotkey_listener = None

        if backend == "hotkey" or (backend is None and model_path is None):
            self._init_hotkey(hotkey)
            return

        if backend == "porcupine" or (backend is None and model_path is not None):
            if self._init_porcupine(model_path, sensitivity, access_key):
                return

        if backend == "openwakeword" or backend is None:
            if self._init_openwakeword(model_path):
                return

        if self._backend is None:
            self._init_hotkey(hotkey)

    def _init_porcupine(
        self,
        model_path: Optional[Path],
        sensitivity: float,
        access_key: Optional[str],
    ) -> bool:
        try:
            import pvporcupine
        except ImportError:
            return False

        try:
            kwargs = {
                "keyword_paths": [str(model_path)],
                "sensitivities": [sensitivity],
            }
            if access_key is not None:
                kwargs["access_key"] = access_key

            self._backend = pvporcupine.create(**kwargs)
            self._backend_name = "porcupine"
            return True
        except Exception as exc:
            raise WakeWordDetectorError(
                f"Failed to initialize Porcupine wake word detector: {exc}"
            ) from exc

    def _init_openwakeword(self, model_path: Optional[Path]) -> bool:
        try:
            import openwakeword
        except ImportError:
            return False

        try:
            self._backend = openwakeword.Model(wakeword_models=[str(model_path)])
            self._backend_name = "openwakeword"
            return True
        except Exception as exc:
            raise WakeWordDetectorError(
                f"Failed to initialize OpenWakeWord detector: {exc}"
            ) from exc

    def _init_hotkey(self, hotkey: str) -> None:
        try:
            from pynput import keyboard
        except ImportError:
            self._backend_name = "hotkey_unavailable"
            return

        self._backend_name = "hotkey"
        self._hotkey_flag = False

        def on_press(key):
            try:
                if key == keyboard.Key.space and self._modifier_pressed():
                    self._hotkey_flag = True
            except AttributeError:
                pass

        def on_release(key):
            pass

        self._hotkey_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._hotkey_listener.start()

    def _modifier_pressed(self) -> bool:
        """Check if Ctrl and Shift are held (for the hotkey backend)."""
        try:
            from pynput import keyboard
            ctrl = any(
                k == keyboard.Key.ctrl or k == keyboard.Key.ctrl_l or k == keyboard.Key.ctrl_r
                for k in self._hotkey_listener._keys_pressed or []
            )
            shift = any(
                k == keyboard.Key.shift or k == keyboard.Key.shift_l or k == keyboard.Key.shift_r
                for k in self._hotkey_listener._keys_pressed or []
            )
            return ctrl and shift
        except Exception:
            return False

    def process_chunk(self, audio_chunk: bytes) -> bool:
        """Process a 16-bit PCM audio chunk and detect the wake word.

        Args:
            audio_chunk: Raw audio bytes. For Porcupine, must be exactly the
                engine's expected frame size (typically 512 bytes @ 16 kHz).

        Returns:
            ``True`` if the wake word was detected in this chunk.
        """
        if self._backend_name == "hotkey":
            triggered = self._hotkey_flag
            self._hotkey_flag = False
            return triggered

        if self._backend_name == "hotkey_unavailable":
            return False

        if self._backend is None:
            return False

        try:
            if self._backend_name == "porcupine":
                # Porcupine expects int16 frames. audio_chunk should be a
                # complete frame (e.g., 512 bytes = 256 samples).
                import struct
                if len(audio_chunk) < self._backend.frame_length * 2:
                    return False
                pcm = struct.unpack_from(
                    "h" * self._backend.frame_length, audio_chunk
                )
                result = self._backend.process(pcm)
                return result >= 0

            elif self._backend_name == "openwakeword":
                # OpenWakeWord expects float32 or int16 depending on version.
                # Here we assume the model handles raw bytes or numpy arrays.
                import numpy as np
                samples = np.frombuffer(audio_chunk, dtype=np.int16)
                predictions = self._backend.predict(samples)
                # predictions is a dict: {model_name: score}
                return any(score > 0.5 for score in predictions.values())
        except Exception as exc:
            raise WakeWordDetectorError(
                f"Wake word processing failed ({self._backend_name}): {exc}"
            ) from exc

        return False

    def start(self) -> None:
        """Start the wake word detector (e.g., start the hotkey listener)."""
        if self._backend_name == "hotkey" and self._hotkey_listener is not None:
            if not self._hotkey_listener.is_alive():
                self._hotkey_listener.start()

    def stop(self) -> None:
        """Stop the wake word detector and release resources."""
        if self._backend_name == "hotkey" and self._hotkey_listener is not None:
            self._hotkey_listener.stop()
            self._hotkey_listener = None

        if self._backend_name == "porcupine" and self._backend is not None:
            self._backend.delete()
            self._backend = None

        if self._backend_name == "openwakeword" and self._backend is not None:
            # OpenWakeWord may have a cleanup method; if not, just drop the ref.
            self._backend = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"backend={self._backend_name}, keyword={self.keyword!r}"
            f")"
        )
