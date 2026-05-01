"""
Streaming speech-to-text using either native Moonshine Voice streams or
sherpa-onnx recognizers.

Wraps:
- Moonshine Voice native streaming models for true online partials
- sherpa-onnx streaming transducers such as zipformer
- buffered Hugging Face Parakeet CTC for local streaming-style evaluation
- sherpa-onnx offline recognition for batch/refinement passes such as Parakeet

Thread safety: NOT thread-safe. All methods must be called from a single
thread (the streaming worker thread). No locking is needed since the
underlying recognizers/streams are single-consumer.
"""

import collections
import contextlib
import importlib.util
import base64
import errno
import json
import numpy as np
import os
import sys
import tarfile
import time
import urllib.parse
import urllib.request


def _moonshine_v2_model(archive_name: str, size_mb: int | None = None) -> dict:
    config = {
        "url": f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{archive_name}.tar.bz2",
        "dir": archive_name,
        "encoder": "encoder_model.ort",
        "decoder": "decoder_model_merged.ort",
        "tokens": "tokens.txt",
        "type": "moonshine_v2",
    }
    if size_mb is not None:
        config["size_mb"] = size_mb
    return config


def _moonshine_native_streaming_model(
    arch: str,
    size_mb: int | None = None,
    update_interval_seconds: float = 0.18,
) -> dict:
    config = {
        "kind": "native_moonshine_online",
        "language": "en",
        "arch": arch,
        "update_interval_seconds": update_interval_seconds,
    }
    if size_mb is not None:
        config["size_mb"] = size_mb
    return config


def _parakeet_ctc_streaming_model(
    model_id: str,
    size_mb: int | None = None,
    update_interval_seconds: float = 0.24,
    endpoint_silence_seconds: float = 0.36,
    min_decode_seconds: float = 0.12,
) -> dict:
    config = {
        "kind": "buffered_parakeet_ctc_online",
        "model_id": model_id,
        "update_interval_seconds": update_interval_seconds,
        "endpoint_silence_seconds": endpoint_silence_seconds,
        "min_decode_seconds": min_decode_seconds,
    }
    if size_mb is not None:
        config["size_mb"] = size_mb
    return config


def _nim_realtime_streaming_model(
    session_model: str | None = None,
    size_mb: int | None = None,
    language: str = "en-US",
) -> dict:
    config = {
        "kind": "nim_realtime_transcription",
        "language": language,
        "automatic_punctuation": True,
    }
    if session_model:
        config["session_model"] = session_model
    if size_mb is not None:
        config["size_mb"] = size_mb
    return config


STREAMING_MODELS = {
    "zipformer-en": {
        "kind": "online_transducer",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        "dir": "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "encoder": "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "decoder": "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "joiner": "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
        "tokens": "tokens.txt",
        "size_mb": 80,
    },
    "zipformer-en-20M": {
        "kind": "online_transducer",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2",
        "dir": "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        "encoder": "encoder-epoch-99-avg-1.onnx",
        "decoder": "decoder-epoch-99-avg-1.onnx",
        "joiner": "joiner-epoch-99-avg-1.onnx",
        "tokens": "tokens.txt",
        "size_mb": 20,
    },
    "parakeet-ctc-0.6b": _parakeet_ctc_streaming_model(
        "nvidia/parakeet-ctc-0.6b",
        size_mb=2400,
        update_interval_seconds=0.24,
        endpoint_silence_seconds=0.36,
        min_decode_seconds=0.12,
    ),
    "nemotron-asr-streaming-nim": _nim_realtime_streaming_model(
        size_mb=3200,
    ),
    "parakeet-ctc-0.6b-nim": _nim_realtime_streaming_model(
        "parakeet-0-6b-ctc-en-us",
        size_mb=3070,
    ),
    "parakeet-ctc-1.1b-nim": _nim_realtime_streaming_model(
        "parakeet-1-1b-ctc-en-us",
        size_mb=5600,
    ),
    "moonshine-tiny-streaming-en": _moonshine_native_streaming_model(
        "tiny-streaming",
        size_mb=30,
        update_interval_seconds=0.12,
    ),
    "moonshine-small-streaming-en": _moonshine_native_streaming_model(
        "small-streaming",
        size_mb=100,
        update_interval_seconds=0.14,
    ),
    "moonshine-medium-streaming-en": _moonshine_native_streaming_model(
        "medium-streaming",
        size_mb=245,
        update_interval_seconds=0.10,
    ),
}

LEGACY_STREAMING_MODEL_HINTS = {
    "moonshine-tiny-en-v2": "moonshine-tiny-streaming-en",
    "moonshine-base-en-v2": "moonshine-medium-streaming-en",
}

OFFLINE_MODELS = {
    "parakeet-tdt-0.6b-v2": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
        "dir": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        "type": "transducer",
        "encoder": "encoder.int8.onnx",
        "decoder": "decoder.int8.onnx",
        "joiner": "joiner.int8.onnx",
        "tokens": "tokens.txt",
        "size_mb": 300,
        "feature_dim": 80,
        "model_type": "nemo_transducer",
    },
    "moonshine-tiny-en-v2": _moonshine_v2_model(
        "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27",
        size_mb=30,
    ),
    "moonshine-base-en-v2": _moonshine_v2_model(
        "sherpa-onnx-moonshine-base-en-quantized-2026-02-27",
        size_mb=60,
    ),
}

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/sherpa-onnx")


def sherpa_onnx_available() -> bool:
    """Return True if the sherpa-onnx Python package is importable."""
    return importlib.util.find_spec("sherpa_onnx") is not None


def moonshine_voice_available() -> bool:
    """Return True if the native Moonshine Voice Python package is importable."""
    return importlib.util.find_spec("moonshine_voice") is not None


def parakeet_ctc_available() -> bool:
    """Return True if the Hugging Face CTC runtime dependencies are importable."""
    return (
        importlib.util.find_spec("transformers") is not None
        and importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("librosa") is not None
    )


def nim_realtime_available() -> bool:
    """Return True if the websocket client dependency is importable."""
    return importlib.util.find_spec("websocket") is not None


def _resolve_streaming_model_name(model_name: str) -> str:
    if model_name in STREAMING_MODELS:
        return model_name
    if model_name in LEGACY_STREAMING_MODEL_HINTS:
        replacement = LEGACY_STREAMING_MODEL_HINTS[model_name]
        raise ValueError(
            f"Streaming model '{model_name}' was removed when switching to native "
            f"Moonshine streaming. Use '{replacement}' instead."
        )
    available = get_streaming_model_names()
    raise ValueError(
        f"Unknown streaming model '{model_name}'. Available: {available}"
    )


def get_streaming_model_names() -> list[str]:
    return list(STREAMING_MODELS.keys())


def get_offline_model_names() -> list[str]:
    return list(OFFLINE_MODELS.keys())


def is_sherpa_offline_model(model_name: str) -> bool:
    return model_name in OFFLINE_MODELS


def streaming_model_available(model_name: str) -> bool:
    model_name = _resolve_streaming_model_name(model_name)
    model_kind = STREAMING_MODELS[model_name]["kind"]
    if model_kind == "online_transducer":
        return sherpa_onnx_available()
    if model_kind == "buffered_parakeet_ctc_online":
        return parakeet_ctc_available()
    if model_kind == "nim_realtime_transcription":
        return nim_realtime_available()
    if model_kind == "native_moonshine_online":
        return moonshine_voice_available()
    return False


def streaming_model_install_hint(model_name: str) -> str:
    if model_name in LEGACY_STREAMING_MODEL_HINTS:
        replacement = LEGACY_STREAMING_MODEL_HINTS[model_name]
        return (
            f"'{model_name}' was the old sherpa-based Moonshine path. "
            f"Use '{replacement}' and install with: pip install moonshine-voice"
        )

    model_name = _resolve_streaming_model_name(model_name)
    model_kind = STREAMING_MODELS[model_name]["kind"]
    if model_kind == "online_transducer":
        return "Install with: pip install 'sherpa-onnx>=1.12.28'"
    if model_kind == "buffered_parakeet_ctc_online":
        return "Install with: pip install 'transformers>=4.57.3' librosa"
    if model_kind == "nim_realtime_transcription":
        return (
            "Install with: pip install websocket-client, set "
            "VOICE_NIM_URL=http://127.0.0.1:9000, and run a local NVIDIA Riva "
            "ASR NIM realtime streaming profile such as NVIDIA Nemotron ASR Streaming"
        )
    if model_kind == "native_moonshine_online":
        return "Install with: pip install moonshine-voice"
    return "Install the backend required by this streaming model"


@contextlib.contextmanager
def _moonshine_cache_override(cache_dir: str | None):
    """Optionally point moonshine-voice at a caller-provided cache dir.

    The native package already defaults to a sensible platform cache. Only
    override it when the caller explicitly passes a non-default cache path.
    """

    if not cache_dir or cache_dir == DEFAULT_CACHE_DIR:
        yield
        return

    env_name = "MOONSHINE_VOICE_CACHE"
    previous = os.environ.get(env_name)
    os.environ[env_name] = os.path.expanduser(cache_dir)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


def _download_model(
    model_name: str, model_catalog: dict[str, dict], cache_dir: str, kind: str
) -> str:
    if model_name not in model_catalog:
        raise ValueError(f"Unknown {kind} model '{model_name}'")

    config = model_catalog[model_name]
    model_dir = os.path.join(cache_dir, config["dir"])

    tokens_path = os.path.join(model_dir, config["tokens"])
    if os.path.exists(tokens_path):
        return model_dir

    os.makedirs(cache_dir, exist_ok=True)
    url = config["url"]
    tarball_path = os.path.join(cache_dir, os.path.basename(url))

    size_mb = config.get("size_mb")
    if size_mb is None:
        print(f"Downloading {kind} model '{model_name}'...")
    else:
        print(f"Downloading {kind} model '{model_name}' (~{size_mb}MB)...")
    print(f"  URL: {url}")

    def _progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tarball_path, reporthook=_progress_hook)
        print()
    except Exception as e:
        if os.path.exists(tarball_path):
            os.remove(tarball_path)
        raise RuntimeError(f"Failed to download model: {e}") from e

    print(f"  Extracting to {cache_dir}...")
    try:
        with tarfile.open(tarball_path, "r:bz2") as tar:
            tar.extractall(path=cache_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to extract model: {e}") from e
    finally:
        if os.path.exists(tarball_path):
            os.remove(tarball_path)

    if not os.path.exists(tokens_path):
        raise RuntimeError(
            "Model extraction succeeded but "
            f"{config['tokens']} was not found in {model_dir}"
        )

    print(f"  Model ready: {model_dir}")
    return model_dir


class StreamingSTT:
    """Streaming speech-to-text with a stable wrapper over multiple backends.

    Supports:
    - native Moonshine Voice streaming models
    - true online sherpa-onnx transducer models such as zipformer
    """

    def __init__(
        self,
        model_name: str = "zipformer-en",
        cache_dir: str = DEFAULT_CACHE_DIR,
        sample_rate: int = 16000,
        device: str = "cpu",
    ):
        self.model_name = _resolve_streaming_model_name(model_name)
        self.cache_dir = cache_dir
        self.sample_rate = sample_rate
        self.device = device
        self.recognizer = None
        self.stream = None
        self.websocket = None
        self.websocket_module = None
        self.model_kind = None
        self.partial_text = ""
        self.completed_texts = collections.deque()
        self.stream_error = None
        self.model_config = dict(STREAMING_MODELS[self.model_name])
        self.model_kind = self.model_config["kind"]
        self.processor = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.final_text = ""
        self.decode_since_samples = 0
        self.trailing_silence_samples = 0
        self.in_utterance = False
        self.target_device = "cpu"
        self.nim_base_url = None
        self.nim_session_config = None
        self.nim_buffer_dirty = False

    @classmethod
    def download_model(cls, model_name: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
        """Download streaming model if not already cached.

        Returns the path to the model directory.
        """
        model_name = _resolve_streaming_model_name(model_name)
        config = STREAMING_MODELS[model_name]
        if config["kind"] == "native_moonshine_online":
            return cls._download_native_moonshine_model(model_name, cache_dir)[0]
        if config["kind"] == "nim_realtime_transcription":
            return cls._nim_base_url_from_env()
        return _download_model(model_name, STREAMING_MODELS, cache_dir, "streaming")

    @staticmethod
    def _download_native_moonshine_model(
        model_name: str, cache_dir: str = DEFAULT_CACHE_DIR
    ) -> tuple[str, object]:
        from moonshine_voice import get_model_for_language, string_to_model_arch

        config = STREAMING_MODELS[model_name]
        wanted_arch = string_to_model_arch(config["arch"])
        with _moonshine_cache_override(cache_dir):
            return get_model_for_language(
                wanted_language=config.get("language", "en"),
                wanted_model_arch=wanted_arch,
            )

    def _handle_moonshine_event(self, event):
        event_name = type(event).__name__

        if event_name == "LineTextChanged":
            line = getattr(event, "line", None)
            text = getattr(line, "text", "")
            self.partial_text = text.strip() if text else ""
            return

        if event_name == "LineCompleted":
            line = getattr(event, "line", None)
            text = getattr(line, "text", "")
            final_text = text.strip() if text else ""
            if final_text:
                self.completed_texts.append(final_text)
            self.partial_text = ""
            return

        if event_name == "Error":
            error = getattr(event, "error", None)
            self.stream_error = error or RuntimeError("Moonshine stream error")

    def _ensure_stream_healthy(self):
        if self.stream_error is None:
            return
        if isinstance(self.stream_error, Exception):
            raise RuntimeError(
                f"Streaming backend failed: {self.stream_error}"
            ) from self.stream_error
        raise RuntimeError(f"Streaming backend failed: {self.stream_error}")

    def _restart_moonshine_stream(self):
        self.partial_text = ""
        self.completed_texts.clear()
        self.stream_error = None

        if self.stream is not None:
            close = getattr(self.stream, "close", None)
            if callable(close):
                close()
            self.stream = None

        if self.recognizer is None:
            return

        update_interval = self.model_config.get("update_interval_seconds", 0.18)
        self.stream = self.recognizer.create_stream(update_interval=update_interval)
        self.stream.add_listener(self._handle_moonshine_event)
        self.stream.start()

    def _reset_buffered_ctc_state(self):
        self.partial_text = ""
        self.final_text = ""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.decode_since_samples = 0
        self.trailing_silence_samples = 0
        self.in_utterance = False

    def _decode_parakeet_ctc_audio(self, samples: np.ndarray) -> str:
        import torch

        if self.recognizer is None or self.processor is None or samples.size == 0:
            return ""

        inputs = self.processor(
            samples,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        move = getattr(inputs, "to", None)
        if callable(move):
            dtype = getattr(self.recognizer, "dtype", None)
            try:
                if dtype is None:
                    inputs = move(self.target_device)
                else:
                    inputs = move(self.target_device, dtype=dtype)
            except TypeError:
                inputs = move(self.target_device)

        with torch.inference_mode():
            generator = getattr(self.recognizer, "generate", None)
            if callable(generator):
                predicted = generator(**inputs)
            else:
                logits = self.recognizer(**inputs).logits
                predicted = torch.argmax(logits, dim=-1)

        decoded = self.processor.batch_decode(predicted, skip_special_tokens=True)
        if not decoded:
            return ""
        return decoded[0].strip()

    @staticmethod
    def _nim_base_url_from_env() -> str:
        base_url = os.environ.get("VOICE_NIM_URL", "http://127.0.0.1:9000").strip()
        if not base_url:
            base_url = "http://127.0.0.1:9000"
        if "://" not in base_url:
            base_url = f"http://{base_url}"
        return base_url.rstrip("/")

    @staticmethod
    def _nim_api_key_from_env() -> str:
        for env_name in ("VOICE_NIM_API_KEY", "NVIDIA_API_KEY"):
            value = os.environ.get(env_name, "").strip()
            if value:
                return value
        return ""

    @classmethod
    def _nim_session_urls(cls) -> tuple[str, str]:
        base_url = cls._nim_base_url_from_env()
        parsed = urllib.parse.urlparse(base_url)
        if not parsed.netloc:
            parsed = urllib.parse.urlparse(f"http://{base_url}")

        http_scheme = "https" if parsed.scheme in ("https", "wss") else "http"
        ws_scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
        base_path = parsed.path.rstrip("/")

        session_url = urllib.parse.urlunparse(
            (
                http_scheme,
                parsed.netloc,
                f"{base_path}/v1/realtime/transcription_sessions",
                "",
                "",
                "",
            )
        )
        websocket_url = urllib.parse.urlunparse(
            (
                ws_scheme,
                parsed.netloc,
                f"{base_path}/v1/realtime",
                "",
                "intent=transcription",
                "",
            )
        )
        return session_url, websocket_url

    @classmethod
    def _nim_http_headers(cls) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = cls._nim_api_key_from_env()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @classmethod
    def _nim_websocket_headers(cls) -> list[str]:
        api_key = cls._nim_api_key_from_env()
        if not api_key:
            return []
        return [f"Authorization: Bearer {api_key}"]

    @classmethod
    def _nim_create_transcription_session(cls) -> dict:
        session_url, _ = cls._nim_session_urls()
        request = urllib.request.Request(
            session_url,
            data=b"{}",
            headers=cls._nim_http_headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5.0) as response:
                payload = response.read().decode("utf-8")
        except Exception as exc:
            raise RuntimeError(
                "Failed to create NVIDIA Riva ASR NIM realtime session at "
                f"{session_url}: {exc}"
            ) from exc
        return json.loads(payload or "{}")

    def _nim_wait_for_event(
        self,
        expected_types: set[str],
        timeout_seconds: float,
    ) -> dict:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            event = self._nim_receive_event(timeout_seconds=max(0.05, remaining))
            if event is None:
                continue
            event_type = event.get("type", "")
            if event_type in expected_types:
                return event
            self._handle_nim_event(event)
        expected = ", ".join(sorted(expected_types))
        raise RuntimeError(f"Timed out waiting for NVIDIA NIM event: {expected}")

    def _nim_receive_event(self, timeout_seconds: float | None) -> dict | None:
        if self.websocket is None or self.websocket_module is None:
            return None

        timeout_exc = getattr(
            self.websocket_module,
            "WebSocketTimeoutException",
            TimeoutError,
        )
        self.websocket.settimeout(timeout_seconds)
        try:
            raw = self.websocket.recv()
        except timeout_exc:
            return None
        except BlockingIOError:
            return None
        except OSError as exc:
            if exc.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
                return None
            raise
        finally:
            self.websocket.settimeout(0.0)

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def _nim_send_event(self, payload: dict):
        if self.websocket is None:
            raise RuntimeError("NVIDIA NIM websocket is not connected")
        self.websocket.send(json.dumps(payload))

    def _nim_handle_completed_transcript(self, event: dict):
        transcript = event.get("transcript", "")
        final_text = transcript.strip() if transcript else ""
        if final_text:
            self.completed_texts.append(final_text)
        self.partial_text = ""
        self.nim_buffer_dirty = False

    def _handle_nim_event(self, event: dict):
        event_type = event.get("type", "")
        if event_type == "conversation.item.input_audio_transcription.delta":
            delta = event.get("delta", "")
            self.partial_text = delta.strip() if delta else ""
            return
        if event_type == "conversation.item.input_audio_transcription.completed":
            self._nim_handle_completed_transcript(event)
            return
        if event_type == "input_audio_buffer.cleared":
            self.partial_text = ""
            self.nim_buffer_dirty = False
            return
        if event_type in {
            "conversation.created",
            "transcription_session.updated",
            "input_audio_buffer.committed",
        }:
            return
        if "error" in event_type.lower():
            error_info = event.get("error", {})
            message = (
                error_info.get("message")
                or event.get("message")
                or "Unknown error"
            )
            self.stream_error = RuntimeError(f"NVIDIA NIM realtime error: {message}")

    def _nim_drain_events(self):
        while True:
            event = self._nim_receive_event(timeout_seconds=0.0)
            if event is None:
                return
            self._handle_nim_event(event)

    def _nim_clear_audio_buffer(self):
        # Nemotron's realtime API accepts append/commit/done client events but
        # does not expose a websocket clear event. Reset only the local client
        # state here and let the server-side stream advance naturally.
        self.partial_text = ""
        self.nim_buffer_dirty = False

    def create_recognizer(self):
        """Initialize the backend-specific recognizer for the selected model."""
        config = self.model_config

        if self.model_kind == "online_transducer":
            import sherpa_onnx

            model_dir = self.download_model(self.model_name, self.cache_dir)
            encoder_path = os.path.join(model_dir, config["encoder"])
            decoder_path = os.path.join(model_dir, config["decoder"])
            joiner_path = os.path.join(model_dir, config["joiner"])
            tokens_path = os.path.join(model_dir, config["tokens"])

            for path, label in [
                (encoder_path, "encoder"),
                (decoder_path, "decoder"),
                (joiner_path, "joiner"),
                (tokens_path, "tokens"),
            ]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found: {label} at {path}")

            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=encoder_path,
                decoder=decoder_path,
                joiner=joiner_path,
                tokens=tokens_path,
                num_threads=2,
                sample_rate=self.sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=1.2,
                rule3_min_utterance_length=20.0,
            )

            self.stream = self.recognizer.create_stream()
        elif self.model_kind == "buffered_parakeet_ctc_online":
            import torch
            from transformers import AutoModelForCTC, AutoProcessor

            requested_cuda = self.device == "cuda" and torch.cuda.is_available()
            self.target_device = "cuda" if requested_cuda else "cpu"
            model_kwargs = {"trust_remote_code": True}
            cache_dir = None if self.cache_dir == DEFAULT_CACHE_DIR else self.cache_dir
            if cache_dir:
                model_kwargs["cache_dir"] = cache_dir
            if requested_cuda:
                model_kwargs["torch_dtype"] = torch.float16
            else:
                model_kwargs["torch_dtype"] = torch.float32

            model_id = config["model_id"]
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            self.recognizer = AutoModelForCTC.from_pretrained(model_id, **model_kwargs)
            move = getattr(self.recognizer, "to", None)
            if callable(move):
                self.recognizer = move(self.target_device)
            eval_fn = getattr(self.recognizer, "eval", None)
            if callable(eval_fn):
                eval_fn()
            self._reset_buffered_ctc_state()
        elif self.model_kind == "native_moonshine_online":
            from moonshine_voice import Transcriber

            model_dir, model_arch = self._download_native_moonshine_model(
                self.model_name, self.cache_dir
            )
            update_interval = config.get("update_interval_seconds", 0.18)
            self.recognizer = Transcriber(
                model_path=model_dir,
                model_arch=model_arch,
                update_interval=update_interval,
            )
            self._restart_moonshine_stream()
        elif self.model_kind == "nim_realtime_transcription":
            import websocket

            self.nim_base_url = self._nim_base_url_from_env()
            self.nim_session_config = self._nim_create_transcription_session()
            session_url, websocket_url = self._nim_session_urls()
            self.websocket_module = websocket
            try:
                self.websocket = websocket.create_connection(
                    websocket_url,
                    header=self._nim_websocket_headers(),
                    timeout=5.0,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to connect to NVIDIA Riva ASR NIM websocket at "
                    f"{websocket_url}: {exc}"
                ) from exc

            self._nim_wait_for_event({"conversation.created"}, timeout_seconds=5.0)

            session_config = dict(self.nim_session_config or {})
            session_config["input_audio_format"] = "pcm16"
            session_config.setdefault("input_audio_transcription", {})
            session_config["input_audio_transcription"]["language"] = config.get(
                "language", "en-US"
            )
            session_model = config.get("session_model")
            if session_model:
                session_config["input_audio_transcription"]["model"] = session_model
            session_config.setdefault("input_audio_params", {})
            session_config["input_audio_params"]["sample_rate_hz"] = self.sample_rate
            session_config["input_audio_params"]["num_channels"] = 1
            session_config.setdefault("recognition_config", {})
            session_config["recognition_config"][
                "enable_automatic_punctuation"
            ] = config.get("automatic_punctuation", True)

            self._nim_send_event(
                {
                    "type": "transcription_session.update",
                    "session": session_config,
                }
            )
            updated = self._nim_wait_for_event(
                {"transcription_session.updated"},
                timeout_seconds=5.0,
            )
            self.nim_session_config = updated.get("session", session_config)
            self.partial_text = ""
            self.completed_texts.clear()
            self.nim_buffer_dirty = False
            self.websocket.settimeout(0.0)
        else:
            raise RuntimeError(f"Unsupported streaming model kind: {self.model_kind}")

        print(f"Streaming recognizer initialized ({self.model_name})")

    def feed_chunk(self, chunk: np.ndarray, is_speech: bool | None = None) -> str:
        """Feed an int16 audio chunk and return the current partial text.

        Args:
            chunk: numpy int16 audio data (single channel, 16kHz)
            is_speech: Optional speech activity flag, unused by native streaming

        Returns:
            Current partial transcription text
        """
        if self.model_kind == "nim_realtime_transcription":
            if self.websocket is None:
                return ""
        elif self.recognizer is None:
            return ""

        if self.model_kind == "online_transducer":
            if self.stream is None:
                return ""

            samples = chunk.astype(np.float32) / 32768.0
            self.stream.accept_waveform(self.sample_rate, samples)

            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)

            result = self.recognizer.get_result(self.stream)
            return self._extract_text(result)

        if self.model_kind == "buffered_parakeet_ctc_online":
            samples = chunk.astype(np.float32) / 32768.0
            self.audio_buffer = np.concatenate((self.audio_buffer, samples))
            self.decode_since_samples += len(samples)

            if is_speech:
                self.in_utterance = True
                self.trailing_silence_samples = 0
            elif self.in_utterance:
                self.trailing_silence_samples += len(samples)

            min_decode_samples = int(
                self.sample_rate * self.model_config.get("min_decode_seconds", 0.12)
            )
            update_interval_samples = int(
                self.sample_rate
                * self.model_config.get("update_interval_seconds", 0.24)
            )
            endpoint_silence_samples = int(
                self.sample_rate
                * self.model_config.get("endpoint_silence_seconds", 0.36)
            )

            should_decode = (
                self.in_utterance
                and self.audio_buffer.size >= min_decode_samples
                and (
                    self.decode_since_samples >= update_interval_samples
                    or self.trailing_silence_samples >= endpoint_silence_samples
                )
            )
            if should_decode:
                text = self._decode_parakeet_ctc_audio(self.audio_buffer)
                self.partial_text = text
                self.decode_since_samples = 0
                if (
                    text
                    and self.trailing_silence_samples >= endpoint_silence_samples
                ):
                    self.final_text = text
            return self.partial_text

        if self.model_kind == "nim_realtime_transcription":
            self._ensure_stream_healthy()
            if self.websocket is None:
                return ""

            audio_bytes = np.asarray(chunk, dtype=np.int16).tobytes()
            self._nim_send_event(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio_bytes).decode("utf-8"),
                }
            )
            self._nim_send_event({"type": "input_audio_buffer.commit"})
            self.nim_buffer_dirty = True
            self._nim_drain_events()
            self._ensure_stream_healthy()
            return self.partial_text

        self._ensure_stream_healthy()
        if self.stream is None:
            return ""

        samples = chunk.astype(np.float32) / 32768.0
        self.stream.add_audio(samples, self.sample_rate)
        return self.partial_text

    @staticmethod
    def _extract_text(result) -> str:
        """Extract text from result (handles both str and object with .text)."""
        if isinstance(result, str):
            return result.strip()
        if hasattr(result, "text"):
            return result.text.strip() if result.text else ""
        return str(result).strip() if result else ""

    def check_endpoint(self) -> tuple[bool, str]:
        """Check if an endpoint (end of utterance) was detected.

        Returns:
            (is_endpoint, final_text) - if is_endpoint is True, final_text
            contains the complete utterance and the stream has been reset.
        """
        if self.model_kind == "nim_realtime_transcription":
            if self.websocket is None:
                return False, ""
        elif self.recognizer is None:
            return False, ""

        if self.model_kind == "online_transducer":
            if self.stream is None:
                return False, ""
            if self.recognizer.is_endpoint(self.stream):
                result = self.recognizer.get_result(self.stream)
                final_text = self._extract_text(result)
                self.recognizer.reset(self.stream)
                return True, final_text
            return False, ""

        if self.model_kind == "buffered_parakeet_ctc_online":
            if not self.final_text:
                return False, ""
            final_text = self.final_text
            self._reset_buffered_ctc_state()
            return True, final_text

        if self.model_kind == "nim_realtime_transcription":
            self._nim_drain_events()
            self._ensure_stream_healthy()
            if self.completed_texts:
                final_text = self.completed_texts.popleft()
                self._nim_clear_audio_buffer()
                return True, final_text
            return False, ""

        self._ensure_stream_healthy()
        if self.completed_texts:
            return True, self.completed_texts.popleft()

        return False, ""

    def reset(self):
        """Reset the stream for a fresh utterance."""
        if self.model_kind == "online_transducer" and self.recognizer is not None:
            self.stream = self.recognizer.create_stream()
            return

        if self.model_kind == "buffered_parakeet_ctc_online":
            self._reset_buffered_ctc_state()
            return

        if self.model_kind == "native_moonshine_online":
            self._restart_moonshine_stream()
            return

        if self.model_kind == "nim_realtime_transcription":
            self.partial_text = ""
            self.completed_texts.clear()
            self.stream_error = None
            self._nim_clear_audio_buffer()
            return

        self.partial_text = ""
        self.completed_texts.clear()
        self.stream_error = None

    def close(self):
        """Release backend resources."""
        if self.model_kind == "native_moonshine_online" and self.stream is not None:
            close = getattr(self.stream, "close", None)
            if callable(close):
                close()
            self.stream = None

        if self.model_kind == "native_moonshine_online" and self.recognizer is not None:
            close = getattr(self.recognizer, "close", None)
            if callable(close):
                close()
            self.recognizer = None
            return

        if self.model_kind == "buffered_parakeet_ctc_online":
            self.recognizer = None
            self.processor = None
            self._reset_buffered_ctc_state()
            return

        if self.model_kind == "nim_realtime_transcription":
            if self.websocket is not None:
                self.websocket.close()
                self.websocket = None
            self.websocket_module = None
            self.partial_text = ""
            self.completed_texts.clear()
            self.stream_error = None
            self.nim_buffer_dirty = False
            return

        self.partial_text = ""
        self.completed_texts.clear()
        self.stream_error = None


class OfflineSTT:
    """Offline speech-to-text using sherpa-onnx OfflineRecognizer."""

    def __init__(
        self,
        model_name: str = "parakeet-tdt-0.6b-v2",
        cache_dir: str = DEFAULT_CACHE_DIR,
        sample_rate: int = 16000,
        provider: str = "cpu",
        num_threads: int = 2,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.sample_rate = sample_rate
        self.provider = provider
        self.num_threads = num_threads
        self.recognizer = None

        if model_name not in OFFLINE_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {get_offline_model_names()}"
            )

        self.model_config = OFFLINE_MODELS[model_name]

    @classmethod
    def download_model(cls, model_name: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
        """Download offline model if not already cached."""
        return _download_model(model_name, OFFLINE_MODELS, cache_dir, "offline")

    def create_recognizer(self):
        """Initialize the sherpa-onnx OfflineRecognizer."""
        import sherpa_onnx

        model_dir = self.download_model(self.model_name, self.cache_dir)
        config = self.model_config
        model_type = config["type"]

        if model_type == "transducer":
            encoder_path = os.path.join(model_dir, config["encoder"])
            decoder_path = os.path.join(model_dir, config["decoder"])
            joiner_path = os.path.join(model_dir, config["joiner"])
            tokens_path = os.path.join(model_dir, config["tokens"])

            for path, label in [
                (encoder_path, "encoder"),
                (decoder_path, "decoder"),
                (joiner_path, "joiner"),
                (tokens_path, "tokens"),
            ]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found: {label} at {path}")

            self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder_path,
                decoder=decoder_path,
                joiner=joiner_path,
                tokens=tokens_path,
                num_threads=self.num_threads,
                sample_rate=self.sample_rate,
                feature_dim=config.get("feature_dim", 80),
                decoding_method="greedy_search",
                provider=self.provider,
                model_type=config.get("model_type", "transducer"),
            )
        elif model_type == "moonshine_v2":
            encoder_path = os.path.join(model_dir, config["encoder"])
            decoder_path = os.path.join(model_dir, config["decoder"])
            tokens_path = os.path.join(model_dir, config["tokens"])

            for path, label in [
                (encoder_path, "encoder"),
                (decoder_path, "merged decoder"),
                (tokens_path, "tokens"),
            ]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found: {label} at {path}")

            self.recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
                encoder=encoder_path,
                decoder=decoder_path,
                tokens=tokens_path,
                num_threads=self.num_threads,
                decoding_method="greedy_search",
                provider=self.provider,
            )
        else:
            raise RuntimeError(
                f"Offline model type '{model_type}' is not supported by this build"
            )
        print(f"Offline recognizer initialized ({self.model_name}, {self.provider})")

    @staticmethod
    def _extract_text(result) -> str:
        if isinstance(result, str):
            return result.strip()
        if hasattr(result, "text"):
            return result.text.strip() if result.text else ""
        return str(result).strip() if result else ""

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe int16 or float32 mono audio and return the final text."""
        if self.recognizer is None:
            return ""

        if audio.dtype == np.int16:
            samples = audio.astype(np.float32) / 32768.0
        else:
            samples = audio.astype(np.float32, copy=False)

        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, samples)
        if hasattr(stream, "input_finished"):
            stream.input_finished()
        self.recognizer.decode_stream(stream)

        result = getattr(stream, "result", None)

        return self._extract_text(result)
