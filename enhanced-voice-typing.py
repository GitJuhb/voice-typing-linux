#!/usr/bin/env python3
"""
Enhanced voice typing with pre-recording buffer
Combines faster-whisper with pre-buffer technique from RealtimeSTT

Features:
- Thread-safe transcription (no blocking in audio callback)
- Proper buffer timing for natural speech
- Accurate Whisper settings (beam_size=5, quality filters enabled)
- Pause/resume hotkey (F12 default, with Wayland socket fallback)
- Voice command detection (window management, text editing, custom commands)
"""

import argparse
import numpy as np
import pyaudio
import webrtcvad
import collections
import subprocess
import sys
import signal
import time
import threading
import queue
import socket
import os
import secrets
import json
import logging
from logging.handlers import RotatingFileHandler
from faster_whisper import WhisperModel

# Voice command support
try:
    from commands import CommandDetector, CommandExecutor, create_default_config

    COMMANDS_AVAILABLE = True
except ImportError:
    COMMANDS_AVAILABLE = False

# GPU optimization imports
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Hotkey support (optional)
try:
    from pynput import keyboard

    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

# Audio visualization (optional)
try:
    from audio_visualizer import AudioVisualizer

    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False

# Streaming STT (optional - sherpa-onnx)
try:
    from streaming_stt import StreamingSTT

    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

# Direct uinput key injection (replaces ydotool for Wayland)
try:
    import evdev
    from evdev import UInput, ecodes
    import struct

    EVDEV_AVAILABLE = True
except ImportError:
    EVDEV_AVAILABLE = False


class FastKeyInjector:
    """Direct /dev/uinput key injection — sub-frame speed, no ydotool daemon.

    Writes all key events in a single write() syscall. 30 backspaces take ~0.3ms
    kernel processing time vs ~50-100ms through ydotool's socket IPC.
    """

    # Keycode mapping for typing printable characters via uinput.
    # Maps characters to (keycode, shift_needed) tuples.
    _CHAR_TO_KEY = {}

    # Build character map from evdev key names
    _UNSHIFTED = "`1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./ "
    _SHIFTED = '~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:"ZXCVBNM<>? '
    _KEYCODES = [
        # Row 1: ` 1-0 - =
        41,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        # Row 2: q-p [ ] backslash
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        43,
        # Row 3: a-l ; '
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        # Row 4: z-m , . / space
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        57,
    ]

    for _i, _ch in enumerate(_UNSHIFTED):
        _CHAR_TO_KEY[_ch] = (_KEYCODES[_i], False)
    for _i, _ch in enumerate(_SHIFTED):
        if _ch != " ":  # space already mapped
            _CHAR_TO_KEY[_ch] = (_KEYCODES[_i], True)

    # Special keys
    _CHAR_TO_KEY["\n"] = (28, False)  # KEY_ENTER
    _CHAR_TO_KEY["\t"] = (15, False)  # KEY_TAB

    KEY_BACKSPACE = 14
    KEY_LEFT = 105
    KEY_LSHIFT = 42
    KEY_LCTRL = 29
    KEY_V = 47

    def __init__(self):
        cap = {ecodes.EV_KEY: list(range(1, 249))}
        self.ui = UInput(cap, name="voice-typing-kbd")
        self._event_struct = struct.Struct("llHHi")  # struct input_event (64-bit)

    def _make_event(self, etype, ecode, value):
        """Create raw input_event bytes."""
        return self._event_struct.pack(0, 0, etype, ecode, value)

    def _build_key_events(self, keycode, pressed):
        """Build press/release + SYN_REPORT as raw bytes."""
        ev = self._make_event(ecodes.EV_KEY, keycode, 1 if pressed else 0)
        syn = self._make_event(ecodes.EV_SYN, ecodes.SYN_REPORT, 0)
        return ev + syn

    def send_backspaces(self, count):
        """Send N backspaces, batched in groups for terminal compatibility.

        Terminals (Ghostty, kitty, etc.) can't process 50+ backspaces in a
        single ~0.3ms burst — their input loops miss events. We batch in
        groups of 10 with a tiny sleep between batches. Still completes
        100 backspaces in ~3ms (well under one 16ms display frame).
        GUI apps handle the burst fine regardless.
        """
        if count <= 0:
            return
        batch_size = 10
        remaining = count
        while remaining > 0:
            n = min(remaining, batch_size)
            buf = bytearray()
            for _ in range(n):
                buf.extend(self._build_key_events(self.KEY_BACKSPACE, True))
                buf.extend(self._build_key_events(self.KEY_BACKSPACE, False))
            os.write(self.ui.fd, bytes(buf))
            remaining -= n
            if remaining > 0:
                time.sleep(0.0003)  # 0.3ms between batches

    def type_text(self, text):
        """Type a string by emitting key events, batched for terminal compatibility.

        Characters are sent in groups of 20 with 0.3ms sleep between batches.
        This prevents overwhelming terminal input loops while still completing
        a 400-char string in ~6ms (well under one 16ms display frame).
        Characters not in the keymap are silently skipped.
        """
        if not text:
            return
        batch_size = 20
        i = 0
        while i < len(text):
            buf = bytearray()
            batch_end = min(i + batch_size, len(text))
            for ch in text[i:batch_end]:
                mapping = self._CHAR_TO_KEY.get(ch)
                if mapping is None:
                    continue
                keycode, shift = mapping
                if shift:
                    buf.extend(self._build_key_events(self.KEY_LSHIFT, True))
                buf.extend(self._build_key_events(keycode, True))
                buf.extend(self._build_key_events(keycode, False))
                if shift:
                    buf.extend(self._build_key_events(self.KEY_LSHIFT, False))
            if buf:
                os.write(self.ui.fd, bytes(buf))
            i = batch_end
            if i < len(text):
                time.sleep(0.0003)  # 0.3ms between batches

    def type_text_burst(self, text):
        """Type a string in a single write() syscall for sub-frame atomic speed.

        All key events are packed into one kernel call. Works perfectly in GUI
        apps (browsers, editors) but can overwhelm terminal input loops.
        Use type_text() for terminal-safe batched typing instead.
        """
        if not text:
            return
        buf = bytearray()
        for ch in text:
            mapping = self._CHAR_TO_KEY.get(ch)
            if mapping is None:
                continue
            keycode, shift = mapping
            if shift:
                buf.extend(self._build_key_events(self.KEY_LSHIFT, True))
            buf.extend(self._build_key_events(keycode, True))
            buf.extend(self._build_key_events(keycode, False))
            if shift:
                buf.extend(self._build_key_events(self.KEY_LSHIFT, False))
        if buf:
            os.write(self.ui.fd, bytes(buf))

    def replace_text(self, chars_to_delete, new_text):
        """Replace text in a single write() syscall — true sub-frame atomic.

        Both deletion and insertion happen in one kernel call.
        This works perfectly in GUI apps (browsers, editors). For terminals,
        use send_backspaces() + type_text() separately instead.
        """
        buf = bytearray()
        for _ in range(chars_to_delete):
            buf.extend(self._build_key_events(self.KEY_BACKSPACE, True))
            buf.extend(self._build_key_events(self.KEY_BACKSPACE, False))
        for ch in new_text:
            mapping = self._CHAR_TO_KEY.get(ch)
            if mapping is None:
                continue
            keycode, shift = mapping
            if shift:
                buf.extend(self._build_key_events(self.KEY_LSHIFT, True))
            buf.extend(self._build_key_events(keycode, True))
            buf.extend(self._build_key_events(keycode, False))
            if shift:
                buf.extend(self._build_key_events(self.KEY_LSHIFT, False))
        if buf:
            os.write(self.ui.fd, bytes(buf))

    def close(self):
        """Release the uinput device."""
        try:
            self.ui.close()
        except Exception:
            pass


class IBusClient:
    """Client for communicating with the IBus voice typing engine.

    Maintains a persistent Unix socket connection. Commands are newline-terminated:
      preedit:TEXT     - streaming partial (shown as underlined preedit)
      commit:TEXT      - atomic text insertion (clears preedit first)
      delete:N         - delete N chars before cursor
      replace:N:TEXT   - delete N chars then commit TEXT
    """

    SOCKET_PATH = os.path.join(
        os.environ.get("XDG_RUNTIME_DIR", "/tmp"),
        f"voice-typing-ibus-{os.getuid()}.sock",
    )

    def __init__(self):
        self._sock = None
        self._lock = threading.Lock()

    @property
    def is_available(self):
        """Check if IBus engine socket exists."""
        return os.path.exists(self.SOCKET_PATH)

    def _ensure_connected(self):
        if self._sock is not None:
            return True
        if not os.path.exists(self.SOCKET_PATH):
            return False
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.SOCKET_PATH)
            self._sock = sock
            return True
        except (ConnectionError, OSError):
            self._sock = None
            return False

    def _send(self, message):
        with self._lock:
            if not self._ensure_connected():
                return False
            try:
                self._sock.sendall((message + "\n").encode("utf-8"))
                return True
            except (BrokenPipeError, ConnectionError, OSError):
                self._sock = None
                if not self._ensure_connected():
                    return False
                try:
                    self._sock.sendall((message + "\n").encode("utf-8"))
                    return True
                except (BrokenPipeError, ConnectionError, OSError):
                    self._sock = None
                    return False

    def send_preedit(self, text):
        return self._send(f"preedit:{text}")

    def send_commit(self, text):
        return self._send(f"commit:{text}")

    def send_delete(self, count):
        return self._send(f"delete:{count}")

    def send_replace(self, count, text):
        return self._send(f"replace:{count}:{text}")

    def close(self):
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None


# Hotkey name to pynput format mapping
HOTKEY_MAP = {
    "f12": "<f12>",
    "f11": "<f11>",
    "f10": "<f10>",
    "scroll_lock": "<scroll_lock>",
    "pause": "<pause>",
}

# Socket for Wayland fallback (per-user, permissioned)
RUNTIME_DIR = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
SOCKET_PATH = os.path.join(RUNTIME_DIR, f"voice-typing-{os.getuid()}.sock")
TOKEN_PATH = os.path.join(RUNTIME_DIR, f"voice-typing-{os.getuid()}.token")

XDG_CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
XDG_STATE_HOME = os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state"))
DEFAULT_CONFIG_PATH = os.path.join(XDG_CONFIG_HOME, "voice-typing", "config.yaml")
DEFAULT_LOG_DIR = os.path.join(XDG_STATE_HOME, "voice-typing")
DEFAULT_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "voice-typing.log")

CONFIG_DEFAULTS = {
    "model": "base",
    "device": "auto",
    "language": None,
    "hotkey": "f12",
    "commands": False,
    "commands_file": "~/.config/voice-typing/commands.yaml",
    "command_arm": False,
    "command_arm_seconds": 10,
    "command_min_confidence": 0.8,
    "command_confirm_below": 0.9,
    "command_confirm_seconds": 5.0,
    "allow_shell": False,
    "max_seconds": 30,
    "queue_size": 2,
    "calibrate_seconds": 1.0,
    "noise_gate": False,
    "noise_gate_multiplier": 1.5,
    "agc": False,
    "agc_target_rms": 4000.0,
    "agc_min_gain": 0.5,
    "agc_max_gain": 3.0,
    "adaptive_vad": True,
    "notify": False,
    "status_interval": 0.0,
    "input_device": None,
    "ptt": False,
    "ptt_hotkey": "f9",
    "ptt_mode": "hold",
    "log_file": DEFAULT_LOG_FILE,
    "log_max_bytes": 1_000_000,
    "log_backups": 5,
    "log_level": "INFO",
    "config": DEFAULT_CONFIG_PATH,
    "viz": False,
    "viz_position": "bottom-right",
    "viz_hide_delay": 1500,
    "streaming": False,
    "streaming_model": "zipformer-en",
    "refinement": False,
    "refinement_model": "large-v3-turbo",
}


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on", "y")


def _load_config(path: str) -> dict:
    if not path:
        return {}
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return {}

    try:
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml

                with open(path, "r") as f:
                    data = yaml.safe_load(f) or {}
                    return data if isinstance(data, dict) else {}
            except Exception as e:
                print(f"⚠️  Failed to load YAML config: {e}")
                return {}
        if path.endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"⚠️  Failed to load config: {e}")

    return {}


def _apply_env_overrides(config: dict) -> dict:
    overrides = {}
    mapping = {
        "model": "VOICE_MODEL",
        "device": "VOICE_DEVICE",
        "language": "VOICE_LANGUAGE",
        "hotkey": "VOICE_HOTKEY",
        "commands": "VOICE_COMMANDS",
        "commands_file": "VOICE_COMMANDS_FILE",
        "command_arm": "VOICE_COMMAND_ARM",
        "command_arm_seconds": "VOICE_COMMAND_ARM_SECONDS",
        "command_min_confidence": "VOICE_COMMAND_MIN_CONFIDENCE",
        "command_confirm_below": "VOICE_COMMAND_CONFIRM_BELOW",
        "command_confirm_seconds": "VOICE_COMMAND_CONFIRM_SECONDS",
        "allow_shell": "VOICE_ALLOW_SHELL",
        "max_seconds": "VOICE_MAX_SECONDS",
        "queue_size": "VOICE_QUEUE_SIZE",
        "calibrate_seconds": "VOICE_CALIBRATE_SECONDS",
        "noise_gate": "VOICE_NOISE_GATE",
        "noise_gate_multiplier": "VOICE_NOISE_GATE_MULTIPLIER",
        "agc": "VOICE_AGC",
        "agc_target_rms": "VOICE_AGC_TARGET_RMS",
        "agc_min_gain": "VOICE_AGC_MIN_GAIN",
        "agc_max_gain": "VOICE_AGC_MAX_GAIN",
        "adaptive_vad": "VOICE_ADAPTIVE_VAD",
        "notify": "VOICE_NOTIFY",
        "status_interval": "VOICE_STATUS_INTERVAL",
        "input_device": "VOICE_INPUT_DEVICE",
        "ptt": "VOICE_PTT",
        "ptt_hotkey": "VOICE_PTT_HOTKEY",
        "ptt_mode": "VOICE_PTT_MODE",
        "log_file": "VOICE_LOG_FILE",
        "log_max_bytes": "VOICE_LOG_MAX_BYTES",
        "log_backups": "VOICE_LOG_BACKUPS",
        "log_level": "VOICE_LOG_LEVEL",
        "viz": "VOICE_VIZ",
        "viz_position": "VOICE_VIZ_POSITION",
        "viz_hide_delay": "VOICE_VIZ_HIDE_DELAY",
        "streaming": "VOICE_STREAMING",
        "streaming_model": "VOICE_STREAMING_MODEL",
        "refinement": "VOICE_REFINEMENT",
        "refinement_model": "VOICE_REFINEMENT_MODEL",
    }

    for key, env_var in mapping.items():
        if env_var in os.environ:
            overrides[key] = os.environ[env_var]

    if "VOICE_NO_ADAPTIVE_VAD" in os.environ and "adaptive_vad" not in overrides:
        overrides["adaptive_vad"] = not _parse_bool(os.environ["VOICE_NO_ADAPTIVE_VAD"])

    if not overrides:
        return config

    merged = dict(config)
    for key, value in overrides.items():
        if key in (
            "commands",
            "command_arm",
            "allow_shell",
            "noise_gate",
            "agc",
            "notify",
            "ptt",
            "adaptive_vad",
            "viz",
            "streaming",
            "refinement",
        ):
            merged[key] = _parse_bool(value)
        elif key in (
            "command_arm_seconds",
            "max_seconds",
            "queue_size",
            "log_max_bytes",
            "log_backups",
            "viz_hide_delay",
        ):
            merged[key] = int(value)
        elif key in (
            "command_min_confidence",
            "command_confirm_below",
            "command_confirm_seconds",
            "calibrate_seconds",
            "noise_gate_multiplier",
            "agc_target_rms",
            "agc_min_gain",
            "agc_max_gain",
            "status_interval",
        ):
            merged[key] = float(value)
        else:
            merged[key] = value

    return merged


def _setup_logging(
    log_file: str, level: str = "INFO", max_bytes: int = 1_000_000, backups: int = 5
):
    if not log_file:
        return None

    log_file = os.path.expanduser(log_file)
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("voice_typing")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backups)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def detect_display_server():
    """Detect if running on Wayland or X11"""
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    wayland_display = os.environ.get("WAYLAND_DISPLAY", "")

    if session_type == "wayland" or wayland_display:
        return "wayland"
    return "x11"


def check_ydotool_daemon():
    """Check if ydotool daemon is running (required for Wayland)"""
    try:
        result = subprocess.run(["pgrep", "-x", "ydotoold"], capture_output=True)
        return result.returncode == 0
    except:
        return False


class VoiceTyping:
    def __init__(
        self,
        model_size="base",
        device="cpu",
        language=None,
        hotkey="f12",
        commands_enabled=False,
        commands_file=None,
        require_command_arm=False,
        command_arm_seconds=10,
        allow_shell_commands=False,
        max_recording_seconds=30,
        queue_size=2,
        calibration_seconds=1.0,
        noise_gate_enabled=False,
        noise_gate_multiplier=1.5,
        agc_enabled=False,
        agc_target_rms=4000.0,
        agc_min_gain=0.5,
        agc_max_gain=3.0,
        adaptive_vad=True,
        command_min_confidence=0.8,
        command_confirm_below=0.9,
        command_confirm_seconds=5.0,
        input_device=None,
        notify=False,
        status_interval=0.0,
        ptt_enabled=False,
        ptt_hotkey="f9",
        ptt_mode="hold",
        logger=None,
        viz_enabled=False,
        viz_position="bottom-right",
        viz_hide_delay=1500,
        streaming_enabled=False,
        streaming_model="zipformer-en",
        refinement_enabled=False,
        refinement_model="large-v3-turbo",
    ):
        self.model_size = model_size
        self.device = device
        self.language = language
        self.hotkey = hotkey
        self.commands_enabled = commands_enabled
        self.require_command_arm = require_command_arm
        self.command_arm_seconds = command_arm_seconds
        self.allow_shell_commands = allow_shell_commands
        self.command_min_confidence = command_min_confidence
        self.command_confirm_below = command_confirm_below
        self.command_confirm_seconds = command_confirm_seconds
        self.input_device = input_device
        self.notify_enabled = notify
        self.status_interval = status_interval
        self.ptt_enabled = ptt_enabled
        self.ptt_hotkey = ptt_hotkey
        self.ptt_mode = ptt_mode
        self.logger = logger

        # Detect display server (Wayland vs X11)
        self.display_server = detect_display_server()
        print(f"🖥️  Display server: {self.display_server.upper()}")

        # IBus client for atomic text insertion (preferred over key injection)
        self.ibus_client = IBusClient()
        if self.ibus_client.is_available:
            print("✅ IBus engine available (atomic text insertion)")
        else:
            print("ℹ️  IBus engine not running (using key injection fallback)")

        # Initialize key injection backend
        self.key_injector = None
        if self.display_server == "wayland":
            if EVDEV_AVAILABLE:
                try:
                    self.key_injector = FastKeyInjector()
                    print(f"✅ Direct uinput key injection (sub-frame speed)")
                except Exception as e:
                    print(f"⚠️  Failed to create uinput device: {e}")
                    print(f"   Falling back to ydotool (needs ydotoold)")
                    if check_ydotool_daemon():
                        print(f"✅ ydotoold daemon running")
                    else:
                        print(f"❌ ydotoold daemon NOT running!")
                        print(f"   Start with: sudo ydotoold &")
            else:
                print(f"⚠️  python-evdev not installed, using ydotool fallback")
                if check_ydotool_daemon():
                    print(f"✅ ydotoold daemon running")
                else:
                    print(f"❌ ydotoold daemon NOT running!")
                    print(f"   Keyboard input won't work without it.")
                    print(f"   Start with: sudo ydotoold &")

        # Initialize command detection if enabled
        self.command_detector = None
        self.command_executor = None
        if commands_enabled and COMMANDS_AVAILABLE:
            self.command_detector = CommandDetector(
                custom_commands_path=commands_file, enabled=True
            )
            self.command_executor = CommandExecutor(
                voice_typing=self, allow_shell_commands=allow_shell_commands
            )
            print(f"🎯 Command detection enabled")
        elif commands_enabled and not COMMANDS_AVAILABLE:
            print(f"⚠️  Commands requested but commands.py not found")

        # Audio configuration
        self.sample_rate = 16000
        self.chunk_size = 320  # 20ms at 16kHz
        self.channels = 1
        self.format = pyaudio.paInt16

        # Limits to avoid unbounded buffers under slow transcription
        self.max_recording_seconds = max_recording_seconds
        self.max_recording_chunks = int(
            self.sample_rate / self.chunk_size * self.max_recording_seconds
        )
        self.dropped_transcriptions = 0
        self.last_audio_status_log = 0.0
        self.last_callback_time = time.time()
        self._stream_reset_needed = False
        self.last_vad_update = 0.0

        # Initialize components
        self.audio = None
        self.stream = None
        self.vad = None
        self.model = None
        self.running = False
        self.input_device_index = None

        # Pause state
        self.is_paused = False
        self.hotkey_listener = None
        self.socket_server = None
        self.socket_thread = None
        self.socket_token = None
        self.bad_socket_tokens = 0

        # Voice activity detection (accuracy-optimized settings)
        self.vad_aggressiveness = 2  # Level 2: better noise rejection
        self.vad_mode = self.vad_aggressiveness
        self.adaptive_vad = adaptive_vad
        self.pre_buffer_size = 30  # 600ms at 20ms chunks
        self.post_buffer_size = 40  # 800ms after silence

        # Noise handling
        self.calibration_seconds = calibration_seconds
        self.noise_gate_enabled = noise_gate_enabled
        self.noise_gate_multiplier = noise_gate_multiplier
        self.noise_floor_rms = 0.0
        self.ambient_rms_ema = 0.0
        self.ambient_ema_alpha = 0.95
        self.agc_enabled = agc_enabled
        self.agc_target_rms = agc_target_rms
        self.agc_min_gain = agc_min_gain
        self.agc_max_gain = agc_max_gain

        # Audio buffers with thread safety
        self.pre_buffer = collections.deque(maxlen=self.pre_buffer_size)
        self.recording_buffer = []
        self.is_recording = False
        self.silence_chunks = 0
        self.buffer_lock = threading.Lock()

        # Context for transcription continuity
        self.previous_text = ""
        self.pinned_buffer = None

        # Thread-safe transcription queue (bounded to avoid memory spikes)
        self.transcription_queue = queue.Queue(maxsize=queue_size)
        self.transcription_thread = None

        # Typing history for scratch that
        self.typing_history = []
        self.max_history = 20
        self.commands_armed_until = 0.0
        self.last_stream_restart = 0.0
        self.pending_command = None
        self.ptt_active = False
        self.ptt_listener = None

        self.last_status_log = 0.0

        # Audio visualizer
        self.visualizer = None
        self.viz_enabled = viz_enabled
        self.viz_position = viz_position
        self.viz_hide_delay = viz_hide_delay

        # Streaming STT (sherpa-onnx streaming + optional faster-whisper refinement)
        self.streaming_enabled = streaming_enabled and STREAMING_AVAILABLE
        self.streaming_model = streaming_model
        self.refinement_enabled = refinement_enabled and streaming_enabled
        self.refinement_model = refinement_model
        self.streaming_stt = None
        self.streaming_thread = None
        self.streaming_queue = queue.Queue(maxsize=200)  # ~4s of 20ms chunks
        self.current_streaming_text = ""
        self.streaming_lock = threading.Lock()

        if streaming_enabled and not STREAMING_AVAILABLE:
            print("streaming requested but sherpa-onnx not installed")
            print("   Install with: pip install sherpa-onnx")

        # Model performance suggestions
        if device == "cuda":
            if model_size in ["large", "large-v3"]:
                print(f"💡 Performance tip: Consider 'distil-large-v3' for 1.5x speed")
            elif model_size == "small":
                print(f"💡 Performance tip: For better accuracy, try 'distil-medium'")

        print(f"Initializing Voice Typing (model: {model_size}, device: {device})")
        self.initialize()

    def initialize(self):
        """Initialize all components"""
        try:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()

            # Resolve input device
            self.input_device_index = self._resolve_input_device()
            if self.input_device is not None and self.input_device_index is None:
                print(f"⚠️  Input device not found: {self.input_device} (using default)")
            elif self.input_device_index is not None:
                info = self.audio.get_device_info_by_index(self.input_device_index)
                print(
                    f"🎙️  Using input device [{self.input_device_index}]: {info.get('name', 'unknown')}"
                )

            # Calibrate ambient noise floor before VAD starts
            self._calibrate_noise_floor()

            # Advanced GPU optimizations
            if self.device == "cuda" and TORCH_AVAILABLE:
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cudnn.benchmark = True

                    if torch.cuda.is_available():
                        torch.cuda.set_per_process_memory_fraction(0.9)
                        torch.cuda.empty_cache()

                        try:
                            self.pinned_buffer = torch.cuda.FloatTensor(
                                16000 * 10
                            ).pin_memory()
                            print(
                                f"🚀 GPU optimizations enabled: Tensor Cores + Pinned Buffers"
                            )
                        except:
                            print(
                                f"🚀 GPU optimizations enabled: Tensor Cores + Memory"
                            )
                except Exception as e:
                    print(f"⚠️  GPU optimizations partially failed: {e}")

            # Initialize VAD with better noise rejection
            self.vad = webrtcvad.Vad(self.vad_aggressiveness)
            self.vad_mode = self.vad_aggressiveness

            # Initialize streaming STT if enabled
            if self.streaming_enabled:
                from streaming_stt import StreamingSTT

                self.streaming_stt = StreamingSTT(
                    model_name=self.streaming_model,
                    sample_rate=self.sample_rate,
                )
                self.streaming_stt.create_recognizer()
                # In streaming mode, reduce post-buffer since endpoint detection
                # is handled by sherpa-onnx
                self.post_buffer_size = 20  # 400ms instead of 800ms

            # Initialize Whisper model (skip if streaming-only without refinement)
            if not self.streaming_enabled or self.refinement_enabled:
                whisper_model = (
                    self.refinement_model
                    if self.refinement_enabled
                    else self.model_size
                )
                print(f"Loading Whisper model '{whisper_model}' on {self.device}...")
                self.model = WhisperModel(
                    whisper_model,
                    device=self.device,
                    compute_type="int8_float16" if self.device == "cuda" else "int8",
                )
                print("Model loaded successfully!")

                # Warm up the model
                print("Warming up model...")
                dummy_audio = np.zeros(16000, dtype=np.float32)
                list(self.model.transcribe(dummy_audio, language=self.language or "en"))
                print("Model warmed up!")
            else:
                print("Streaming-only mode (no Whisper refinement)")
                self.model = None

            # Initialize audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback,
            )

            print(f"Audio stream initialized (sample rate: {self.sample_rate} Hz)")

            # Start audio visualizer if enabled
            if self.viz_enabled and VISUALIZER_AVAILABLE:
                self.visualizer = AudioVisualizer(
                    position=self.viz_position,
                    hide_delay_ms=self.viz_hide_delay,
                    sample_rate=self.sample_rate,
                )
                self.visualizer.start()
                print(f"🎨 Audio visualizer enabled ({self.viz_position})")
            elif self.viz_enabled and not VISUALIZER_AVAILABLE:
                print(f"⚠️  Visualizer requested but audio_visualizer.py not available")

        except Exception as e:
            print(f"Initialization error: {e}")
            self.cleanup()
            raise

    def _toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            print(f"\n⏸️  PAUSED - Press {self.hotkey.upper()} to resume")
            self._notify("Voice typing", "Paused")
        else:
            print(f"\n▶️  RESUMED - Listening...")
            self._notify("Voice typing", "Resumed")

    def _start_hotkey_listener(self):
        """Start hotkey listener - tries pynput, falls back to socket for Wayland"""
        hotkey_started = False

        # Try pynput first (works on X11 and XWayland)
        if PYNPUT_AVAILABLE:
            try:
                hotkey_str = HOTKEY_MAP.get(self.hotkey, f"<{self.hotkey}>")
                self.hotkey_listener = keyboard.GlobalHotKeys(
                    {hotkey_str: self._toggle_pause}
                )
                self.hotkey_listener.start()

                # Test if it actually works (may fail silently on Wayland)
                time.sleep(0.1)
                if self.hotkey_listener.is_alive():
                    print(f"🎹 Hotkey {self.hotkey.upper()} registered (pynput)")
                    hotkey_started = True
            except Exception as e:
                print(f"⚠️  pynput hotkey failed: {e}")

        if self.ptt_enabled and PYNPUT_AVAILABLE:
            try:
                self._start_ptt_listener()
            except Exception as e:
                print(f"⚠️  PTT listener failed: {e}")

        # Always start socket server as fallback (useful on Wayland)
        self._start_socket_server()

        if not hotkey_started:
            print(f"⚠️  pynput not available - using socket fallback for Wayland")
            self._print_wayland_instructions()

    def _start_socket_server(self):
        """Start Unix socket server for external pause control (Wayland fallback)"""
        # Remove existing socket
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        if os.path.exists(TOKEN_PATH):
            try:
                os.remove(TOKEN_PATH)
            except:
                pass

        try:
            socket_dir = os.path.dirname(SOCKET_PATH)
            if socket_dir:
                os.makedirs(socket_dir, exist_ok=True)

            old_umask = os.umask(0o077)
            self.socket_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                self.socket_server.bind(SOCKET_PATH)
            finally:
                os.umask(old_umask)
            os.chmod(SOCKET_PATH, 0o600)
            self.socket_server.listen(1)
            self.socket_server.settimeout(0.5)  # Allow periodic check for shutdown
            self._write_socket_token()

            self.socket_thread = threading.Thread(
                target=self._socket_listener, daemon=True, name="SocketListener"
            )
            self.socket_thread.start()
        except Exception as e:
            print(f"⚠️  Socket server failed: {e}")

    def _write_socket_token(self):
        """Create a per-session token to guard the socket."""
        try:
            self.socket_token = secrets.token_hex(16)
            with open(TOKEN_PATH, "w") as f:
                f.write(self.socket_token)
            os.chmod(TOKEN_PATH, 0o600)
        except Exception as e:
            self.socket_token = None
            print(f"⚠️  Failed to write socket token: {e}")

    def _socket_listener(self):
        """Listen for pause/resume commands via socket"""
        while self.running:
            try:
                conn, _ = self.socket_server.accept()
                data = conn.recv(128).decode(errors="ignore").strip()
                conn.close()

                parts = data.split()
                if not parts:
                    continue

                cmd = parts[0]
                token = parts[1] if len(parts) > 1 else None
                if self.socket_token and token != self.socket_token:
                    self.bad_socket_tokens += 1
                    if self.bad_socket_tokens == 1:
                        print(
                            "⚠️  Invalid socket token received (further warnings suppressed)"
                        )
                    continue

                if cmd in ("toggle", "pause", "resume"):
                    if cmd == "toggle":
                        self._toggle_pause()
                    elif cmd == "pause" and not self.is_paused:
                        self._toggle_pause()
                    elif cmd == "resume" and self.is_paused:
                        self._toggle_pause()
                elif cmd in ("ptt_down", "ptt_up", "ptt_toggle"):
                    if not self.ptt_enabled:
                        continue
                    if cmd == "ptt_down":
                        self._set_ptt(True)
                    elif cmd == "ptt_up":
                        self._set_ptt(False)
                    else:
                        self._set_ptt(not self.ptt_active)
            except socket.timeout:
                continue
            except Exception:
                if self.running:
                    continue
                break

    def _print_wayland_instructions(self):
        """Print instructions for Wayland hotkey setup"""
        print("\n" + "=" * 50)
        print("📋 WAYLAND HOTKEY SETUP")
        print("=" * 50)
        print(f"Run this to toggle pause with {self.hotkey.upper()}:")
        print(f"  ./voice-toggle")
        print("")
        print("Or add to your compositor's keybindings:")
        if os.path.exists(TOKEN_PATH):
            print(f'  Command: echo "toggle $(cat {TOKEN_PATH})" | nc -U {SOCKET_PATH}')
        else:
            print(f"  Command: echo toggle | nc -U {SOCKET_PATH}")
        print("")
        print("For xbindkeys, add to ~/.xbindkeysrc:")
        if os.path.exists(TOKEN_PATH):
            print(f'  "echo toggle $(cat {TOKEN_PATH}) | nc -U {SOCKET_PATH}"')
        else:
            print(f'  "echo toggle | nc -U {SOCKET_PATH}"')
        print(f"    {self.hotkey.upper()}")
        if self.ptt_enabled:
            if os.path.exists(TOKEN_PATH):
                print("PTT commands via socket:")
                print(f'  echo "ptt_down $(cat {TOKEN_PATH})" | nc -U {SOCKET_PATH}')
                print(f'  echo "ptt_up $(cat {TOKEN_PATH})" | nc -U {SOCKET_PATH}')
            else:
                print(f"  echo ptt_down | nc -U {SOCKET_PATH}")
                print(f"  echo ptt_up | nc -U {SOCKET_PATH}")
        print("=" * 50 + "\n")

    def _log(self, message: str, level: str = "info"):
        """Log to file if configured."""
        if not self.logger:
            return
        log_fn = getattr(self.logger, level, None)
        if log_fn:
            log_fn(message)

    def _status_snapshot(self) -> str:
        """Build a short status line."""
        try:
            qsize = self.transcription_queue.qsize()
        except Exception:
            qsize = 0
        status = "paused" if self.is_paused else "active"
        ptt = "ptt:on" if self.ptt_active else "ptt:off"
        streaming = "streaming:on" if self.streaming_enabled else "streaming:off"
        return f"status={status} queue={qsize} dropped={self.dropped_transcriptions} vad={self.vad_mode} {ptt} {streaming}"

    def _notify(self, title: str, body: str | None = None):
        """Send desktop notification if enabled."""
        if not self.notify_enabled:
            return
        try:
            args = ["notify-send", title]
            if body:
                args.append(body)
            subprocess.run(args, check=False)
        except Exception:
            pass

    def _parse_pynput_key(self, key_str: str):
        """Parse a key name into pynput key or char."""
        key_str = key_str.lower().strip()
        special_map = {
            "f1": keyboard.Key.f1,
            "f2": keyboard.Key.f2,
            "f3": keyboard.Key.f3,
            "f4": keyboard.Key.f4,
            "f5": keyboard.Key.f5,
            "f6": keyboard.Key.f6,
            "f7": keyboard.Key.f7,
            "f8": keyboard.Key.f8,
            "f9": keyboard.Key.f9,
            "f10": keyboard.Key.f10,
            "f11": keyboard.Key.f11,
            "f12": keyboard.Key.f12,
            "pause": keyboard.Key.pause,
            "scroll_lock": keyboard.Key.scroll_lock,
        }
        if key_str in special_map:
            return special_map[key_str]
        if len(key_str) == 1:
            return key_str
        return None

    def _set_ptt(self, active: bool):
        """Set push-to-talk active state."""
        if self.ptt_active == active:
            return
        self.ptt_active = active
        state = "ON" if active else "OFF"
        print(f"🎙️  PTT {state}")
        self._notify("Push-to-talk", f"{state}")

    def _start_ptt_listener(self):
        """Start push-to-talk listener."""
        key = self._parse_pynput_key(self.ptt_hotkey)
        if key is None:
            raise ValueError(f"Unknown PTT hotkey: {self.ptt_hotkey}")

        def on_press(k):
            if k == key or (hasattr(k, "char") and k.char == key):
                if self.ptt_mode == "toggle":
                    self._set_ptt(not self.ptt_active)
                else:
                    self._set_ptt(True)

        def on_release(k):
            if self.ptt_mode == "hold":
                if k == key or (hasattr(k, "char") and k.char == key):
                    self._set_ptt(False)

        self.ptt_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.ptt_listener.daemon = True
        self.ptt_listener.start()

    def _resolve_input_device(self):
        """Resolve input device index by id or name substring."""
        if self.input_device is None:
            return None

        try:
            device_count = self.audio.get_device_count()
        except Exception:
            return None

        # Numeric index
        if isinstance(self.input_device, int):
            return self.input_device
        if isinstance(self.input_device, str) and self.input_device.isdigit():
            return int(self.input_device)

        # Name substring match
        needle = str(self.input_device).lower()
        for idx in range(device_count):
            info = self.audio.get_device_info_by_index(idx)
            if (
                info.get("maxInputChannels", 0) > 0
                and needle in info.get("name", "").lower()
            ):
                return idx

        return None

    @staticmethod
    def list_input_devices():
        """List available input devices."""
        audio = pyaudio.PyAudio()
        try:
            device_count = audio.get_device_count()
            print("Input devices:")
            for idx in range(device_count):
                info = audio.get_device_info_by_index(idx)
                if info.get("maxInputChannels", 0) > 0:
                    name = info.get("name", "unknown")
                    default_tag = " (default)" if info.get("defaultSampleRate") else ""
                    print(f"  [{idx}] {name}{default_tag}")
        finally:
            audio.terminate()

    def _rms(self, audio_chunk: np.ndarray) -> float:
        """Compute RMS for an int16 audio chunk."""
        if audio_chunk.size == 0:
            return 0.0
        float_chunk = audio_chunk.astype(np.float32, copy=False)
        return float(np.sqrt(np.mean(float_chunk * float_chunk)))

    def _apply_agc(
        self, audio_chunk: np.ndarray, rms: float
    ) -> tuple[np.ndarray, float]:
        """Apply automatic gain control to reach target RMS."""
        if rms <= 0.0:
            return audio_chunk, rms

        gain = self.agc_target_rms / rms
        gain = max(self.agc_min_gain, min(self.agc_max_gain, gain))
        if gain == 1.0:
            return audio_chunk, rms

        scaled = audio_chunk.astype(np.float32, copy=False) * gain
        np.clip(scaled, -32768.0, 32767.0, out=scaled)
        adjusted = scaled.astype(np.int16)
        return adjusted, self._rms(adjusted)

    def _noise_gate_threshold(self) -> float:
        """Compute noise gate threshold from ambient noise floor."""
        floor = self.noise_floor_rms if self.noise_floor_rms > 0.0 else 100.0
        return floor * self.noise_gate_multiplier

    def _update_ambient(self, rms: float):
        """Update ambient noise estimate."""
        if rms <= 0.0:
            return
        if self.ambient_rms_ema == 0.0:
            self.ambient_rms_ema = rms
        else:
            self.ambient_rms_ema = (
                self.ambient_ema_alpha * self.ambient_rms_ema
                + (1.0 - self.ambient_ema_alpha) * rms
            )

    def _maybe_update_vad_mode(self):
        """Adjust VAD aggressiveness based on ambient noise."""
        if not self.adaptive_vad:
            return
        now = time.time()
        if now - self.last_vad_update < 1.0:
            return

        floor = self.noise_floor_rms if self.noise_floor_rms > 0.0 else 100.0
        ratio = self.ambient_rms_ema / floor if floor > 0.0 else 1.0

        if ratio < 1.5:
            target = 1
        elif ratio < 3.0:
            target = 2
        else:
            target = 3

        if target != self.vad_mode:
            try:
                self.vad.set_mode(target)
                self.vad_mode = target
                self.last_vad_update = now
                print(f"🎚️  VAD aggressiveness set to {target}")
            except Exception:
                pass

    def _calibrate_noise_floor(self):
        """Sample ambient audio to estimate noise floor."""
        if self.calibration_seconds <= 0:
            return

        try:
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
            )
            samples = int(self.sample_rate * self.calibration_seconds / self.chunk_size)
            rms_values = []
            for _ in range(samples):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16)
                rms_values.append(self._rms(chunk))
            stream.stop_stream()
            stream.close()

            if rms_values:
                self.noise_floor_rms = max(50.0, float(np.median(rms_values)))
                self.ambient_rms_ema = self.noise_floor_rms
                print(f"🔇 Noise floor calibrated (RMS≈{self.noise_floor_rms:.0f})")
        except Exception as e:
            print(f"⚠️  Noise calibration failed: {e}")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback - MUST be fast, no blocking operations"""
        if not self.running:
            return (None, pyaudio.paComplete)

        self.last_callback_time = time.time()

        # Skip all processing when paused
        if self.is_paused:
            return (None, pyaudio.paContinue)

        # Push-to-talk gate
        if self.ptt_enabled and not self.ptt_active:
            return (None, pyaudio.paContinue)

        # Check for audio overflow
        if status:
            now = time.time()
            if now - self.last_audio_status_log > 5:
                if status & pyaudio.paInputOverflow:
                    print("⚠️ Audio buffer overflow")
                if status & pyaudio.paInputUnderflow:
                    print("⚠️ Audio buffer underflow")
                self.last_audio_status_log = now

        raw_chunk = np.frombuffer(in_data, dtype=np.int16)
        raw_rms = self._rms(raw_chunk)

        if self.noise_gate_enabled and raw_rms < self._noise_gate_threshold():
            is_speech = False
        else:
            is_speech = self.vad.is_speech(in_data, self.sample_rate)

        # Push to streaming STT queue (non-blocking, every chunk)
        if self.streaming_enabled:
            try:
                self.streaming_queue.put_nowait(raw_chunk.copy())
            except queue.Full:
                pass  # Drop if queue full - streaming is best-effort

        # Push to audio visualizer (non-blocking)
        if self.visualizer:
            self.visualizer.push_audio(raw_chunk)
            self.visualizer.set_speaking(is_speech)

        if not is_speech:
            self._update_ambient(raw_rms)
            self._maybe_update_vad_mode()

        audio_chunk = raw_chunk
        if self.agc_enabled:
            audio_chunk, _ = self._apply_agc(raw_chunk, raw_rms)

        # In streaming mode, the streaming_worker handles audio accumulation,
        # endpoint detection, and queuing for refinement. Skip the batch VAD path.
        if not self.streaming_enabled:
            with self.buffer_lock:
                if is_speech:
                    if not self.is_recording:
                        self.recording_buffer = list(self.pre_buffer)
                        self.is_recording = True

                    self.recording_buffer.append(audio_chunk.copy())
                    self.silence_chunks = 0
                    if len(self.recording_buffer) >= self.max_recording_chunks:
                        audio_to_process = self.recording_buffer.copy()
                        self._enqueue_transcription(audio_to_process)

                        self.is_recording = False
                        self.silence_chunks = 0
                        self.recording_buffer = []

                elif self.is_recording:
                    self.recording_buffer.append(audio_chunk.copy())
                    self.silence_chunks += 1

                    if len(self.recording_buffer) >= self.max_recording_chunks:
                        audio_to_process = self.recording_buffer.copy()
                        self._enqueue_transcription(audio_to_process)

                        self.is_recording = False
                        self.silence_chunks = 0
                        self.recording_buffer = []
                    elif self.silence_chunks >= self.post_buffer_size:
                        audio_to_process = self.recording_buffer.copy()
                        self._enqueue_transcription(audio_to_process)

                        self.is_recording = False
                        self.silence_chunks = 0
                        self.recording_buffer = []

                self.pre_buffer.append(audio_chunk.copy())

        return (None, pyaudio.paContinue)

    def _enqueue_transcription(self, audio_to_process, streaming_text=None):
        """Queue audio for transcription without blocking the audio callback."""
        item = (
            (audio_to_process, streaming_text)
            if self.streaming_enabled
            else audio_to_process
        )
        try:
            self.transcription_queue.put_nowait(item)
        except queue.Full:
            # Drop oldest to keep latency bounded
            self.dropped_transcriptions += 1
            try:
                self.transcription_queue.get_nowait()
                self.transcription_queue.task_done()
            except queue.Empty:
                pass
            try:
                self.transcription_queue.put_nowait(item)
            except queue.Full:
                self.dropped_transcriptions += 1

    def transcription_worker(self):
        """Background thread for Whisper transcription (also serves as refinement in streaming mode)"""
        while self.running:
            try:
                queue_item = self.transcription_queue.get(timeout=0.5)
                if self.streaming_enabled and isinstance(queue_item, tuple):
                    audio_buffer, streaming_text = queue_item
                    self._process_audio(audio_buffer, streaming_text=streaming_text)
                else:
                    audio_buffer = queue_item
                    self._process_audio(audio_buffer)
                if self.dropped_transcriptions:
                    dropped = self.dropped_transcriptions
                    self.dropped_transcriptions = 0
                    print(f"⚠️ Dropped {dropped} segment(s) due to backlog")
                self.transcription_queue.task_done()
            except queue.Empty:
                # Watchdog: if audio callback hasn't fired in 3s, flag stream for reset
                if not self.is_paused and time.time() - self.last_callback_time > 3.0:
                    print(
                        "⚠️  Audio stream stale (no callbacks for 3s), flagging reset..."
                    )
                    self._stream_reset_needed = True
                    self.last_callback_time = time.time()  # prevent repeated flags
                continue
            except Exception as e:
                print(f"❌ Transcription worker error: {e}")

    def _process_audio(self, recording_buffer, streaming_text=None):
        """Process recorded audio with Whisper.

        In streaming mode (streaming_text is not None), this acts as the refinement
        pass: compare turbo result with streaming output and correct if different.
        """
        if not recording_buffer:
            return

        try:
            audio_data = np.concatenate(recording_buffer)
            audio_float = audio_data.astype(np.float32, copy=False) / 32768.0

            is_refinement = self.streaming_enabled and streaming_text is not None

            if is_refinement:
                print("🔄 Refining...")
            else:
                print("⚡ Transcribing...")
            start_time = time.time()

            batch_prompt = (
                self.previous_text[-200:]
                if self.previous_text
                else "Clear speech dictation."
            )

            if is_refinement:
                # Refinement pass: turbo with beam_size=3 for reliability.
                # Uses clean prompt (only refined text, not streaming errors)
                # to prevent hallucination like "you" / "so" on long segments.
                refinement_prompt = (
                    self.previous_text[-200:]
                    if self.previous_text
                    else "Clear, well-punctuated dictation."
                )
                segments, info = self.model.transcribe(
                    audio_float,
                    language=self.language or "en",
                    initial_prompt=refinement_prompt,
                    temperature=0.0,
                    beam_size=3,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=200,
                    ),
                )
            else:
                # Batch mode: full accuracy settings
                segments, info = self.model.transcribe(
                    audio_float,
                    language=self.language or "en",
                    initial_prompt=batch_prompt,
                    temperature=0.0,
                    beam_size=5,
                    condition_on_previous_text=True,
                    without_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=1000,
                        speech_pad_ms=400,
                    ),
                )

            text = ""
            for segment in segments:
                text += segment.text.strip() + " "
            text = text.strip()

            if text:
                transcribe_time = time.time() - start_time

                if is_refinement:
                    # Compare refinement result with streaming output
                    streaming_normalized = " ".join(streaming_text.lower().split())
                    refined_normalized = " ".join(text.lower().split())

                    if streaming_normalized == refined_normalized:
                        print(f"✅ [{transcribe_time:.2f}s] Confirmed: '{text}'")
                    else:
                        # Sanity check: reject refinement if output is much shorter
                        stream_word_count = len(streaming_normalized.split())
                        refined_word_count = len(refined_normalized.split())
                        too_short = (
                            stream_word_count > 3
                            and refined_word_count < stream_word_count * 0.4
                        )

                        # Check if user has moved on (new streaming text typed)
                        with self.streaming_lock:
                            new_streaming = self.current_streaming_text

                        if too_short:
                            print(
                                f"⚠️  [{transcribe_time:.2f}s] Rejected refinement (too short {refined_word_count}/{stream_word_count} words): '{streaming_text}' -> '{text}'"
                            )
                            self._log(
                                f"refinement_rejected streaming='{streaming_text}' refined='{text}' reason=too_short"
                            )
                        elif new_streaming:
                            print(
                                f"⏭️  [{transcribe_time:.2f}s] Skipped correction (user moved on): '{streaming_text}' -> '{text}'"
                            )
                            self._log(
                                f"refinement_skipped streaming='{streaming_text}' refined='{text}' reason=user_moved_on"
                            )
                        else:
                            # Use LCP diff to minimize backspacing
                            s_lower = streaming_text.lower()
                            t_lower = text.lower()
                            common_len = 0
                            for ci in range(min(len(s_lower), len(t_lower))):
                                if s_lower[ci] == t_lower[ci]:
                                    common_len = ci + 1
                                else:
                                    break
                            chars_to_delete = len(streaming_text) - common_len
                            new_suffix = text[common_len:]

                            max_replace = 100
                            # Only reject if refined text has almost no word overlap
                            # (protects against hallucination, but allows start-trimming)
                            stream_words = set(streaming_text.lower().split())
                            refined_words = set(text.lower().split())
                            word_overlap = len(stream_words & refined_words)
                            no_overlap = (
                                common_len == 0
                                and word_overlap <= 1
                                and len(text) < len(streaming_text)
                            )
                            if no_overlap:
                                print(
                                    f"⚠️  [{transcribe_time:.2f}s] Rejected refinement (no overlap): '{streaming_text}' -> '{text}'"
                                )
                                self._log(
                                    f"refinement_rejected streaming='{streaming_text}' refined='{text}' reason=no_overlap"
                                )
                            elif chars_to_delete > max_replace:
                                print(
                                    f"⚠️  [{transcribe_time:.2f}s] Skipped refinement (replace={chars_to_delete}>{max_replace}): '{streaming_text}' -> '{text}'"
                                )
                                self._log(
                                    f"refinement_skipped streaming='{streaming_text}' refined='{text}' reason=replace_too_large chars={chars_to_delete}"
                                )
                            else:
                                print(
                                    f"🔄 [{transcribe_time:.2f}s] Refined (bs={chars_to_delete}): '{streaming_text}' -> '{text}'"
                                )
                                self._log(
                                    f"refinement_correction streaming='{streaming_text}' refined='{text}' bs={chars_to_delete}"
                                )
                                if chars_to_delete > 0 or new_suffix:
                                    if self.ibus_client.is_available:
                                        self.ibus_client.send_replace(
                                            chars_to_delete, new_suffix
                                        )
                                    elif self.key_injector:
                                        self.key_injector.replace_text(
                                            chars_to_delete, new_suffix
                                        )
                                    else:
                                        if chars_to_delete > 0:
                                            self._send_backspaces(chars_to_delete)
                                        if new_suffix:
                                            self._type_raw(new_suffix)
                else:
                    print(f"✅ [{transcribe_time:.2f}s] '{text}'")

                try:
                    qsize = self.transcription_queue.qsize()
                except Exception:
                    qsize = 0
                self._log(
                    f"transcribed latency={transcribe_time:.3f}s chars={len(text)} queue={qsize} dropped={self.dropped_transcriptions}"
                )

                self.previous_text = (self.previous_text + " " + text)[-500:]

                # Flag audio stream for reset after long transcriptions.
                # PyAudio's internal ring buffer can overflow during long Whisper runs,
                # leaving the stream unable to deliver new audio callbacks.
                # The main loop handles the actual restart (thread-safe).
                if transcribe_time > 0.5 and not self.streaming_enabled:
                    self._stream_reset_needed = True

                # In refinement mode, text is already typed by streaming - skip unless corrected
                if is_refinement:
                    # Process punctuation on the final text for history
                    if self.commands_enabled and COMMANDS_AVAILABLE:
                        from commands import process_punctuation

                        text = process_punctuation(text)
                    # Update typing history with the final result
                    self.typing_history.append((text, len(text)))
                    if len(self.typing_history) > self.max_history:
                        self.typing_history.pop(0)
                    return

                # Process punctuation commands if commands enabled
                if self.commands_enabled and COMMANDS_AVAILABLE:
                    from commands import process_punctuation

                    text = process_punctuation(text)

                # Check for voice commands if enabled
                if self.command_detector and self.command_executor:
                    if self._handle_pending_command(text):
                        return
                    if self._handle_command_mode(text):
                        return

                    intent, confidence, params = self.command_detector.detect(text)

                    if (
                        intent != "dictation"
                        and confidence >= self.command_min_confidence
                    ):
                        action = params.get("action", intent)
                        print(f"🎯 Command: {action} ({confidence:.0%})")

                        # Handle force_dictation (e.g., "type hello world")
                        if intent == "force_dictation":
                            forced_text = params.get("text", "")
                            if forced_text:
                                self.type_text(forced_text)
                        else:
                            if (
                                self.require_command_arm
                                and time.time() > self.commands_armed_until
                            ):
                                print("⚠️  Command ignored (say 'command mode' to arm)")
                                self.type_text(text)
                                return

                            if confidence < self.command_confirm_below:
                                self._set_pending_command(
                                    intent, params, confidence, text
                                )
                                return

                            executed = self.command_executor.execute(intent, params)
                            if executed is False:
                                print(f"⚠️  Command not executed, typing as text")
                                self.type_text(text)
                    else:
                        self.type_text(text)
                else:
                    self.type_text(text)
            else:
                print("❌ No speech detected")

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            self._log(f"transcription_error {e}", level="error")
            import traceback

            traceback.print_exc()

    def streaming_worker(self):
        """Background thread for streaming STT (Pass 1: real-time partial results).

        Drains all available chunks in a batch, feeds them to sherpa-onnx,
        then updates the display once. Throttles typing to ~150ms intervals
        to prevent rapid-fire corrections.
        When an endpoint is detected, queues the full audio for refinement.
        """
        # Accumulate chunks for refinement pass
        streaming_audio_buffer = []
        last_type_time = 0.0
        type_interval = 0.15  # Minimum seconds between typing updates
        pending_partial = ""  # Latest partial waiting to be typed
        needs_leading_space = False  # Add space before next utterance
        utterance_prefix = (
            ""  # Space prefix for current utterance (persists across partials)
        )

        while self.running:
            # Drain all available chunks in a batch
            chunks = []
            try:
                chunk = self.streaming_queue.get(timeout=0.1)
                chunks.append(chunk)
            except queue.Empty:
                # No new audio - but check if we have a pending partial to type
                if pending_partial:
                    self._type_streaming_partial(pending_partial)
                    pending_partial = ""
                    last_type_time = time.time()
                continue

            # Drain any additional queued chunks (non-blocking)
            while True:
                try:
                    chunks.append(self.streaming_queue.get_nowait())
                except queue.Empty:
                    break

            if not self.streaming_enabled or self.streaming_stt is None:
                continue

            # Feed all chunks to sherpa-onnx
            endpoint_hit = False
            endpoint_idx = -1
            for idx, chunk in enumerate(chunks):
                streaming_audio_buffer.append(chunk.copy())
                partial = self.streaming_stt.feed_chunk(chunk)
                if partial:
                    # Set utterance prefix once at start of new utterance
                    if needs_leading_space:
                        utterance_prefix = " "
                        needs_leading_space = False
                    # Always apply prefix so LCP diff stays consistent
                    pending_partial = utterance_prefix + partial

                # Check for endpoint after each chunk
                is_endpoint, final_text = self.streaming_stt.check_endpoint()
                if is_endpoint:
                    endpoint_hit = True
                    endpoint_idx = idx
                    if final_text:
                        pending_partial = utterance_prefix + final_text
                    break  # Stop processing more chunks after endpoint

            # Type the pending partial if enough time has passed
            now = time.time()
            if pending_partial and (now - last_type_time >= type_interval):
                self._type_streaming_partial(pending_partial)
                pending_partial = ""
                last_type_time = now

            # Handle endpoint (end of utterance)
            if endpoint_hit and streaming_audio_buffer:
                # Flush any remaining pending partial
                if pending_partial:
                    self._type_streaming_partial(pending_partial)
                    pending_partial = ""
                    last_type_time = time.time()

                with self.streaming_lock:
                    streamed = self.current_streaming_text
                    self.current_streaming_text = ""

                if streamed:
                    # IBus: commit preedit as final text (clears underline)
                    if self.ibus_client.is_available:
                        self.ibus_client.send_commit(streamed.lstrip())

                    # Strip leading space before sending to refinement
                    # (turbo won't produce a leading space, so comparison must be fair)
                    refinement_text = streamed.lstrip()
                    if self.refinement_enabled and refinement_text:
                        audio_copy = streaming_audio_buffer.copy()
                        self._enqueue_transcription(
                            audio_copy, streaming_text=refinement_text
                        )
                    self._log(
                        f"streaming_endpoint text='{streamed}' chunks={len(streaming_audio_buffer)}"
                    )
                    # Track for scratch-that
                    self.typing_history.append((streamed, len(streamed)))
                    if len(self.typing_history) > self.max_history:
                        self.typing_history.pop(0)
                    # Only update previous_text from streaming if no refinement
                    # (refinement path updates it with higher-quality text)
                    if not self.refinement_enabled:
                        self.previous_text = (self.previous_text + " " + streamed)[
                            -500:
                        ]
                    needs_leading_space = True  # Next utterance gets a space prefix

                # Reset prefix for next utterance
                utterance_prefix = ""
                streaming_audio_buffer = []

                # Carry over unprocessed chunks from after the endpoint
                if endpoint_idx >= 0 and endpoint_idx + 1 < len(chunks):
                    remaining = chunks[endpoint_idx + 1 :]
                    for leftover in remaining:
                        streaming_audio_buffer.append(leftover.copy())
                        self.streaming_stt.feed_chunk(leftover)

            # Prevent unbounded buffer growth
            max_streaming_chunks = int(
                self.sample_rate / self.chunk_size * self.max_recording_seconds
            )
            if len(streaming_audio_buffer) >= max_streaming_chunks:
                if pending_partial:
                    self._type_streaming_partial(pending_partial)
                    pending_partial = ""
                    last_type_time = time.time()
                with self.streaming_lock:
                    streamed = self.current_streaming_text
                    self.current_streaming_text = ""
                if streamed:
                    audio_copy = streaming_audio_buffer.copy()
                    self._enqueue_transcription(audio_copy, streaming_text=streamed)
                streaming_audio_buffer = []
                self.streaming_stt.reset()

    def _type_streaming_partial(self, new_partial: str):
        """Incrementally type streaming partial results.

        With IBus: updates preedit text atomically (no backspace needed).
        Without IBus: uses LCP diff with backspace correction via evdev.
        """
        with self.streaming_lock:
            new_partial = new_partial.lower()
            old_text = self.current_streaming_text

            if not new_partial or new_partial == old_text:
                return

            # IBus path: atomic preedit update (no LCP diff needed)
            if self.ibus_client.is_available and self.ibus_client.send_preedit(
                new_partial
            ):
                self.current_streaming_text = new_partial
                return

            # Evdev fallback: LCP diff with backspace correction
            common_len = 0
            for i in range(min(len(old_text), len(new_partial))):
                if old_text[i] == new_partial[i]:
                    common_len = i + 1
                else:
                    break

            chars_to_delete = len(old_text) - common_len
            new_chars = new_partial[common_len:]

            if chars_to_delete > 0:
                self._send_backspaces(chars_to_delete)

            if new_chars:
                self._type_raw(new_chars)

            self.current_streaming_text = new_partial

    def _send_backspaces(self, count: int):
        """Send backspace key presses."""
        if count <= 0:
            return
        try:
            if self.ibus_client.is_available and self.ibus_client.send_delete(count):
                return
            if self.key_injector:
                self.key_injector.send_backspaces(count)
            elif self.display_server == "wayland":
                env = self._ydotool_env()
                key_args = []
                for _ in range(count):
                    key_args.extend(["14:1", "14:0"])
                subprocess.run(["ydotool", "key"] + key_args, check=True, env=env)
            else:
                subprocess.run(
                    ["xdotool", "key", "--repeat", str(count), "BackSpace"],
                    check=True,
                )
        except Exception as e:
            print(f"Backspace error: {e}")

    def _type_raw(self, text: str):
        """Type text without adding to typing history (for streaming partials)."""
        if not text:
            return
        try:
            if self.ibus_client.is_available and self.ibus_client.send_commit(text):
                return
            if self.key_injector:
                self.key_injector.type_text(text)
            elif self.display_server == "wayland":
                subprocess.run(
                    ["ydotool", "type", "-d", "0", "-H", "0", "--", text],
                    check=True,
                    env=self._ydotool_env(),
                )
            else:
                subprocess.run(["xdotool", "type", "--delay", "0", text], check=True)
        except Exception as e:
            print(f"Type error: {e}")

    def _replace_typed_text(self, old_text: str, new_text: str):
        """Replace previously typed text with new text (for refinement corrections)."""
        if not old_text:
            self._type_raw(new_text)
            return

        if self.key_injector:
            # Single write() syscall: backspace + retype in one kernel call
            self.key_injector.replace_text(len(old_text), new_text)
        else:
            self._send_backspaces(len(old_text))
            if new_text:
                self._type_raw(new_text)

    def _ydotool_env(self):
        """Get environment with ydotool socket path set"""
        env = os.environ.copy()
        env["YDOTOOL_SOCKET"] = "/run/ydotoold/socket"
        return env

    def _handle_command_mode(self, text: str) -> bool:
        """Handle command arming phrases. Returns True if consumed."""
        if not self.require_command_arm:
            return False

        text_clean = text.lower().strip().strip(".")
        if text_clean in ("command mode", "commands mode", "arm commands"):
            self.commands_armed_until = time.time() + self.command_arm_seconds
            print(f"🛡️  Command mode enabled for {self.command_arm_seconds}s")
            self._notify("Voice commands", f"Enabled for {self.command_arm_seconds}s")
            return True
        if text_clean in ("dictation mode", "cancel command mode", "disarm commands"):
            self.commands_armed_until = 0.0
            print("🛡️  Command mode disabled")
            self._notify("Voice commands", "Disabled")
            return True
        return False

    def _handle_pending_command(self, text: str) -> bool:
        """Handle confirmations for pending commands."""
        if not self.pending_command:
            return False

        now = time.time()
        if now - self.pending_command["created"] > self.command_confirm_seconds:
            self.pending_command = None
            return False

        if self.require_command_arm and now > self.commands_armed_until:
            self.pending_command = None
            print("⚠️  Pending command expired (command mode off)")
            return True

        text_clean = text.lower().strip().strip(".")
        if text_clean in ("confirm", "yes", "do it"):
            pending = self.pending_command
            self.pending_command = None

            intent = pending["intent"]
            params = pending["params"]
            if intent == "force_dictation":
                forced_text = params.get("text", "")
                if forced_text:
                    self.type_text(forced_text)
                return True

            executed = self.command_executor.execute(intent, params)
            if not executed:
                self.type_text(pending["text"])
            return True

        if text_clean in ("cancel", "no", "never mind"):
            self.pending_command = None
            print("❎ Command canceled")
            return True

        # Any other utterance cancels the pending command
        self.pending_command = None
        return False

    def _set_pending_command(
        self, intent: str, params: dict, confidence: float, text: str
    ):
        """Set a pending command for confirmation."""
        self.pending_command = {
            "intent": intent,
            "params": params,
            "confidence": confidence,
            "text": text,
            "created": time.time(),
        }
        action = params.get("action", intent)
        print(f"🤔 Confirm command '{action}'? Say 'confirm' or 'cancel'.")

    def type_text(self, text):
        """Type text using IBus commit (preferred), evdev, ydotool, or xdotool."""
        try:
            if self.ibus_client.is_available and self.ibus_client.send_commit(text):
                pass
            elif self.key_injector:
                self.key_injector.type_text(text)
            elif self.display_server == "wayland":
                subprocess.run(
                    ["ydotool", "type", "-d", "0", "-H", "0", "--", text],
                    check=True,
                    env=self._ydotool_env(),
                )
            else:
                subprocess.run(["xdotool", "type", "--delay", "0", text], check=True)
            print(f"⌨️  Typed: '{text}'")

            # Track for scratch that
            self.typing_history.append((text, len(text)))
            if len(self.typing_history) > self.max_history:
                self.typing_history.pop(0)

        except Exception as e:
            print(f"❌ Failed to type: {e}")

    def _scratch_that(self):
        """Delete last typed text by sending backspaces."""
        if self.typing_history:
            last_text, char_count = self.typing_history.pop()
            try:
                self._send_backspaces(char_count)
                print(f"🔙 Scratched: '{last_text}'")
                return True
            except Exception as e:
                print(f"❌ Failed to scratch: {e}")
                return False
        else:
            print("⚠️  Nothing to scratch")
            return False

    def run(self):
        """Start voice typing"""
        try:
            print("\n" + "=" * 50)
            print("🎙️  VOICE TYPING ACTIVE")
            print("=" * 50)

            # Display hotkey instructions based on display server
            if self.display_server == "wayland":
                print(f"⚠️  WAYLAND: Hotkey requires compositor binding")
                print(f"   Bind {self.hotkey.upper()} to: ./voice-toggle")
                print(f"   Or: echo toggle | nc -U {SOCKET_PATH}")
            else:
                print(f"Hotkey: {self.hotkey.upper()} to pause/resume")

            print("Press Ctrl+C to stop")
            print("=" * 50 + "\n")

            self.running = True

            # Start hotkey listener
            self._start_hotkey_listener()

            # Start transcription thread
            self.transcription_thread = threading.Thread(
                target=self.transcription_worker,
                daemon=True,
                name="TranscriptionWorker",
            )
            self.transcription_thread.start()

            # Start streaming thread if enabled
            if self.streaming_enabled:
                self.streaming_thread = threading.Thread(
                    target=self.streaming_worker,
                    daemon=True,
                    name="StreamingWorker",
                )
                self.streaming_thread.start()
                mode = (
                    "streaming + refinement"
                    if self.refinement_enabled
                    else "streaming-only"
                )
                print(f"🔊 Streaming STT active ({mode})")

            # Start audio stream
            self.stream.start_stream()

            # Keep main thread alive
            while self.running:
                try:
                    stream_active = self.stream and self.stream.is_active()
                except Exception:
                    stream_active = False
                # Reset stream if flagged by worker or if stream died
                needs_reset = self._stream_reset_needed or not stream_active
                if needs_reset:
                    now = time.time()
                    if now - self.last_stream_restart > 1.0:
                        self.last_stream_restart = now
                        reason = "flagged" if self._stream_reset_needed else "stopped"
                        self._stream_reset_needed = False
                        print(f"🔄 Audio stream reset ({reason})")
                        try:
                            self._restart_audio_stream()
                            self.last_callback_time = time.time()
                        except Exception as e:
                            print(f"⚠️  Audio stream restart failed: {e}")
                if (
                    self.status_interval
                    and time.time() - self.last_status_log >= self.status_interval
                ):
                    self.last_status_log = time.time()
                    status_line = self._status_snapshot()
                    print(f"ℹ️  {status_line}")
                    self._log(f"status {status_line}")
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
        except Exception as e:
            print(f"❌ Runtime error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        self.running = False

        # Stop hotkey listener
        if self.hotkey_listener:
            try:
                self.hotkey_listener.stop()
            except:
                pass

        # Stop audio visualizer
        if self.visualizer:
            try:
                self.visualizer.stop()
            except:
                pass

        # Close socket
        if self.socket_server:
            try:
                self.socket_server.close()
            except:
                pass
        if os.path.exists(SOCKET_PATH):
            try:
                os.remove(SOCKET_PATH)
            except:
                pass
        if os.path.exists(TOKEN_PATH):
            try:
                os.remove(TOKEN_PATH)
            except:
                pass

        # Close IBus client
        if self.ibus_client:
            try:
                self.ibus_client.close()
            except:
                pass

        # Release uinput device
        if self.key_injector:
            try:
                self.key_injector.close()
            except:
                pass

        # Wait for threads
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=2.0)
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=2.0)

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass

        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass

        print("✅ Cleanup complete")

    def _restart_audio_stream(self):
        """Attempt to restart the audio stream."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass

        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback,
        )
        self.stream.start_stream()


def _build_defaults(config: dict) -> dict:
    merged = dict(CONFIG_DEFAULTS)
    normalized = dict(config)
    if "no_adaptive_vad" in normalized and "adaptive_vad" not in normalized:
        normalized["adaptive_vad"] = not bool(normalized["no_adaptive_vad"])
    merged.update(normalized)
    return merged


def _build_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhanced Voice Typing")
    parser.add_argument(
        "--config", default=defaults["config"], help="Config file (yaml/json)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        default=False,
        help="List input devices and exit",
    )
    parser.add_argument(
        "--model",
        default=defaults["model"],
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "distil-large-v3",
            "distil-medium",
            "large-v3",
            "large-v3-turbo",
        ],
        help="Whisper model size",
    )
    parser.add_argument(
        "--device",
        default=defaults["device"],
        choices=["auto", "cpu", "cuda"],
        help="Device to run on",
    )
    parser.add_argument(
        "--language",
        default=defaults["language"],
        help="Language code (e.g., en, es, fr)",
    )
    parser.add_argument(
        "--hotkey",
        default=defaults["hotkey"],
        choices=["f12", "f11", "f10", "scroll_lock", "pause"],
        help="Hotkey for pause/resume (default: f12)",
    )
    parser.add_argument(
        "--commands",
        action=argparse.BooleanOptionalAction,
        default=defaults["commands"],
        help="Enable voice command detection (window/edit/custom)",
    )
    parser.add_argument(
        "--commands-file",
        default=defaults["commands_file"],
        help="Custom commands YAML file",
    )
    parser.add_argument(
        "--command-arm",
        action=argparse.BooleanOptionalAction,
        default=defaults["command_arm"],
        help='Require "command mode" to execute commands',
    )
    parser.add_argument(
        "--command-arm-seconds",
        type=int,
        default=defaults["command_arm_seconds"],
        help="How long command mode stays armed (seconds)",
    )
    parser.add_argument(
        "--command-min-confidence",
        type=float,
        default=defaults["command_min_confidence"],
        help="Minimum confidence to treat as a command",
    )
    parser.add_argument(
        "--command-confirm-below",
        type=float,
        default=defaults["command_confirm_below"],
        help="Ask for confirmation below this confidence",
    )
    parser.add_argument(
        "--command-confirm-seconds",
        type=float,
        default=defaults["command_confirm_seconds"],
        help="Time window to confirm a command",
    )
    parser.add_argument(
        "--allow-shell",
        action=argparse.BooleanOptionalAction,
        default=defaults["allow_shell"],
        help="Allow custom shell commands from commands.yaml",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=defaults["max_seconds"],
        help="Max seconds per recording before forced flush",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=defaults["queue_size"],
        help="Max queued recordings before dropping",
    )
    parser.add_argument(
        "--calibrate-seconds",
        type=float,
        default=defaults["calibrate_seconds"],
        help="Seconds to sample ambient noise on startup",
    )
    parser.add_argument(
        "--noise-gate",
        action=argparse.BooleanOptionalAction,
        default=defaults["noise_gate"],
        help="Enable noise gate based on ambient noise",
    )
    parser.add_argument(
        "--noise-gate-multiplier",
        type=float,
        default=defaults["noise_gate_multiplier"],
        help="Noise gate threshold multiplier over noise floor",
    )
    parser.add_argument(
        "--agc",
        action=argparse.BooleanOptionalAction,
        default=defaults["agc"],
        help="Enable automatic gain control",
    )
    parser.add_argument(
        "--agc-target-rms",
        type=float,
        default=defaults["agc_target_rms"],
        help="AGC target RMS amplitude",
    )
    parser.add_argument(
        "--agc-min-gain",
        type=float,
        default=defaults["agc_min_gain"],
        help="AGC minimum gain",
    )
    parser.add_argument(
        "--agc-max-gain",
        type=float,
        default=defaults["agc_max_gain"],
        help="AGC maximum gain",
    )
    parser.add_argument(
        "--adaptive-vad",
        action=argparse.BooleanOptionalAction,
        default=defaults["adaptive_vad"],
        help="Enable adaptive VAD aggressiveness",
    )
    parser.add_argument(
        "--notify",
        action=argparse.BooleanOptionalAction,
        default=defaults["notify"],
        help="Enable desktop notifications (notify-send)",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=defaults["status_interval"],
        help="Seconds between status lines (0 to disable)",
    )
    parser.add_argument(
        "--input-device",
        default=defaults["input_device"],
        help="Input device index or name substring",
    )
    parser.add_argument(
        "--ptt",
        action=argparse.BooleanOptionalAction,
        default=defaults["ptt"],
        help="Enable push-to-talk",
    )
    parser.add_argument(
        "--ptt-hotkey",
        default=defaults["ptt_hotkey"],
        help="Push-to-talk hotkey (default: f9)",
    )
    parser.add_argument(
        "--ptt-mode",
        default=defaults["ptt_mode"],
        choices=["hold", "toggle"],
        help="Push-to-talk mode",
    )
    parser.add_argument(
        "--log-file",
        default=defaults["log_file"],
        help="Log file path (empty to disable)",
    )
    parser.add_argument(
        "--log-max-bytes",
        type=int,
        default=defaults["log_max_bytes"],
        help="Max log size before rotation",
    )
    parser.add_argument(
        "--log-backups",
        type=int,
        default=defaults["log_backups"],
        help="Number of rotated logs to keep",
    )
    parser.add_argument(
        "--log-level",
        default=defaults["log_level"],
        help="Log level (INFO, DEBUG, ERROR)",
    )
    parser.add_argument(
        "--viz",
        action=argparse.BooleanOptionalAction,
        default=defaults["viz"],
        help="Enable audio visualization popup",
    )
    parser.add_argument(
        "--viz-position",
        default=defaults["viz_position"],
        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
        help="Visualization popup position",
    )
    parser.add_argument(
        "--viz-hide-delay",
        type=int,
        default=defaults["viz_hide_delay"],
        help="Milliseconds to wait before hiding after silence",
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=defaults["streaming"],
        help="Enable streaming STT (sherpa-onnx, words appear as you speak)",
    )
    parser.add_argument(
        "--streaming-model",
        default=defaults["streaming_model"],
        choices=["zipformer-en", "zipformer-en-20M"],
        help="Streaming model (zipformer-en=80MB accurate, zipformer-en-20M=20MB fast)",
    )
    parser.add_argument(
        "--refinement",
        action=argparse.BooleanOptionalAction,
        default=defaults["refinement"],
        help="Enable Whisper refinement after streaming (adds punctuation/casing, needs GPU)",
    )
    parser.add_argument(
        "--refinement-model",
        default=defaults["refinement_model"],
        help="Whisper model for refinement pass (default: large-v3-turbo)",
    )
    return parser


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="Config file (yaml/json)"
    )
    pre_args, _ = pre_parser.parse_known_args()

    config = _load_config(pre_args.config)
    config = _apply_env_overrides(config)
    defaults = _build_defaults(config)

    parser = _build_parser(defaults)
    args = parser.parse_args()

    if args.list_devices:
        VoiceTyping.list_input_devices()
        return

    logger = _setup_logging(
        args.log_file, args.log_level, args.log_max_bytes, args.log_backups
    )
    if logger:
        logger.info("starting voice typing")
        logger.info(
            "config model=%s device=%s commands=%s input_device=%s ptt=%s streaming=%s",
            args.model,
            args.device,
            args.commands,
            args.input_device,
            args.ptt,
            args.streaming,
        )

    # Device selection
    if args.device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                args.device = "cuda"
                print("CUDA available, using GPU")
            else:
                args.device = "cpu"
                print("CUDA not available, using CPU")
        except ImportError:
            print("PyTorch not installed, using CPU")
            args.device = "cpu"

    # Create default commands config if commands enabled
    if args.commands and COMMANDS_AVAILABLE:
        create_default_config(args.commands_file)

    vt = VoiceTyping(
        model_size=args.model,
        device=args.device,
        language=args.language,
        hotkey=args.hotkey,
        commands_enabled=args.commands,
        commands_file=args.commands_file,
        require_command_arm=args.command_arm,
        command_arm_seconds=args.command_arm_seconds,
        allow_shell_commands=args.allow_shell,
        max_recording_seconds=args.max_seconds,
        queue_size=args.queue_size,
        calibration_seconds=args.calibrate_seconds,
        noise_gate_enabled=args.noise_gate,
        noise_gate_multiplier=args.noise_gate_multiplier,
        agc_enabled=args.agc,
        agc_target_rms=args.agc_target_rms,
        agc_min_gain=args.agc_min_gain,
        agc_max_gain=args.agc_max_gain,
        adaptive_vad=args.adaptive_vad,
        command_min_confidence=args.command_min_confidence,
        command_confirm_below=args.command_confirm_below,
        command_confirm_seconds=args.command_confirm_seconds,
        input_device=args.input_device,
        notify=args.notify,
        status_interval=args.status_interval,
        ptt_enabled=args.ptt,
        ptt_hotkey=args.ptt_hotkey,
        ptt_mode=args.ptt_mode,
        logger=logger,
        viz_enabled=args.viz,
        viz_position=args.viz_position,
        viz_hide_delay=args.viz_hide_delay,
        streaming_enabled=args.streaming,
        streaming_model=args.streaming_model,
        refinement_enabled=args.refinement,
        refinement_model=args.refinement_model,
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        vt.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    vt.run()


if __name__ == "__main__":
    main()
