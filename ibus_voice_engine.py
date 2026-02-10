#!/usr/bin/env python3
"""IBus input method engine for voice typing.

Receives text commands over a Unix socket from the voice typing STT pipeline
and uses IBus commit_text / update_preedit_text to atomically insert text
into any focused application.

This eliminates the need for uinput/ydotool key injection and works uniformly
across terminals (Ghostty, kitty) and GUI apps (Firefox, editors).

Architecture:
  enhanced-voice-typing.py --[socket]--> ibus_voice_engine.py --[IBus API]--> focused app

Socket protocol (newline-terminated commands):
  preedit:TEXT     - Show TEXT as underlined preedit (streaming partials)
  commit:TEXT      - Clear preedit and atomically commit TEXT
  delete:N         - Delete N characters before cursor
  replace:N:TEXT   - Delete N chars before cursor, then commit TEXT
"""

import os
import sys
import socket
import threading
import atexit

import gi

gi.require_version("IBus", "1.0")
from gi.repository import IBus, GLib

RUNTIME_DIR = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
IBUS_SOCKET_PATH = os.path.join(RUNTIME_DIR, f"voice-typing-ibus-{os.getuid()}.sock")
IBUS_CAPS_PATH = os.path.join(RUNTIME_DIR, f"voice-typing-ibus-caps-{os.getuid()}")

# Global reference to the active engine instance (set by factory)
_active_engine = None


class VoiceTypingEngine(IBus.Engine):
    """IBus engine that receives voice typing text over a socket."""

    __gtype_name__ = "VoiceTypingEngine"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._enabled = False
        self._surrounding = False

    def do_process_key_event(self, keyval, keycode, state):
        """Pass through all keyboard events — we only inject text, not intercept keys."""
        return False

    def do_enable(self):
        self._enabled = True
        print("IBus VoiceTyping engine enabled")

    def do_disable(self):
        self._enabled = False
        self._hide_preedit()
        print("IBus VoiceTyping engine disabled")

    def do_set_capabilities(self, caps):
        """Track client capabilities — especially surrounding text support."""
        self._surrounding = bool(caps & IBus.Capabilite.SURROUNDING_TEXT)
        self._write_caps()
        print(f"Client caps: surrounding_text={self._surrounding} (0x{caps:x})")

    def do_focus_in(self):
        self._write_caps()

    def do_focus_out(self):
        self._hide_preedit()

    def _write_caps(self):
        """Write current capabilities to file for voice typing script."""
        try:
            with open(IBUS_CAPS_PATH, "w") as f:
                f.write("surrounding\n" if self._surrounding else "basic\n")
        except OSError:
            pass

    def do_reset(self):
        self._hide_preedit()

    def commit(self, text):
        """Atomically commit text to the focused application."""
        ibus_text = IBus.Text.new_from_string(text)
        self.commit_text(ibus_text)

    def preedit(self, text):
        """Show preedit preview text. Underlined in terminals, plain in browsers."""
        if not text:
            self._hide_preedit()
            return
        ibus_text = IBus.Text.new_from_string(text)
        if not self._surrounding:
            # Terminal: underline as visual indicator that refinement is pending
            ibus_text.append_attribute(
                IBus.AttrType.UNDERLINE, IBus.AttrUnderline.SINGLE, 0, len(text)
            )
        self.update_preedit_text_with_mode(
            ibus_text, len(text), True, IBus.PreeditFocusMode.CLEAR
        )

    def delete_chars(self, count):
        """Delete N characters before cursor."""
        if count > 0:
            self.delete_surrounding_text(-count, count)

    def replace_chars(self, count, text):
        """Delete N chars then commit new text."""
        if count > 0:
            self.delete_surrounding_text(-count, count)
        if text:
            self.commit(text)

    def _hide_preedit(self):
        self.hide_preedit_text()


class VoiceTypingEngineFactory(IBus.Factory):
    """Factory that creates VoiceTypingEngine instances on demand from IBus."""

    __gtype_name__ = "VoiceTypingEngineFactory"

    _engine_count = 0

    def __init__(self, bus):
        self._bus = bus
        super().__init__(object_path=IBus.PATH_FACTORY, connection=bus.get_connection())

    def do_create_engine(self, engine_name):
        global _active_engine
        VoiceTypingEngineFactory._engine_count += 1
        obj_path = (
            f"/org/freedesktop/IBus/Engine/{VoiceTypingEngineFactory._engine_count}"
        )
        engine = VoiceTypingEngine(
            engine_name=engine_name,
            object_path=obj_path,
            connection=self._bus.get_connection(),
        )
        _active_engine = engine
        print(f"Created engine: {engine_name} at {obj_path}")
        return engine


def _handle_socket_command(line):
    """Parse and dispatch a socket command to the IBus engine via GLib main thread."""
    global _active_engine

    if _active_engine is None:
        return

    if ":" not in line:
        return

    cmd, _, payload = line.partition(":")
    cmd = cmd.strip()

    if cmd == "preedit":
        GLib.idle_add(_active_engine.preedit, payload)
    elif cmd == "commit":
        GLib.idle_add(_active_engine.preedit, "")
        GLib.idle_add(_active_engine.commit, payload)
    elif cmd == "delete":
        try:
            count = int(payload.strip())
            GLib.idle_add(_active_engine.delete_chars, count)
        except ValueError:
            pass
    elif cmd == "replace":
        # Format: replace:N:new text
        parts = payload.split(":", 1)
        if len(parts) == 2:
            try:
                count = int(parts[0])
                text = parts[1]
                GLib.idle_add(_active_engine.replace_chars, count, text)
            except ValueError:
                pass


def _handle_client(conn):
    """Handle a persistent client connection, reading newline-terminated commands."""
    buf = b""
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str:
                    _handle_socket_command(line_str)
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _socket_listener():
    """Accept connections from voice typing STT pipeline."""
    if os.path.exists(IBUS_SOCKET_PATH):
        os.remove(IBUS_SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old_umask = os.umask(0o077)
    try:
        server.bind(IBUS_SOCKET_PATH)
    finally:
        os.umask(old_umask)
    os.chmod(IBUS_SOCKET_PATH, 0o600)
    server.listen(2)

    print(f"IBus socket listening: {IBUS_SOCKET_PATH}")

    while True:
        try:
            conn, _ = server.accept()
            client_thread = threading.Thread(
                target=_handle_client, args=(conn,), daemon=True
            )
            client_thread.start()
        except Exception as e:
            print(f"Socket accept error: {e}")
            continue


def _cleanup():
    for path in (IBUS_SOCKET_PATH, IBUS_CAPS_PATH):
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


def main():
    IBus.init()

    bus = IBus.Bus()
    if not bus.is_connected():
        print("Cannot connect to IBus daemon")
        sys.exit(1)

    component = IBus.Component.new(
        "org.freedesktop.IBus.VoiceTyping",
        "Voice Typing Input Method",
        "1.0",
        "MIT",
        "Voice Typing",
        "",
        "",
        "voice-typing",
    )

    engine_desc = IBus.EngineDesc.new(
        "voice-typing",
        "Voice Typing",
        "Voice-to-text input via streaming STT",
        "en",
        "MIT",
        "Voice Typing",
        "audio-input-microphone",
        "us",
    )
    component.add_engine(engine_desc)

    factory = VoiceTypingEngineFactory(bus)
    rc = bus.register_component(component)
    print(f"register_component: {rc}")
    rn = bus.request_name("org.freedesktop.IBus.VoiceTyping", 0)
    print(f"request_name: {rn}")

    # Try to verify the engine is listed
    engines = bus.list_engines()
    found = [e.get_name() for e in engines if "voice" in e.get_name().lower()]
    print(f"Engines with 'voice': {found}")

    # Start socket listener in background thread
    socket_thread = threading.Thread(
        target=_socket_listener, daemon=True, name="IBusSocketListener"
    )
    socket_thread.start()

    print("IBus Voice Typing engine running")
    print(f"Socket: {IBUS_SOCKET_PATH}")

    atexit.register(_cleanup)

    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nShutting down IBus Voice Typing engine")
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
