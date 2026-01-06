"""
Audio Visualizer - GTK4 layer-shell spectrum analyzer overlay for voice typing.

Displays FFT frequency bars when speech is detected, auto-hides after silence.
Works on Wayland via wlr-layer-shell protocol, falls back to regular window on X11.
"""

import queue
import threading
import time
import numpy as np
import os

# Check display server
DISPLAY_SERVER = os.environ.get('XDG_SESSION_TYPE', 'x11')

# Pre-load GTK4 layer-shell BEFORE GTK4 (required for proper initialization)
LAYER_SHELL_AVAILABLE = False
if DISPLAY_SERVER == 'wayland':
    try:
        import gi
        gi.require_version('Gtk4LayerShell', '1.0')
        from gi.repository import Gtk4LayerShell
        LAYER_SHELL_AVAILABLE = True
    except Exception:
        pass


class AudioVisualizer:
    """GTK4 layer-shell spectrum analyzer overlay for voice activity visualization."""

    def __init__(
        self,
        position: str = "bottom-right",
        hide_delay_ms: int = 1500,
        num_bars: int = 12,
        bar_width: int = 8,
        bar_gap: int = 3,
        window_height: int = 60,
        margin: int = 20,
        sample_rate: int = 16000,
    ):
        # Config
        self.position = position.lower()
        self.hide_delay_ms = hide_delay_ms
        self.num_bars = num_bars
        self.bar_width = bar_width
        self.bar_gap = bar_gap
        self.window_height = window_height
        self.margin = margin
        self.sample_rate = sample_rate

        # Audio data (thread-safe)
        self.audio_queue: queue.Queue = queue.Queue(maxsize=10)
        self.spectrum_data: np.ndarray = np.zeros(num_bars)
        self.data_lock = threading.Lock()

        # GTK state (set in GTK thread)
        self.window = None
        self.drawing_area = None
        self.app = None
        self.visible = False
        self.hide_timer_id = None

        # Threading
        self.gtk_thread: threading.Thread = None
        self.running = False
        self._gtk_ready = threading.Event()

    def start(self):
        """Start visualizer in background thread."""
        if self.running:
            return

        self.running = True
        self.gtk_thread = threading.Thread(
            target=self._gtk_main,
            daemon=True,
            name="AudioVisualizer"
        )
        self.gtk_thread.start()
        # Wait for GTK to initialize
        self._gtk_ready.wait(timeout=5.0)

    def stop(self):
        """Stop visualizer."""
        self.running = False
        if self.app:
            try:
                import gi
                gi.require_version('Gtk', '4.0')
                from gi.repository import GLib
                GLib.idle_add(self._quit_app)
            except Exception:
                pass

    def _quit_app(self):
        """Quit GTK app from GTK thread."""
        if self.app:
            self.app.quit()
        return False

    def push_audio(self, chunk: np.ndarray):
        """Push audio chunk for visualization (non-blocking, from audio thread)."""
        if not self.running:
            return
        try:
            self.audio_queue.put_nowait(chunk.copy())
        except queue.Full:
            pass  # Drop if queue full - visualization is best-effort

    def set_speaking(self, is_speaking: bool):
        """Update speech state (called from audio thread)."""
        if not self.running or not self._gtk_ready.is_set():
            return
        try:
            import gi
            gi.require_version('Gtk', '4.0')
            from gi.repository import GLib
            GLib.idle_add(self._update_speaking_state, is_speaking)
        except Exception:
            pass

    def _gtk_main(self):
        """GTK main loop running in dedicated thread."""
        try:
            import gi
            gi.require_version('Gtk', '4.0')
            from gi.repository import Gtk, GLib, Gio

            # Create application
            self.app = Gtk.Application(
                application_id="voice.typing.visualizer",
                flags=Gio.ApplicationFlags.FLAGS_NONE
            )
            self.app.connect("activate", self._on_activate)

            # Run GTK main loop
            self.app.run([])
        except Exception as e:
            print(f"Visualizer error: {e}")
            self.running = False
            self._gtk_ready.set()

    def _on_activate(self, app):
        """GTK application activated."""
        import gi
        gi.require_version('Gtk', '4.0')
        from gi.repository import Gtk, GLib

        self._setup_window()
        app.add_window(self.window)

        # Start processing audio queue (~30fps)
        GLib.timeout_add(33, self._process_audio_queue)

        self._gtk_ready.set()

    def _setup_window(self):
        """Create GTK4 layer-shell window."""
        import gi
        gi.require_version('Gtk', '4.0')
        from gi.repository import Gtk, Gdk

        # Create window
        self.window = Gtk.Window()
        self.window.set_title("Voice Visualizer")
        self.window.set_decorated(False)
        self.window.set_resizable(False)

        # Calculate window size
        total_width = self.num_bars * (self.bar_width + self.bar_gap) - self.bar_gap + 20
        self.window.set_default_size(total_width, self.window_height)

        # Try layer-shell for Wayland (wlroots compositors: Sway, Hyprland, etc.)
        use_layer_shell = False
        if LAYER_SHELL_AVAILABLE:
            try:
                if Gtk4LayerShell.is_supported():
                    Gtk4LayerShell.init_for_window(self.window)
                    Gtk4LayerShell.set_layer(self.window, Gtk4LayerShell.Layer.OVERLAY)
                    Gtk4LayerShell.set_namespace(self.window, "voice-typing-viz")
                    Gtk4LayerShell.set_keyboard_mode(self.window, Gtk4LayerShell.KeyboardMode.NONE)
                    self._apply_layer_shell_position(Gtk4LayerShell)
                    use_layer_shell = True
                else:
                    print("Note: Layer-shell not supported (GNOME?). Using floating window.")
            except Exception as e:
                print(f"Layer-shell init failed: {e}")

        # Fallback: floating window (GNOME, KDE, X11)
        if not use_layer_shell:
            # Request floating/utility window hints
            self.window.set_deletable(False)
            self.window.set_focus_on_click(False)

        # Drawing area for spectrum
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_content_width(total_width)
        self.drawing_area.set_content_height(self.window_height)
        self.drawing_area.set_draw_func(self._on_draw)

        self.window.set_child(self.drawing_area)

        # Apply CSS for styling
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            window {
                background-color: rgba(20, 20, 25, 0.9);
                border-radius: 10px;
            }
        """)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # Start hidden
        self.window.set_visible(False)
        self.visible = False

    def _apply_layer_shell_position(self, ls):
        """Set window anchors based on position config."""
        # Clear all anchors
        for edge in [ls.Edge.TOP, ls.Edge.BOTTOM, ls.Edge.LEFT, ls.Edge.RIGHT]:
            ls.set_anchor(self.window, edge, False)
            ls.set_margin(self.window, edge, 0)

        # Apply position
        if "top" in self.position:
            ls.set_anchor(self.window, ls.Edge.TOP, True)
            ls.set_margin(self.window, ls.Edge.TOP, self.margin)
        else:
            ls.set_anchor(self.window, ls.Edge.BOTTOM, True)
            ls.set_margin(self.window, ls.Edge.BOTTOM, self.margin)

        if "left" in self.position:
            ls.set_anchor(self.window, ls.Edge.LEFT, True)
            ls.set_margin(self.window, ls.Edge.LEFT, self.margin)
        else:
            ls.set_anchor(self.window, ls.Edge.RIGHT, True)
            ls.set_margin(self.window, ls.Edge.RIGHT, self.margin)

    def _update_speaking_state(self, is_speaking: bool):
        """GTK thread: update visibility based on speech."""
        import gi
        gi.require_version('Gtk', '4.0')
        from gi.repository import GLib

        if is_speaking:
            # Cancel any pending hide
            if self.hide_timer_id:
                GLib.source_remove(self.hide_timer_id)
                self.hide_timer_id = None
            # Show window
            if not self.visible and self.window:
                self.window.set_visible(True)
                self.visible = True
        else:
            # Schedule hide after delay
            if self.visible and not self.hide_timer_id:
                self.hide_timer_id = GLib.timeout_add(
                    self.hide_delay_ms,
                    self._hide_window
                )
        return False

    def _hide_window(self):
        """Hide window after silence delay."""
        if self.window:
            self.window.set_visible(False)
        self.visible = False
        self.hide_timer_id = None
        return False

    def _process_audio_queue(self):
        """Process pending audio data and redraw."""
        if not self.running:
            return False

        # Drain queue, process latest
        latest_chunk = None
        while True:
            try:
                latest_chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                break

        if latest_chunk is not None:
            self._compute_spectrum(latest_chunk)
            if self.visible and self.drawing_area:
                self.drawing_area.queue_draw()

        return True  # Continue timer

    def _compute_spectrum(self, audio: np.ndarray) -> np.ndarray:
        """Compute spectrum bars from audio chunk using FFT."""
        # Normalize to float [-1, 1]
        audio_float = audio.astype(np.float32) / 32768.0

        # Apply Hann window to reduce spectral leakage
        if len(audio_float) > 0:
            windowed = audio_float * np.hanning(len(audio_float))
        else:
            return self.spectrum_data

        # Compute FFT (only positive frequencies)
        fft_result = np.abs(np.fft.rfft(windowed))

        if len(fft_result) < 2:
            return self.spectrum_data

        # Map FFT bins to visualization bars (logarithmic scaling)
        # Focus on speech frequencies: 80Hz - 4000Hz
        bin_hz = self.sample_rate / len(audio)
        min_bin = max(1, int(80 / bin_hz))
        max_bin = min(len(fft_result) - 1, int(4000 / bin_hz))

        if max_bin <= min_bin:
            return self.spectrum_data

        # Logarithmic frequency bands for perceptual balance
        log_bins = np.logspace(
            np.log10(min_bin),
            np.log10(max_bin),
            self.num_bars + 1
        ).astype(int)

        bars = np.zeros(self.num_bars)
        for i in range(self.num_bars):
            start, end = log_bins[i], log_bins[i + 1]
            if start < end and end <= len(fft_result):
                bars[i] = np.mean(fft_result[start:end])

        # Convert to dB scale, normalize to [0, 1]
        bars = np.clip(20 * np.log10(bars + 1e-10), -60, 0)
        bars = (bars + 60) / 60  # Normalize: -60dB->0, 0dB->1

        # Smooth with previous frame (exponential moving average)
        with self.data_lock:
            alpha = 0.35  # Smoothing factor
            self.spectrum_data = alpha * bars + (1 - alpha) * self.spectrum_data
            return self.spectrum_data.copy()

    def _on_draw(self, area, ctx, width, height):
        """Draw spectrum bars using Cairo."""
        # Get current spectrum
        with self.data_lock:
            bars = self.spectrum_data.copy()

        # Colors (gradient from cyan to magenta based on magnitude)
        color_low = (0.2, 0.7, 0.9)   # Cyan for low
        color_high = (0.9, 0.2, 0.6)  # Magenta for high

        bar_total_width = self.bar_width + self.bar_gap
        start_x = (width - (self.num_bars * bar_total_width - self.bar_gap)) / 2

        for i, magnitude in enumerate(bars):
            # Interpolate color based on magnitude
            r = color_low[0] + (color_high[0] - color_low[0]) * magnitude
            g = color_low[1] + (color_high[1] - color_low[1]) * magnitude
            b = color_low[2] + (color_high[2] - color_low[2]) * magnitude

            ctx.set_source_rgb(r, g, b)

            bar_height = max(3, magnitude * (height - 16))
            x = start_x + i * bar_total_width
            y = height - 8 - bar_height

            # Draw rounded rectangle for bar
            self._rounded_rect(ctx, x, y, self.bar_width, bar_height, 3)
            ctx.fill()

    def _rounded_rect(self, ctx, x, y, w, h, r):
        """Draw rounded rectangle path."""
        import math
        ctx.new_path()
        ctx.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
        ctx.arc(x + w - r, y + r, r, 1.5 * math.pi, 2 * math.pi)
        ctx.arc(x + w - r, y + h - r, r, 0, 0.5 * math.pi)
        ctx.arc(x + r, y + h - r, r, 0.5 * math.pi, math.pi)
        ctx.close_path()
