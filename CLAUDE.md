# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Linux voice typing using faster-whisper with a pre-recording buffer to capture speech beginnings.
- **X11**: Uses xdotool for keyboard input
- **Wayland**: Uses ydotool for keyboard input (auto-detected)

## Development Environment

```bash
nix-shell              # Enter dev environment
./voice                # Run with defaults (small model)
./voice --model tiny   # Fastest
./voice --model base   # Good balance (default in code)
./voice --device cuda  # GPU acceleration
./voice --language es  # Specific language
./voice --hotkey f11   # Custom pause hotkey
```

## Pause/Resume Hotkey

Press **F12** (default) to pause/resume transcription.

### X11/XWayland
- pynput handles hotkey directly
- Works automatically

### Wayland (pure)
- Uses Unix socket fallback at `${XDG_RUNTIME_DIR:-/tmp}/voice-typing-$UID.sock`
- Socket token at `${XDG_RUNTIME_DIR:-/tmp}/voice-typing-$UID.token` (used by `voice-toggle`)
- Helper script: `./voice-toggle`
- Or bind in compositor: `echo "toggle $(cat ${XDG_RUNTIME_DIR:-/tmp}/voice-typing-$UID.token)" | nc -U ${XDG_RUNTIME_DIR:-/tmp}/voice-typing-$UID.sock`

### Custom Hotkey
```bash
./voice --hotkey scroll_lock  # Options: f12, f11, f10, scroll_lock, pause
```

### Command Safety
```bash
# Require "command mode" before executing commands
./voice --commands --command-arm --command-arm-seconds 10

# Ask for confirmation below a confidence threshold
./voice --commands --command-min-confidence 0.8 --command-confirm-below 0.9

# Allow custom shell commands from commands.yaml
./voice --commands --allow-shell
```

### Audio Noise Controls
```bash
# Calibrate ambient noise and enable a noise gate
./voice --calibrate-seconds 1.0 --noise-gate --noise-gate-multiplier 1.5

# Automatic gain control
./voice --agc --agc-target-rms 4000

# Disable adaptive VAD
./voice --no-adaptive-vad
```

### Push-to-Talk
```bash
# Hold to talk
./voice --ptt --ptt-hotkey f9 --ptt-mode hold

# Toggle mode
./voice --ptt --ptt-hotkey f9 --ptt-mode toggle
```

### Input Devices
```bash
./voice --list-devices
./voice --input-device 2
./voice --input-device "USB Microphone"
```

### Audio Visualization
```bash
# Enable spectrum analyzer overlay (shows when speaking)
./voice --viz

# Position in different corner
./voice --viz --viz-position top-left    # Options: top-left, top-right, bottom-left, bottom-right

# Adjust hide delay after silence (ms)
./voice --viz --viz-hide-delay 2000
```

The visualizer shows an FFT spectrum analyzer overlay that appears when speech is detected and auto-hides after silence. Uses GTK4 with layer-shell for Wayland overlay support.

### Configuration
Default config: `${XDG_CONFIG_HOME:-~/.config}/voice-typing/config.yaml`

Environment overrides (prefix `VOICE_`): `VOICE_MODEL`, `VOICE_DEVICE`, `VOICE_HOTKEY`,
`VOICE_COMMANDS`, `VOICE_NOISE_GATE`, `VOICE_PTT`, `VOICE_LOG_FILE`,
`VOICE_ADAPTIVE_VAD` (or legacy `VOICE_NO_ADAPTIVE_VAD`).

### Logs
```bash
./voice --log-file ~/.local/state/voice-typing/voice-typing.log --log-max-bytes 1000000 --log-backups 5
```

## Wayland Setup

On pure Wayland, both xdotool and pynput don't work. The app auto-detects Wayland and uses ydotool instead.

### Requirements

1. **Start ydotool daemon** (required for keyboard input):
```bash
sudo ydotoold &
# Or enable NixOS service from ydotool-service.nix
```

2. **Add user to input group** (for /dev/uinput access):
```bash
sudo usermod -aG input $USER
# Then logout/login
```

3. **Bind F12 hotkey in compositor** (pynput doesn't work on Wayland):
- GNOME: Settings → Keyboard → Custom Shortcuts
- KDE: System Settings → Shortcuts
- Sway/Hyprland: `bindsym F12 exec echo "toggle $(cat ${XDG_RUNTIME_DIR:-/tmp}/voice-typing-$UID.token)" | nc -U ${XDG_RUNTIME_DIR:-/tmp}/voice-typing-$UID.sock`

### Testing Wayland Setup
```bash
# Check display server
echo $XDG_SESSION_TYPE  # Should say "wayland"

# Check ydotool daemon
pgrep ydotoold  # Should return a PID

# Test keyboard input
ydotool type "hello"

# Test pause toggle
./voice-toggle
```

## Voice Commands Mode

Enable with `--commands` flag to auto-detect voice commands vs dictation.

```bash
./voice --commands                 # Enable command detection
./voice --commands --commands-file ~/.config/voice-typing/commands.yaml
```

### How It Works
- Text is analyzed for command patterns before typing
- High-confidence matches (>80%) execute as commands
- Ambiguous text or sentences are typed as dictation
- "type hello world" forces dictation mode

### Built-in Commands

**Window Management:**
| Voice | Action |
|-------|--------|
| "switch window" / "next window" | Alt+Tab |
| "previous window" | Alt+Shift+Tab |
| "focus [app]" | Activate window by name |
| "minimize" | Minimize window |
| "maximize" | Maximize window |
| "close window" | Alt+F4 |
| "next workspace" | Super+PageDown |
| "previous workspace" | Super+PageUp |

**Text Editing:**
| Voice | Action |
|-------|--------|
| "select all" | Ctrl+A |
| "copy" / "copy that" | Ctrl+C |
| "paste" / "paste that" | Ctrl+V |
| "cut" / "cut that" | Ctrl+X |
| "undo" | Ctrl+Z |
| "redo" | Ctrl+Shift+Z |
| "delete line" | Ctrl+Shift+K |
| "save" / "save file" | Ctrl+S |
| "new line" / "enter" | Return |
| "escape" | Escape |

**Other:**
| Voice | Action |
|-------|--------|
| "open [app]" | Launch application |
| "search for [query]" / "google [query]" | Web search |
| "type [text]" | Force typing (bypass command detection) |

### Dictation Features

**Punctuation (with smart spacing):**
| Voice | Output |
|-------|--------|
| "period" / "full stop" | . |
| "comma" | , |
| "question mark" | ? |
| "exclamation mark" | ! |
| "colon" | : |
| "semicolon" | ; |
| "dash" | - |
| "open quote" / "close quote" | " |
| "open paren" / "close paren" | ( ) |
| "ellipsis" | ... |
| "at sign" | @ |
| "hashtag" | # |

Example: "hello period how are you question mark" → "hello. how are you?"

**Correction:**
| Voice | Action |
|-------|--------|
| "scratch that" / "scratch" | Delete last transcription |
| "undo that" | Same as scratch |

**Paragraph Control:**
| Voice | Action |
|-------|--------|
| "new line" / "newline" | Single Enter |
| "new paragraph" | Double Enter |

### Custom Commands

Create `~/.config/voice-typing/commands.yaml`:
```yaml
commands:
  "open browser":
    type: launch
    app: firefox

  "screenshot":
    type: key
    keys: Print

  "lock screen":
    type: key
    keys: super+l
```

### Ambiguity Detection
Phrases that look like sentences are typed as dictation:
- Starts with pronoun: "I need to switch window" → dictation
- Questions: "Can you open terminal?" → dictation
- Contains conjunctions: "copy and paste it" → dictation

### Code Locations
- `commands.py` - CommandDetector and CommandExecutor classes
- Detection logic: `commands.py:144` (detect method)
- Integration: `enhanced-voice-typing.py:386` (_process_audio)

## Architecture

### Core Files
- `enhanced-voice-typing.py` - Main implementation
- `commands.py` - Voice command detection and execution
- `audio_visualizer.py` - GTK4 spectrum analyzer overlay
- `voice` - Launcher script with defaults
- `voice-toggle` - Wayland pause/resume helper
- `shell.nix` - Nix environment
- `ydotool-service.nix` - NixOS ydotool daemon (future)

### Threading Model
The application uses a **producer-consumer pattern** with up to 5 threads:

1. **Audio Callback Thread** (PyAudio) - Fast, non-blocking
   - Runs VAD on each 20ms chunk
   - Manages pre-buffer (circular deque)
   - Queues completed recordings for transcription
   - Checks `is_paused` flag
   - Protected by `buffer_lock`

2. **Transcription Worker Thread** - Background processing
   - Pulls audio from `transcription_queue`
   - Runs Whisper inference (1-2s)
   - Types result via xdotool

3. **Hotkey Listener Thread** (pynput) - Global hotkey
   - Toggles `is_paused` flag

4. **Socket Listener Thread** - Wayland fallback
   - Accepts toggle/pause/resume commands
   - Path: `${XDG_RUNTIME_DIR:-/tmp}/voice-typing-$UID.sock`

5. **Visualizer Thread** (GTK4) - Audio visualization (optional)
   - Runs GTK main loop for overlay window
   - Receives audio via thread-safe queue
   - Computes FFT spectrum at ~30fps
   - Auto-shows on speech, hides after silence delay

### Audio Pipeline
1. **Pre-Buffer**: Circular buffer holds 600ms (30 chunks × 20ms)
2. **VAD Detection**: WebRTC VAD (aggressiveness=2) detects speech
3. **Recording**: Pre-buffer + live audio + 800ms post-silence
4. **Queue**: Audio buffer copied and queued (non-blocking)
5. **Transcription**: Worker thread runs Whisper with beam_size=5
6. **Output**: xdotool types transcribed text

### Key Code Locations
- `VoiceTyping` class: `enhanced-voice-typing.py:55`
- Hotkey toggle: `enhanced-voice-typing.py:171`
- Socket listener: `enhanced-voice-typing.py:226`
- Audio callback: `enhanced-voice-typing.py:264`
- Transcription worker: `enhanced-voice-typing.py:310`
- Whisper transcription: `enhanced-voice-typing.py:322`

## Whisper Settings (Accuracy-Optimized)

Settings in `_process_audio()` at line 336:
- `beam_size=5` - Beam search for accuracy
- `condition_on_previous_text=True` - Context awareness enabled
- `vad_filter=True` with `min_silence_duration_ms=1000`, `speech_pad_ms=400`
- Quality filters use faster-whisper defaults
- Context prompt: Last 200 chars of previous transcription

### GPU Optimizations
When `--device cuda`:
- TF32 Tensor Core precision enabled
- cudnn benchmark mode
- 90% GPU memory allocation
- Pinned memory for CPU-GPU transfers
- Model warm-up on startup

## Configuration Constants

| Setting | Value | Location |
|---------|-------|----------|
| Sample rate | 16kHz | line 63 |
| Chunk size | 320 samples (20ms) | line 64 |
| Pre-buffer | 30 chunks (600ms) | line 83 |
| Post-buffer | 40 chunks (800ms) | line 84 |
| VAD aggressiveness | 2 | line 82 |
| Beam size | 5 | line 341 |
| Socket path | ${XDG_RUNTIME_DIR:-/tmp}/voice-typing-$UID.sock | line 61 |
| Socket token | ${XDG_RUNTIME_DIR:-/tmp}/voice-typing-$UID.token | line 62 |

## Thread Safety

- `buffer_lock` protects `recording_buffer`, `is_recording`, `silence_chunks`, `pre_buffer`
- `transcription_queue` (thread-safe Queue) passes audio between threads
- `is_paused` flag is atomic (single boolean)
- Audio chunks are `.copy()`'d before storage to prevent mutation

## Current Limitations

- pynput may not work on pure Wayland (use socket fallback)
- Custom shell commands are disabled unless `--allow-shell` is set

## Testing

```bash
arecord -d 5 test.wav && aplay test.wav  # Test mic
xdotool type "test"                       # Test keyboard injection
./voice-toggle                            # Test pause toggle
```

### Test Cases
1. Quick: "Hello world, this is a test"
2. Pauses: "The cat... sat on the mat" (400ms pause)
3. Rapid fire: Multiple utterances quickly
4. Long form: Full paragraph dictation
5. Pause test: Speak → F12 → speak → F12 → verify only resumed speech typed

<active-work>
todos: /home/jordan/.claude/todos/d36a0e54-ef28-4f25-aa61-90bff23a94d9-agent-d36a0e54-ef28-4f25-aa61-90bff23a94d9.json
plan: /home/jordan/.claude/plans/soft-doodling-phoenix.md
</active-work>
