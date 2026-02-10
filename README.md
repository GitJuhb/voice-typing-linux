# Voice Typing for Linux

Fast, accurate voice typing for Linux with IBus atomic text insertion, two-pass streaming STT, and CUDA acceleration. Works on Wayland and X11 — in terminals, browsers, and every app.

## Features

- **IBus input method engine** — Atomic text insertion via `commit_text`. No key injection lag, no garbled output in terminals. One unified path for every app.
- **Two-pass streaming STT** — sherpa-onnx streams words as you speak (~100ms latency), then faster-whisper turbo refines for accuracy. Text appears instantly, corrections happen seamlessly.
- **Preedit-until-refinement** — Streaming partials stay as preedit (preview text) until Whisper confirms or corrects them. No visible backspacing or flickering.
- **GPU acceleration** — TF32 Tensor Cores, cudnn benchmark mode, pinned memory transfers, model warm-up. Refinement takes ~0.1-0.2s on CUDA.
- **Pre-recording buffer** — 600ms circular buffer captures speech before VAD triggers. Never miss the first word.
- **Voice commands** — Window management, text editing, app launching, web search. Automatic dictation vs command disambiguation.
- **Audio visualizer** — GTK4 spectrum analyzer overlay, auto-shows on speech, auto-hides on silence.
- **Push-to-talk** — Hold or toggle modes with configurable hotkey.
- **NixOS-ready** — Full Nix shell with all dependencies, NixOS service module included.

## Quick Start

```bash
git clone https://github.com/GitJuhb/voice-typing-linux.git
cd voice-typing-linux

# NixOS (recommended)
nix-shell
./voice --streaming --device cuda

# Other distros
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python enhanced-voice-typing.py --streaming --device cuda
```

## Architecture

Two processes communicate via Unix socket:

```
                        ┌─────────────────────────────────────────────┐
                        │         enhanced-voice-typing.py            │
                        │                                             │
  Microphone ──▶ PyAudio ──▶ WebRTC VAD ──▶ Pre-Buffer (600ms)       │
                        │         │                                   │
                        │         ▼                                   │
                        │   ┌──────────┐     ┌───────────────────┐    │
                        │   │ sherpa-  │     │ faster-whisper    │    │
                        │   │ onnx     │────▶│ turbo (refine)    │    │
                        │   │ (stream) │     │ (GPU/CPU)         │    │
                        │   └────┬─────┘     └────────┬──────────┘    │
                        │        │ partials           │ final text    │
                        └────────┼────────────────────┼───────────────┘
                                 │                    │
                            Unix socket           Unix socket
                           preedit:text           commit:text
                                 │                    │
                        ┌────────┴────────────────────┴───────────────┐
                        │           ibus_voice_engine.py              │
                        │                                             │
                        │  IBus.Engine ──▶ update_preedit_text()      │
                        │              ──▶ commit_text()              │
                        │                                             │
                        │  Keyboard passthrough (do_process_key_event │
                        │  returns False — normal typing unaffected)  │
                        └─────────────────────────────────────────────┘
                                          │
                                          ▼
                                    Focused App
                              (Ghostty, Firefox, etc.)
```

**Pass 1 (sherpa-onnx):** Streams each 20ms audio chunk through a zipformer transducer. Partial results update the IBus preedit text in real-time.

**Pass 2 (faster-whisper turbo):** On endpoint detection (silence), accumulated audio goes to Whisper large-v3-turbo. The preedit clears and the refined text commits atomically. If streaming and refined text match, it confirms. If they differ, it corrects — no backspacing, no flicker.

**Fallback:** If the IBus engine isn't running, falls back to direct uinput key injection via python-evdev (sub-millisecond), then ydotool, then xdotool.

## IBus Setup

The IBus engine gives you atomic text insertion in every app — terminals, browsers, editors.

### 1. Install the component

```bash
mkdir -p ~/.local/share/ibus/component
cp voice-typing-ibus.xml ~/.local/share/ibus/component/
```

Edit the `<exec>` path in the XML to point to your checkout's `ibus-engine-voice-typing` script.

### 2. Restart IBus and add the engine

```bash
ibus restart
# Add "Voice Typing" input source in GNOME Settings → Keyboard → Input Sources
# Or via CLI:
ibus engine voice-typing
```

### 3. Run both processes

```bash
# Terminal 1: IBus engine
python ibus_voice_engine.py

# Terminal 2: Voice typing
./voice --streaming --device cuda
```

When the IBus engine is running, voice typing auto-detects it and routes all text through IBus. When it's not running, key injection is used as fallback.

## Usage

```bash
# Default batch mode (speak → pause → text appears)
./voice

# Streaming mode (words appear as you speak, refined on pause)
./voice --streaming
./voice --streaming --device cuda --model large-v3-turbo

# Smaller streaming model (~20MB instead of ~80MB)
./voice --streaming --streaming-model zipformer-en-20M

# Audio visualizer overlay
./voice --viz --viz-position top-right

# Voice commands
./voice --commands
./voice --commands --command-arm --command-arm-seconds 10

# Push-to-talk
./voice --ptt --ptt-hotkey f9 --ptt-mode hold

# Custom hotkey, language, model
./voice --hotkey f11 --language es --model medium

# Noise controls
./voice --calibrate-seconds 1.0 --noise-gate --agc

# List audio devices
./voice --list-devices
./voice --input-device "Jabra Evolve2 30"
```

### Pause/Resume

- **X11/XWayland:** Press F12 (pynput handles it directly)
- **Wayland:** Bind F12 in your compositor to `./voice-toggle`, or: `echo toggle | nc -U /run/user/$UID/voice-typing-$UID.sock`

### Models

First run downloads models automatically to `~/.cache/`.

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| tiny | 39 MB | Fastest | Quick notes |
| base | 74 MB | Fast | General typing |
| small | 244 MB | Moderate | Good balance |
| large-v3-turbo | ~1.5 GB | Fast (GPU) | **Best for streaming refinement** |

Streaming models (sherpa-onnx): `zipformer-en` (~80MB), `zipformer-en-20M` (~20MB).

## Voice Commands

Enable with `--commands`. Spoken text is analyzed for command patterns — high-confidence matches execute as commands, everything else is typed as dictation.

| Voice | Action |
|-------|--------|
| "switch window" | Alt+Tab |
| "close window" | Alt+F4 |
| "select all" / "copy" / "paste" | Ctrl+A / Ctrl+C / Ctrl+V |
| "undo" / "redo" | Ctrl+Z / Ctrl+Shift+Z |
| "new line" / "new paragraph" | Enter / Double Enter |
| "scratch that" | Delete last transcription |
| "open [app]" | Launch application |
| "search for [query]" | Web search |
| "type [text]" | Force dictation mode |

Punctuation: "period", "comma", "question mark", "exclamation mark", etc. — inserted with smart spacing.

Custom commands via `~/.config/voice-typing/commands.yaml`.

## Configuration

Config file: `~/.config/voice-typing/config.yaml`

```yaml
model: large-v3-turbo
device: cuda
streaming: true
streaming_model: zipformer-en
commands: true
noise_gate: true
adaptive_vad: true
```

Environment overrides (prefix `VOICE_`): `VOICE_MODEL`, `VOICE_DEVICE`, `VOICE_HOTKEY`, `VOICE_STREAMING`, `VOICE_STREAMING_MODEL`, `VOICE_REFINEMENT_MODEL`, `VOICE_COMMANDS`, `VOICE_NOISE_GATE`, `VOICE_PTT`, `VOICE_LOG_FILE`, `VOICE_ADAPTIVE_VAD`.

## Project Structure

```
voice-typing-linux/
├── voice                      # Launcher script
├── voice-toggle               # Wayland pause/resume helper
├── enhanced-voice-typing.py   # Main STT pipeline, IBus client, streaming worker
├── ibus_voice_engine.py       # IBus input method engine (separate process)
├── ibus-engine-voice-typing   # IBus engine launcher script
├── voice-typing-ibus.xml      # IBus component descriptor
├── streaming_stt.py           # sherpa-onnx streaming wrapper
├── commands.py                # Voice command detection and execution
├── audio_visualizer.py        # GTK4 spectrum analyzer overlay
├── shell.nix                  # Nix environment (Python + system deps)
├── ydotool-service.nix        # NixOS ydotool daemon module
├── nix/voice-typing.nix       # NixOS service module
├── systemd/                   # systemd user service template
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Package metadata
└── setup.py                   # Package setup
```

## Threading Model

Up to 6 concurrent threads:

1. **Audio callback** (PyAudio) — Non-blocking VAD + pre-buffer, queues recordings
2. **Transcription worker** — Whisper inference, refinement comparison
3. **Streaming worker** — sherpa-onnx real-time partials, endpoint detection
4. **Hotkey listener** (pynput) — Global F12 toggle
5. **Socket listener** — Wayland fallback, accepts toggle/pause/resume
6. **Visualizer** (GTK4) — FFT spectrum overlay at ~30fps

## Troubleshooting

### No audio input
```bash
# Check PipeWire sources
wpctl status | grep -A5 Sources
wpctl set-default <device-id>  # Set correct mic

# Test recording
arecord -d 5 test.wav && aplay test.wav
```

### IBus engine not connecting
```bash
# Check if engine is registered
ibus list-engine | grep voice

# Restart IBus
ibus restart

# Verify socket exists
ls /run/user/$UID/voice-typing-ibus-$UID.sock
```

### Text not appearing (Wayland)
```bash
# Check if uinput is accessible (fallback mode)
ls -la /dev/uinput
sudo usermod -aG input $USER  # Then logout/login
```

## Technical Details

- **Speech Recognition:** OpenAI Whisper via faster-whisper (CTranslate2)
- **Streaming STT:** sherpa-onnx zipformer transducer
- **Text Insertion:** IBus commit_text (primary), evdev uinput (fallback), ydotool/xdotool (legacy)
- **Audio:** PyAudio + PortAudio, 16kHz mono, 20ms chunks
- **VAD:** WebRTC Voice Activity Detection (aggressiveness 2)
- **Pre-buffer:** 600ms (30 chunks), post-silence: 800ms (40 chunks)
- **GPU:** TF32 Tensor Cores, cudnn benchmark, 90% VRAM allocation, pinned memory

## License

MIT License — see [LICENSE](LICENSE)

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — speech recognition model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — CTranslate2 optimized inference
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — streaming speech recognition
- [IBus](https://github.com/ibus/ibus) — intelligent input bus for Linux
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) — pre-buffer technique inspiration
