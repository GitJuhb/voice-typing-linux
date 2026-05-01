# Voice Typing for Linux

Fast, accurate voice typing for Linux with IBus atomic text insertion, streaming STT, and optional post-commit correction. The default streaming path is now NVIDIA Riva ASR NIM with `nemotron-asr-streaming-nim`, while local Parakeet CTC remains available as the zero-service fallback. Works on Wayland and X11 — in terminals, browsers, and every app.

## Features

- **IBus input method engine** — Atomic text insertion via `commit_text`. No key injection lag, no garbled output in terminals. One unified path for every app.
- **Streaming-first STT** — NVIDIA Riva ASR NIM `nemotron-asr-streaming-nim` is now the default streaming backend. Local buffered `parakeet-ctc-0.6b` remains available as the zero-service fallback, older Parakeet CTC NIM profiles remain selectable, Moonshine native remains selectable, Parakeet TDT remains available as an optional post-commit correction model, and zipformer remains available as a sherpa fallback.
- **Immediate IBus commit** — Endpoint text commits immediately on IBus. Optional post-commit correction can replace the last utterance afterward when enabled.
- **GPU acceleration** — TF32 Tensor Cores, cudnn benchmark mode, pinned memory transfers, model warm-up.
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
                        │   │ Moonshine│     │ sherpa-onnx       │    │
                        │   │ native   │────▶│ parakeet optional │    │
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

**Pass 1 (streaming):** The default streaming path is NVIDIA Riva ASR NIM with `nemotron-asr-streaming-nim`. Local buffered `parakeet-ctc-0.6b` remains the fallback that works without any external service. The older Parakeet CTC NIM profiles remain available as compatibility baselines. Moonshine native remains available for local true-online partials, and zipformer remains available as the sherpa true-online fallback.

**Pass 2 (optional post-commit correction):** When enabled, endpoint audio can go to Parakeet TDT after the streaming text is already committed. If the correction is accepted, the last utterance is replaced in place.

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

# Streaming mode (words appear as you speak)
./voice --streaming
./voice --streaming --streaming-model parakeet-ctc-0.6b
./voice --streaming --post-commit-correction --device cuda
./voice --streaming --post-commit-correction --correction-model large-v3-turbo

# Streaming model selection
./voice --streaming --streaming-model parakeet-ctc-0.6b
VOICE_NIM_URL=http://127.0.0.1:9000 ./voice --streaming --streaming-model nemotron-asr-streaming-nim
VOICE_NIM_URL=http://127.0.0.1:9000 ./voice --streaming --streaming-model parakeet-ctc-0.6b-nim
VOICE_NIM_URL=http://127.0.0.1:9000 ./voice --streaming --streaming-model parakeet-ctc-1.1b-nim
./voice --streaming --streaming-model zipformer-en-20M
./voice --streaming --streaming-model moonshine-tiny-streaming-en
./voice --streaming --streaming-model moonshine-small-streaming-en
./voice --streaming --streaming-model moonshine-medium-streaming-en

# Batch Parakeet
./voice --model parakeet-tdt-0.6b-v2 --device cuda

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

| Model | Size | Backend | Use Case |
|-------|------|---------|----------|
| tiny / base / small / medium / large-v3-turbo | 39 MB to 1.5 GB | faster-whisper | Multilingual batch / fallback |
| parakeet-tdt-0.6b-v2 | ~300 MB | sherpa-onnx | Default English batch/post-commit correction |

Streaming models:
- `parakeet-ctc-0.6b` (default, buffered local streaming)
- `nemotron-asr-streaming-nim` (default and recommended NVIDIA Riva ASR NIM realtime backend)
- `parakeet-ctc-0.6b` (buffered local fallback)
- `parakeet-ctc-0.6b-nim` (NVIDIA Riva ASR NIM realtime websocket backend)
- `parakeet-ctc-1.1b-nim` (older Parakeet CTC NIM baseline on large cards)
- `moonshine-medium-streaming-en` (native streaming)
- `moonshine-small-streaming-en` (smaller native streaming)
- `moonshine-tiny-streaming-en` (smallest native streaming)
- `zipformer-en` (sherpa true-online fallback)
- `zipformer-en-20M` (small sherpa true-online fallback)

### NVIDIA Nemotron NIM

Best GPU streaming setup on a large NVIDIA card:

```bash
export NGC_API_KEY=<your-ngc-key>
docker login nvcr.io

docker run -it --rm --name=nemotron-asr-streaming \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -e NGC_API_KEY \
  -e NIM_HTTP_API_PORT=9000 \
  -e NIM_GRPC_API_PORT=50051 \
  -e NIM_TAGS_SELECTOR=mode=str \
  -p 9000:9000 \
  -p 50051:50051 \
  nvcr.io/nim/nvidia/nemotron-asr-streaming:latest

curl http://localhost:9000/v1/health/ready
VOICE_NIM_URL=http://127.0.0.1:9000 ./voice --streaming --streaming-model nemotron-asr-streaming-nim --device cuda
```

Older Parakeet CTC NIM profiles remain supported when you explicitly select `parakeet-ctc-0.6b-nim` or `parakeet-ctc-1.1b-nim`.

This backend talks to NIM over the official realtime websocket API:
- `POST /v1/realtime/transcription_sessions`
- `WS /v1/realtime?intent=transcription`

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
model: parakeet-tdt-0.6b-v2
device: cuda
streaming: true
streaming_model: nemotron-asr-streaming-nim
post_commit_correction: false
correction_model: parakeet-tdt-0.6b-v2
commands: true
noise_gate: true
adaptive_vad: true
```

Environment overrides (prefix `VOICE_`): `VOICE_MODEL`, `VOICE_DEVICE`, `VOICE_HOTKEY`, `VOICE_STREAMING`, `VOICE_STREAMING_MODEL`, `VOICE_POST_COMMIT_CORRECTION`, `VOICE_CORRECTION_MODEL`, `VOICE_COMMANDS`, `VOICE_NOISE_GATE`, `VOICE_PTT`, `VOICE_LOG_FILE`, `VOICE_ADAPTIVE_VAD`, `VOICE_NIM_URL`, `VOICE_NIM_API_KEY`. Legacy `VOICE_REFINEMENT*` env vars are still accepted.

## Project Structure

```
voice-typing-linux/
├── voice                      # Launcher script
├── voice-toggle               # Wayland pause/resume helper
├── enhanced-voice-typing.py   # Main STT pipeline, IBus client, streaming worker
├── ibus_voice_engine.py       # IBus input method engine (separate process)
├── ibus-engine-voice-typing   # IBus engine launcher script
├── voice-typing-ibus.xml      # IBus component descriptor
├── streaming_stt.py           # Streaming backends + offline model wrappers
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
2. **Transcription worker** — Offline model inference, optional post-commit correction comparison
3. **Streaming worker** — Parakeet CTC buffered streaming, Moonshine native, or zipformer partials with endpoint detection
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

- **Speech Recognition:** sherpa-onnx Parakeet TDT (default) or Whisper via faster-whisper
- **Streaming STT:** Parakeet CTC by default, with Moonshine native and zipformer available as alternatives
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
- [Moonshine Voice](https://github.com/moonshine-ai/moonshine) — native streaming speech recognition
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — streaming and offline speech recognition
- [IBus](https://github.com/ibus/ibus) — intelligent input bus for Linux
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) — pre-buffer technique inspiration
