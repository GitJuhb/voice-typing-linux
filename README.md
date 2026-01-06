# Voice Typing for Linux

A fast, accurate voice typing solution for Linux that works seamlessly on both Wayland and X11. Uses OpenAI's Whisper model through `faster-whisper` for speech recognition with a pre-recording buffer to ensure no words are missed at the beginning of speech.

## ✨ Features

- 🎤 **Never miss a word** - Pre-recording buffer captures speech before voice detection triggers
- 🚀 **Fast and accurate** - Uses `faster-whisper` with optimized settings
- 🖥️ **Cross-platform** - Works on both Wayland (ydotool) and X11 (xdotool)
- 🔧 **Easy to use** - Simple command: just type `voice`
- 🎯 **Multiple models** - Choose between tiny (fastest) to large (most accurate)
- 🐧 **NixOS-ready** - Includes Nix shell configuration

## 📸 Demo

```bash
$ voice
Using 'small' model for better accuracy
Options: tiny (fastest), base, small, medium, large-v2, large-v3 (most accurate)
Running on Wayland - will use ydotool (requires ydotoold)
Auto-detected device: cpu

🎤 Enhanced voice typing active!
Speak naturally - initial words won't be missed!
Press Ctrl+C to stop

🎤 [processing] ✓ Hello, this is a test of the voice typing system.
```

## 🚀 Installation

### Prerequisites

- Linux (tested on NixOS, Ubuntu, Arch)
- Python 3.11+
- Microphone
- For Wayland: `ydotool`
- For X11: `xdotool`

### Quick Install

1. Clone the repository:
```bash
git clone https://github.com/GitJuhb/voice-typing-linux.git
cd voice-typing-linux
```

2. **For NixOS users** (recommended):
```bash
nix-shell
./voice  # or create an alias
```

3. **For other Linux distributions**:
```bash
# Install system dependencies
sudo apt-get install portaudio19-dev python3-dev python3-pip xdotool  # Ubuntu/Debian
sudo pacman -S portaudio python python-pip xdotool                    # Arch
sudo dnf install portaudio-devel python3-devel python3-pip xdotool    # Fedora

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Run
./voice
```

### Wayland Setup

For Wayland users, you need to set up ydotool:

1. Install ydotool:
```bash
# Ubuntu/Debian
sudo apt-get install ydotool

# Arch
sudo pacman -S ydotool

# NixOS - already included in shell.nix
```

2. **For NixOS**, add the included `ydotool-service.nix` to your configuration:
```nix
# In your configuration.nix
imports = [
  ./ydotool-service.nix  # Or copy the contents
];
```

3. **For other distributions**, start ydotoold:
```bash
# Run as user (temporary)
ydotoold &

# Or create a systemd service (permanent)
sudo systemctl enable --now ydotoold
```

**Wayland note:** Pause/resume uses a per-user socket and token in
`${XDG_RUNTIME_DIR:-/tmp}`. `./voice-toggle` handles this automatically.

## 📖 Usage

### Basic Usage

```bash
# Start voice typing with default settings (small model)
voice

# Use a different model
voice --model base    # More accurate
voice --model tiny    # Fastest
voice --model large   # Most accurate

# Use GPU acceleration (if available)
voice --device cuda
```

### Safety & Reliability Options

```bash
# Require "command mode" before executing voice commands
voice --commands --command-arm --command-arm-seconds 10

# Ask for confirmation when confidence is low
voice --commands --command-min-confidence 0.8 --command-confirm-below 0.9

# Allow custom shell commands from commands.yaml
voice --commands --allow-shell

# Limit max chunk length and queue depth (reduce memory spikes)
voice --max-seconds 20 --queue-size 2

# Noise calibration + noise gate + AGC
voice --calibrate-seconds 1.0 --noise-gate --noise-gate-multiplier 1.5
voice --agc --agc-target-rms 4000

# Push-to-talk (hold or toggle)
voice --ptt --ptt-hotkey f9 --ptt-mode hold

# Input device selection
voice --list-devices
voice --input-device 2
voice --input-device "USB Microphone"

# Status + notifications
voice --status-interval 2 --notify
```

## ⚙️ Configuration

Default config: `${XDG_CONFIG_HOME:-~/.config}/voice-typing/config.yaml`

Example:
```yaml
model: small
device: auto
commands: true
command_arm: true
command_arm_seconds: 10
noise_gate: true
agc: true
adaptive_vad: true
ptt: false
status_interval: 2
```

Environment overrides (prefix `VOICE_`): `VOICE_MODEL`, `VOICE_DEVICE`, `VOICE_HOTKEY`,
`VOICE_COMMANDS`, `VOICE_NOISE_GATE`, `VOICE_PTT`, `VOICE_LOG_FILE`,
`VOICE_ADAPTIVE_VAD` (or legacy `VOICE_NO_ADAPTIVE_VAD`).

## 🧾 Logs

Rotating log file defaults to:
`${XDG_STATE_HOME:-~/.local/state}/voice-typing/voice-typing.log`

Configure via:
```bash
voice --log-file ~/.local/state/voice-typing/voice-typing.log --log-max-bytes 1000000 --log-backups 5
```

## 🧰 systemd user service

Template: `systemd/voice-typing.service` (edit paths if needed)
```bash
systemctl --user enable --now /path/to/repo/systemd/voice-typing.service
```

## 📦 pipx

```bash
pipx install .
voice
```

## ❄️ Nix module

See `nix/voice-typing.nix` for a minimal module.

### Available Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny | 39 MB | Fastest | Good | Quick notes, commands |
| base | 74 MB | Fast | Better | General typing |
| small | 244 MB | Moderate | Great | **Default - best balance** |
| medium | 769 MB | Slower | Excellent | Professional documents |
| large | 1550 MB | Slowest | Best | Maximum accuracy |

### Tips for Best Results

1. **Speak naturally** - The system waits for natural pauses
2. **Complete sentences** - Speak in complete thoughts for best accuracy
3. **Clear speech** - Enunciate clearly but naturally
4. **Quiet environment** - Reduces errors from background noise

## 🔧 How It Works

The voice typing system uses several key techniques:

1. **Pre-recording Buffer**: Continuously records the last 1.5 seconds of audio in a circular buffer
2. **Voice Activity Detection**: Uses WebRTC VAD to detect when you start speaking
3. **Smart Buffering**: When speech is detected, includes the pre-recorded buffer to capture the beginning
4. **Whisper Transcription**: Uses faster-whisper for accurate speech-to-text conversion
5. **Keyboard Injection**: Types the transcribed text using ydotool (Wayland) or xdotool (X11)

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│ Microphone  │────▶│  Pre-Buffer +    │────▶│   Whisper   │
│   (Audio)   │     │  VAD Detection   │     │   (STT)     │
└─────────────┘     └──────────────────┘     └─────────────┘
                                                     │
                                                     ▼
                                              ┌─────────────┐
                                              │ ydotool/    │
                                              │ xdotool     │
                                              └─────────────┘
```

## ⚙️ Configuration

Create an alias in your shell configuration:

```bash
# ~/.bashrc or ~/.zshrc
alias voice="/path/to/voice-typing-linux/voice"
alias "voice stop"="pkill -f 'python.*enhanced-voice-typing'"
```

For NixOS users with Home Manager:

```nix
home.shellAliases = {
  voice = "/path/to/voice-typing-linux/voice";
  "voice stop" = "pkill -f 'python.*enhanced-voice-typing'";
};
```

## 🔍 Troubleshooting

### "No microphone found"
```bash
# Check your microphone is connected
arecord -l

# Test recording
arecord -d 5 test.wav && aplay test.wav

# Ensure PulseAudio/PipeWire is running
pactl info
```

### "ydotool: failed to connect socket"
```bash
# Check if ydotoold is running
ps aux | grep ydotoold

# Start it manually
ydotoold &

# Check permissions
ls -la /dev/uinput
```

### Missing words at the beginning
- This should not happen with the pre-buffer!
- If it does, try increasing `PRE_BUFFER_DURATION_SEC` in `enhanced-voice-typing.py`

### Poor accuracy
- Try a larger model: `voice --model base` or `voice --model small`
- Check microphone quality: `arecord -V stereo -r 16000 -f S16_LE -d 5 test.wav`
- Reduce background noise

### High CPU usage
- Use a smaller model: `voice --model tiny`
- Enable GPU acceleration if available: `voice --device cuda`

## 🛠️ Technical Details

- **Speech Recognition**: OpenAI Whisper via faster-whisper (CTranslate2 optimized)
- **Audio Backend**: PyAudio with PortAudio
- **VAD**: WebRTC Voice Activity Detection
- **Keyboard Input**: ydotool (Wayland) / xdotool (X11)
- **Pre-buffer**: 1.5 seconds circular buffer
- **Language**: Python 3.11+
- **Sample Rate**: 16kHz mono

### Project Structure

```
voice-typing-linux/
├── voice                      # Main launcher script
├── enhanced-voice-typing.py   # Core voice typing implementation
├── shell.nix                  # Nix shell configuration
├── requirements.txt           # Python dependencies
├── ydotool-service.nix       # NixOS ydotool service
└── README.md                 # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Some areas for improvement:

- [ ] Add support for multiple languages
- [ ] Implement hotkey to toggle voice typing
- [ ] Add punctuation commands ("period", "comma", etc.)
- [ ] Create GUI for settings
- [ ] Add support for custom wake words
- [ ] Implement noise suppression

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- [OpenAI](https://github.com/openai/whisper) for the Whisper model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) team for the optimized implementation
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) for inspiration on the pre-buffer technique
- The NixOS community for the excellent packaging system

## 💡 Tips & Tricks

1. **For coding**: Use "base" or "small" model for better technical term recognition
2. **For long sessions**: The model stays loaded, so no startup delay after first use
3. **Multiple languages**: Add `--language es` for Spanish, `--language fr` for French, etc.
4. **Save battery**: Use CPU mode on laptops: `voice --device cpu`

---

Created with ❤️ for the Linux community. If you find this useful, please star the repository!
