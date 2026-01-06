from pathlib import Path

from setuptools import setup


def read_requirements():
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.exists():
        return []
    lines = []
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


setup(
    name="voice-typing-linux",
    version="0.1.0",
    description="Voice typing for Linux using faster-whisper",
    scripts=["voice", "voice-toggle", "enhanced-voice-typing.py"],
    install_requires=read_requirements(),
)
