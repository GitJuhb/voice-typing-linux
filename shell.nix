{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    python311Packages.pip
    python311Packages.virtualenv
    python311Packages.tkinter  # Required by pyautogui
    
    # System dependencies
    ffmpeg
    sox
    ydotool
    xdotool
    portaudio  # For pyaudio
    
    # For pyautogui
    scrot  # Screenshot tool
    xorg.libX11
    xorg.libXext
    xorg.libXinerama
    
    # Build dependencies
    gcc
    pkg-config
    stdenv.cc.cc.lib
    zlib
  ];
  
  shellHook = ''
    # Set up library paths
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d .venv ]; then
      echo "Creating virtual environment..."
      python -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install Python packages
    echo "Installing/updating Python packages..."
    pip install --upgrade pip
    
    # Install RealtimeSTT and dependencies
    pip install RealtimeSTT
    pip install faster-whisper
    pip install pyautogui
    pip install python-xlib  # Required by pyautogui on Linux
    pip install pillow       # For pyautogui screenshots
    pip install opencv-python-headless  # Optional for pyautogui
    
    # Install torch (CPU version by default, comment out and use CUDA version if needed)
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    # For CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118
    
    echo ""
    echo "Voice typing environment ready!"
    echo "Run: python realtime-voice-typing.py"
    echo "Or: python realtime-voice-typing.py --model base --device cuda"
    echo ""
  '';
}