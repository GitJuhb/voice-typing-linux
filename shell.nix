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
    wmctrl     # Window focus commands
    netcat     # Socket toggle (voice-toggle)
    xbindkeys  # Wayland hotkey fallback
    portaudio  # For pyaudio
    
    # For pyautogui
    scrot  # Screenshot tool
    xorg.libX11
    xorg.libXext
    xorg.libXinerama

    # Audio visualization (GTK4 layer-shell overlay)
    gtk4
    gtk4-layer-shell
    gobject-introspection
    glib
    cairo
    python311Packages.pygobject3
    python311Packages.pycairo

    # Build dependencies
    gcc
    pkg-config
    stdenv.cc.cc.lib
    zlib
  ];
  
  shellHook = ''
    # Set up library paths
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"

    # GTK4 layer-shell must be preloaded BEFORE libwayland-client for Wayland overlays
    export LD_PRELOAD="${pkgs.gtk4-layer-shell}/lib/libgtk4-layer-shell.so''${LD_PRELOAD:+:$LD_PRELOAD}"

    # GTK4 layer-shell typelib path
    export GI_TYPELIB_PATH="${pkgs.gtk4}/lib/girepository-1.0:${pkgs.gtk4-layer-shell}/lib/girepository-1.0:${pkgs.glib.out}/lib/girepository-1.0:$GI_TYPELIB_PATH"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d .venv ]; then
      echo "Creating virtual environment..."
      python -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install Python packages only when requirements change
    echo "Checking Python packages..."
    pip install --upgrade pip

    if [ -f requirements.txt ]; then
      REQ_HASH="$(sha256sum requirements.txt | awk '{print $1}')"
      REQ_FILE=".venv/.requirements.sha"
      if [ ! -f "$REQ_FILE" ] || [ "$(cat "$REQ_FILE")" != "$REQ_HASH" ]; then
        echo "Installing/updating Python packages..."
        pip install -r requirements.txt
        echo "$REQ_HASH" > "$REQ_FILE"
      fi
    fi

    # Optional CUDA torch install (set VOICE_TYPING_CUDA=1)
    if [ -n "$VOICE_TYPING_CUDA" ]; then
      # Using CUDA 12.4 - driver 570.x supports CUDA 12.8.1 per research
      pip install torch --index-url https://download.pytorch.org/whl/cu124
    fi
    
    echo ""
    echo "Voice typing environment ready!"
    echo "Run: python enhanced-voice-typing.py"
    echo "Or: python enhanced-voice-typing.py --model base --device cuda"
    echo ""
  '';
}
