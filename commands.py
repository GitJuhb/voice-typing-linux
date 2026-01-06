"""
Voice command detection and execution for voice typing.

Provides automatic detection of voice commands vs dictation text,
with support for window management, text editing, and custom commands.
"""

import re
import subprocess
import os
import time
from pathlib import Path
from typing import Optional

# Try to load YAML for custom commands
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# =============================================================================
# PUNCTUATION COMMANDS
# =============================================================================

PUNCTUATION_COMMANDS = {
    # Sentence endings
    "period": ".",
    "full stop": ".",
    "comma": ",",
    "question mark": "?",
    "exclamation mark": "!",
    "exclamation point": "!",

    # Common punctuation
    "colon": ":",
    "semicolon": ";",
    "dash": " - ",
    "hyphen": "-",

    # Quotes and brackets
    "open quote": '"',
    "close quote": '"',
    "quote": '"',
    "open paren": "(",
    "close paren": ")",
    "open bracket": "[",
    "close bracket": "]",

    # Symbols
    "ellipsis": "...",
    "at sign": "@",
    "hashtag": "#",
    "dollar sign": "$",
    "percent": "%",
    "ampersand": "&",
    "asterisk": "*",
    "slash": "/",
    "backslash": "\\",
}

# Punctuation that should NOT have a space added after
NO_SPACE_AFTER = {'(', '"', "'", '[', ' - '}


def process_punctuation(text: str) -> str:
    """
    Replace spoken punctuation words with symbols, handling spacing.

    "hello period how are you" → "hello. how are you"
    "open paren test close paren" → "(test)"
    """
    words = text.split()
    result = []

    i = 0
    while i < len(words):
        word = words[i]
        word_lower = word.lower().rstrip('.,!?')

        # Check for two-word punctuation commands
        two_word = None
        if i + 1 < len(words):
            two_word = f"{word_lower} {words[i+1].lower().rstrip('.,!?')}"

        if two_word and two_word in PUNCTUATION_COMMANDS:
            punct = PUNCTUATION_COMMANDS[two_word]
            result.append(punct)
            if punct not in NO_SPACE_AFTER:
                result.append(' ')
            i += 2
        elif word_lower in PUNCTUATION_COMMANDS:
            punct = PUNCTUATION_COMMANDS[word_lower]
            result.append(punct)
            if punct not in NO_SPACE_AFTER:
                result.append(' ')
            i += 1
        else:
            if result and not result[-1].endswith(' '):
                result.append(' ')
            result.append(word)
            i += 1

    return ''.join(result).strip()


# =============================================================================
# BUILTIN COMMANDS
# =============================================================================

# Exact phrase commands (high confidence - these are rarely used in dictation)
EXACT_COMMANDS = {
    # Window management
    "switch window": {"intent": "window", "action": "alt_tab"},
    "next window": {"intent": "window", "action": "alt_tab"},
    "previous window": {"intent": "window", "action": "alt_tab_reverse"},
    "minimize": {"intent": "window", "action": "minimize"},
    "minimize window": {"intent": "window", "action": "minimize"},
    "maximize": {"intent": "window", "action": "maximize"},
    "maximize window": {"intent": "window", "action": "maximize"},
    "close window": {"intent": "window", "action": "close", "destructive": True},
    "next workspace": {"intent": "window", "action": "next_workspace"},
    "previous workspace": {"intent": "window", "action": "prev_workspace"},

    # Text editing
    "select all": {"intent": "edit", "action": "select_all"},
    "copy": {"intent": "edit", "action": "copy"},
    "copy that": {"intent": "edit", "action": "copy"},
    "paste": {"intent": "edit", "action": "paste"},
    "paste that": {"intent": "edit", "action": "paste"},
    "cut": {"intent": "edit", "action": "cut"},
    "cut that": {"intent": "edit", "action": "cut"},
    "undo": {"intent": "edit", "action": "undo"},
    "redo": {"intent": "edit", "action": "redo"},
    "delete line": {"intent": "edit", "action": "delete_line"},
    "save": {"intent": "edit", "action": "save"},
    "save file": {"intent": "edit", "action": "save"},
    "new line": {"intent": "edit", "action": "new_line"},
    "enter": {"intent": "edit", "action": "enter"},
    "tab": {"intent": "edit", "action": "tab"},
    "escape": {"intent": "edit", "action": "escape"},
    "go back": {"intent": "edit", "action": "go_back"},
    "go forward": {"intent": "edit", "action": "go_forward"},

    # Paragraph control
    "new paragraph": {"intent": "edit", "action": "new_paragraph"},
    "next paragraph": {"intent": "edit", "action": "new_paragraph"},
    "newline": {"intent": "edit", "action": "new_line"},

    # Correction commands
    "scratch that": {"intent": "correction", "action": "scratch"},
    "scratch": {"intent": "correction", "action": "scratch"},
    "undo that": {"intent": "correction", "action": "scratch"},
}

# Pattern commands with parameter capture (medium confidence)
PATTERN_COMMANDS = [
    # "focus firefox" → focus app named firefox
    (r"^focus (.+)$", {"intent": "window", "action": "focus", "param": "app"}),
    # "open terminal" → launch app
    (r"^open (.+)$", {"intent": "launch", "action": "open", "param": "app"}),
    # "search for cats" → web search
    (r"^search for (.+)$", {"intent": "search", "action": "web_search", "param": "query"}),
    # "google something" → web search
    (r"^google (.+)$", {"intent": "search", "action": "web_search", "param": "query"}),
    # "type hello world" → force dictation
    (r"^type (.+)$", {"intent": "force_dictation", "action": "type", "param": "text"}),
]

# Patterns that indicate the text is likely dictation, not a command
DICTATION_INDICATORS = [
    r"^(i|we|you|they|he|she|it|my|our|your|the|a|an)\s",  # Starts with pronoun/article
    r"\?$",                             # Questions
    r"(please|could|would|can|should|might)\s",  # Polite/modal verbs
    r"(is|was|are|were|be|been|being)\s",  # Being verbs in middle
    r"(that|which|who|whom|whose)\s",   # Relative pronouns
    r"\b(and|but|or|so|because|although|however)\b",  # Conjunctions
    r"(ing|ed|ly)\b.*\b(ing|ed|ly)\b",  # Multiple verb forms = sentence
]


# =============================================================================
# COMMAND EXECUTION
# =============================================================================

# xdotool key mappings
XDOTOOL_KEYS = {
    # Window management
    "alt_tab": "alt+Tab",
    "alt_tab_reverse": "alt+shift+Tab",
    "minimize": "super+h",
    "maximize": "super+Up",
    "close": "alt+F4",
    "next_workspace": "super+Page_Down",
    "prev_workspace": "super+Page_Up",

    # Text editing
    "select_all": "ctrl+a",
    "copy": "ctrl+c",
    "paste": "ctrl+v",
    "cut": "ctrl+x",
    "undo": "ctrl+z",
    "redo": "ctrl+shift+z",
    "delete_line": "ctrl+shift+k",
    "save": "ctrl+s",
    "new_line": "Return",
    "new_paragraph": "Return Return",
    "enter": "Return",
    "tab": "Tab",
    "escape": "Escape",
    "go_back": "alt+Left",
    "go_forward": "alt+Right",
}


class CommandDetector:
    """Detects voice commands and distinguishes them from dictation."""

    def __init__(self, custom_commands_path: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self.custom_commands = {}
        self.last_command = None

        if custom_commands_path and YAML_AVAILABLE:
            self._load_custom_commands(custom_commands_path)

    def _load_custom_commands(self, path: str):
        """Load custom commands from YAML file."""
        path = os.path.expanduser(path)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
                    if data and 'commands' in data:
                        self.custom_commands = data['commands']
                        print(f"📋 Loaded {len(self.custom_commands)} custom commands")
            except Exception as e:
                print(f"⚠️  Failed to load custom commands: {e}")

    def detect(self, text: str) -> tuple[str, float, dict]:
        """
        Detect if text is a command or dictation.

        Returns: (intent, confidence, params)
        - intent: 'dictation' | 'window' | 'edit' | 'launch' | 'search' | 'custom'
        - confidence: 0.0-1.0
        - params: extracted parameters (action, app name, etc.)
        """
        if not self.enabled:
            return ('dictation', 1.0, {})

        text_clean = text.lower().strip()

        # Remove common Whisper artifacts
        text_clean = text_clean.strip('.')
        text_clean = re.sub(r'\s+', ' ', text_clean)

        # Check if it looks like dictation first (fast rejection)
        if self._is_likely_dictation(text_clean):
            return ('dictation', 1.0, {})

        # Check exact matches (highest confidence)
        if text_clean in EXACT_COMMANDS:
            cmd = EXACT_COMMANDS[text_clean]
            return (cmd['intent'], 0.95, cmd)

        # Check custom commands
        for phrase, cmd_config in self.custom_commands.items():
            if text_clean == phrase.lower():
                return ('custom', 0.90, {'action': phrase, **cmd_config})

        # Check pattern commands
        for pattern, cmd in PATTERN_COMMANDS:
            match = re.match(pattern, text_clean)
            if match:
                params = cmd.copy()
                if 'param' in cmd and match.groups():
                    params[cmd['param']] = match.group(1).strip()
                return (cmd['intent'], 0.85, params)

        # Default to dictation
        return ('dictation', 1.0, {})

    def _is_likely_dictation(self, text: str) -> bool:
        """Check if text has patterns suggesting it's dictation, not a command."""
        # Very short = could be command
        if len(text.split()) <= 3:
            return False

        # Check dictation indicators
        for pattern in DICTATION_INDICATORS:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False


# Kernel keycodes for ydotool (Wayland)
# See: /usr/include/linux/input-event-codes.h
YDOTOOL_KEYCODES = {
    'Return': '28',
    'Tab': '15',
    'Escape': '1',
    'BackSpace': '14',
    'ctrl': '29',
    'shift': '42',
    'alt': '56',
    'super': '125',
    # Letters (lowercase)
    'a': '30', 'b': '48', 'c': '46', 'd': '32', 'e': '18', 'f': '33',
    'g': '34', 'h': '35', 'i': '23', 'j': '36', 'k': '37', 'l': '38',
    'm': '50', 'n': '49', 'o': '24', 'p': '25', 'q': '16', 'r': '19',
    's': '31', 't': '20', 'u': '22', 'v': '47', 'w': '17', 'x': '45',
    'y': '21', 'z': '44',
    # Function keys
    'F1': '59', 'F2': '60', 'F3': '61', 'F4': '62', 'F5': '63', 'F6': '64',
    'F7': '65', 'F8': '66', 'F9': '67', 'F10': '68', 'F11': '87', 'F12': '88',
    # Arrow keys
    'Up': '103', 'Down': '108', 'Left': '105', 'Right': '106',
    'Page_Up': '104', 'Page_Down': '109',
    'PageUp': '104', 'PageDown': '109',
}


class CommandExecutor:
    """Executes detected voice commands."""

    def __init__(self, voice_typing=None, allow_shell_commands: bool = False):
        self.last_executed = None
        self.voice_typing = voice_typing  # Reference for scratch that
        self.allow_shell_commands = allow_shell_commands
        self.destructive_confirm_window = 2.0
        self._pending_destructive = {}

    def _get_display_server(self):
        """Get display server from voice_typing or default to x11"""
        if self.voice_typing and hasattr(self.voice_typing, 'display_server'):
            return self.voice_typing.display_server
        return 'x11'

    def _send_key(self, key_combo: str) -> bool:
        """Send a key or key combo using xdotool or ydotool depending on display server."""
        display = self._get_display_server()

        if display == 'wayland':
            return self._send_key_ydotool(key_combo)
        else:
            return self._send_key_xdotool(key_combo)

    def _send_key_xdotool(self, key_combo: str) -> bool:
        """Send key using xdotool (X11)"""
        keys = key_combo.split()
        subprocess.run(['xdotool', 'key'] + keys)
        return True

    def _ydotool_env(self):
        """Get environment with ydotool socket path set"""
        env = os.environ.copy()
        env['YDOTOOL_SOCKET'] = '/run/ydotoold/socket'
        return env

    def _send_key_ydotool(self, key_combo: str) -> bool:
        """Send key using ydotool (Wayland)"""
        env = self._ydotool_env()
        # Parse xdotool-style key combo (e.g., "ctrl+c", "alt+Tab")
        for key_part in key_combo.split():
            if '+' in key_part:
                # Modifier combo like ctrl+c
                parts = key_part.lower().split('+')
                modifiers = parts[:-1]
                key = parts[-1]

                # Press modifiers
                for mod in modifiers:
                    if mod in YDOTOOL_KEYCODES:
                        subprocess.run(['ydotool', 'key', f'{YDOTOOL_KEYCODES[mod]}:1'], env=env)

                # Press and release key
                if key in YDOTOOL_KEYCODES:
                    kc = YDOTOOL_KEYCODES[key]
                    subprocess.run(['ydotool', 'key', f'{kc}:1', f'{kc}:0'], env=env)
                elif key.capitalize() in YDOTOOL_KEYCODES:
                    kc = YDOTOOL_KEYCODES[key.capitalize()]
                    subprocess.run(['ydotool', 'key', f'{kc}:1', f'{kc}:0'], env=env)

                # Release modifiers (reverse order)
                for mod in reversed(modifiers):
                    if mod in YDOTOOL_KEYCODES:
                        subprocess.run(['ydotool', 'key', f'{YDOTOOL_KEYCODES[mod]}:0'], env=env)
            else:
                # Single key
                key = key_part
                if key in YDOTOOL_KEYCODES:
                    kc = YDOTOOL_KEYCODES[key]
                    subprocess.run(['ydotool', 'key', f'{kc}:1', f'{kc}:0'], env=env)
                elif key.capitalize() in YDOTOOL_KEYCODES:
                    kc = YDOTOOL_KEYCODES[key.capitalize()]
                    subprocess.run(['ydotool', 'key', f'{kc}:1', f'{kc}:0'], env=env)

        return True

    def execute(self, intent: str, params: dict) -> bool:
        """
        Execute a command based on intent and parameters.

        Returns True if command was executed, False otherwise.
        """
        action = params.get('action', '')

        try:
            if params.get('destructive'):
                now = time.time()
                last = self._pending_destructive.get(action, 0.0)
                if now - last > self.destructive_confirm_window:
                    self._pending_destructive[action] = now
                    print("⚠️  Destructive command: say it again to confirm")
                    return True
                self._pending_destructive.pop(action, None)

            if intent == 'window':
                return self._execute_window(action, params)
            elif intent == 'edit':
                return self._execute_edit(action)
            elif intent == 'launch':
                return self._execute_launch(params)
            elif intent == 'search':
                return self._execute_search(params)
            elif intent == 'custom':
                return self._execute_custom(params)
            elif intent == 'correction':
                return self._execute_correction(action, params)
            elif intent == 'force_dictation':
                # Return False to let caller know to type the text
                return False
        except Exception as e:
            print(f"❌ Command execution failed: {e}")
            return False

        return False

    def _execute_correction(self, action: str, params: dict) -> bool:
        """Execute correction commands like scratch that."""
        if action == 'scratch':
            if self.voice_typing and hasattr(self.voice_typing, '_scratch_that'):
                return self.voice_typing._scratch_that()
            else:
                print("⚠️ Scratch not available (no voice_typing reference)")
                return False
        return False

    def _execute_window(self, action: str, params: dict) -> bool:
        """Execute window management commands."""
        if action == 'focus':
            app = params.get('app', '')
            if app:
                # Try wmctrl first (works on both X11 and Wayland with XWayland)
                result = subprocess.run(
                    ['wmctrl', '-a', app],
                    capture_output=True
                )
                if result.returncode != 0 and self._get_display_server() == 'x11':
                    # Fall back to xdotool search (X11 only)
                    subprocess.run([
                        'xdotool', 'search', '--name', app,
                        'windowactivate', '--sync'
                    ])
                return True
        elif action in XDOTOOL_KEYS:
            key = XDOTOOL_KEYS[action]
            return self._send_key(key)

        return False

    def _execute_edit(self, action: str) -> bool:
        """Execute text editing commands."""
        if action in XDOTOOL_KEYS:
            keys = XDOTOOL_KEYS[action]
            return self._send_key(keys)
        return False

    def _execute_launch(self, params: dict) -> bool:
        """Launch an application."""
        app = params.get('app', '')
        if not app:
            return False

        # Common app aliases
        app_map = {
            'browser': 'firefox',
            'terminal': 'x-terminal-emulator',
            'editor': 'code',
            'files': 'nautilus',
            'calculator': 'gnome-calculator',
        }

        app_cmd = app_map.get(app.lower(), app)

        try:
            subprocess.Popen(
                [app_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except FileNotFoundError:
            # Try xdg-open
            try:
                subprocess.Popen(
                    ['xdg-open', f'{app_cmd}:'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return True
            except:
                return False

    def _execute_search(self, params: dict) -> bool:
        """Execute web search."""
        query = params.get('query', '')
        if not query:
            return False

        # URL encode the query
        import urllib.parse
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/search?q={encoded_query}"

        subprocess.Popen(
            ['xdg-open', url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True

    def _execute_custom(self, params: dict) -> bool:
        """Execute custom user-defined command."""
        action = params.get('action', '')
        cmd_type = params.get('type', 'key')

        if cmd_type == 'key':
            # Send keyboard shortcut
            keys = params.get('keys', '')
            if keys:
                return self._send_key(keys)
        elif cmd_type == 'launch':
            # Launch app
            return self._execute_launch(params)
        elif cmd_type == 'shell':
            # Run shell command (be careful!)
            shell_cmd = params.get('command', '')
            if shell_cmd:
                if not self.allow_shell_commands:
                    print("⚠️  Shell commands are disabled (use --allow-shell)")
                    return True
                subprocess.Popen(
                    shell_cmd,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return True

        return False


def create_default_config(path: str = "~/.config/voice-typing/commands.yaml"):
    """Create default commands config file if it doesn't exist."""
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        return

    default_config = """# Voice Typing Custom Commands
# Add your own voice commands here

commands:
  # Example: Launch apps
  "open browser":
    type: launch
    app: firefox

  "open code":
    type: launch
    app: code

  # Example: Keyboard shortcuts
  "screenshot":
    type: key
    keys: Print

  "lock screen":
    type: key
    keys: super+l

  # Example: Shell commands (use with caution!)
  # "update system":
  #   type: shell
  #   command: "notify-send 'Starting update...'"
"""

    with open(path, 'w') as f:
        f.write(default_config)

    print(f"📝 Created default config at {path}")
