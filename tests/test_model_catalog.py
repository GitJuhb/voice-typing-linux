import importlib.util
import errno
import json
import os
import pathlib
import socket
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

import numpy as np

import streaming_stt


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "enhanced-voice-typing.py"
IBUS_ENGINE_PATH = ROOT / "ibus_voice_engine.py"


def load_main_module():
    spec = importlib.util.spec_from_file_location("enhanced_voice_typing", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    fake_pyaudio = types.ModuleType("pyaudio")
    fake_pyaudio.PyAudio = object
    fake_pyaudio.paInt16 = 8
    fake_pyaudio.paContinue = 0

    fake_webrtcvad = types.ModuleType("webrtcvad")
    fake_webrtcvad.Vad = lambda _mode=0: object()

    fake_faster_whisper = types.ModuleType("faster_whisper")
    fake_faster_whisper.WhisperModel = object

    fake_torch = types.ModuleType("torch")

    with mock.patch.dict(
        sys.modules,
        {
            "pyaudio": fake_pyaudio,
            "webrtcvad": fake_webrtcvad,
            "faster_whisper": fake_faster_whisper,
            "torch": fake_torch,
        },
    ):
        spec.loader.exec_module(module)
    return module


def load_ibus_module():
    spec = importlib.util.spec_from_file_location("ibus_voice_engine", IBUS_ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)

    fake_ibus = types.SimpleNamespace(
        Engine=type("Engine", (), {}),
        Factory=type("Factory", (), {}),
        Text=type(
            "Text",
            (),
            {"new_from_string": staticmethod(lambda text: text)},
        ),
        Capabilite=types.SimpleNamespace(SURROUNDING_TEXT=1),
        AttrType=types.SimpleNamespace(UNDERLINE=1),
        AttrUnderline=types.SimpleNamespace(SINGLE=1),
        PreeditFocusMode=types.SimpleNamespace(CLEAR=1),
        PATH_FACTORY="/fake",
    )
    fake_glib = types.SimpleNamespace(idle_add=lambda fn, *args: fn(*args))
    fake_repository = types.ModuleType("gi.repository")
    fake_repository.IBus = fake_ibus
    fake_repository.GLib = fake_glib
    fake_gi = types.ModuleType("gi")
    fake_gi.require_version = lambda *_args, **_kwargs: None
    fake_gi.repository = fake_repository

    with mock.patch.dict(
        sys.modules,
        {
            "gi": fake_gi,
            "gi.repository": fake_repository,
        },
    ):
        spec.loader.exec_module(module)
    return module


class StreamingCatalogTests(unittest.TestCase):
    def test_streaming_models_include_native_moonshine_and_zipformer(self):
        names = set(streaming_stt.get_streaming_model_names())
        self.assertIn("zipformer-en", names)
        self.assertIn("zipformer-en-20M", names)
        self.assertIn("parakeet-ctc-0.6b", names)
        self.assertIn("nemotron-asr-streaming-nim", names)
        self.assertIn("parakeet-ctc-0.6b-nim", names)
        self.assertIn("parakeet-ctc-1.1b-nim", names)
        self.assertIn("moonshine-tiny-streaming-en", names)
        self.assertIn("moonshine-small-streaming-en", names)
        self.assertIn("moonshine-medium-streaming-en", names)
        self.assertNotIn("moonshine-tiny-en-v2", names)
        self.assertNotIn("moonshine-base-en-v2", names)

    def test_offline_models_include_parakeet_and_moonshine(self):
        names = set(streaming_stt.get_offline_model_names())
        self.assertIn("parakeet-tdt-0.6b-v2", names)
        self.assertIn("moonshine-tiny-en-v2", names)
        self.assertIn("moonshine-base-en-v2", names)

    def test_streaming_backend_availability_is_model_specific(self):
        with (
            mock.patch.object(
                streaming_stt, "moonshine_voice_available", return_value=True
            ),
            mock.patch.object(
                streaming_stt, "parakeet_ctc_available", return_value=True
            ),
            mock.patch.object(
                streaming_stt, "nim_realtime_available", return_value=False
            ),
            mock.patch.object(
                streaming_stt, "sherpa_onnx_available", return_value=False
            ),
        ):
            self.assertTrue(
                streaming_stt.streaming_model_available("moonshine-medium-streaming-en")
            )
            self.assertTrue(
                streaming_stt.streaming_model_available("parakeet-ctc-0.6b")
            )
            self.assertFalse(
                streaming_stt.streaming_model_available("parakeet-ctc-1.1b-nim")
            )
            self.assertFalse(
                streaming_stt.streaming_model_available("nemotron-asr-streaming-nim")
            )
            self.assertFalse(streaming_stt.streaming_model_available("zipformer-en"))

    def test_parakeet_ctc_requires_librosa(self):
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name):
            if name == "librosa":
                return None
            return real_find_spec(name)

        with mock.patch.object(
            streaming_stt.importlib.util,
            "find_spec",
            side_effect=fake_find_spec,
        ):
            self.assertFalse(streaming_stt.parakeet_ctc_available())

    def test_parakeet_ctc_install_hint_mentions_librosa(self):
        self.assertIn(
            "librosa",
            streaming_stt.streaming_model_install_hint("parakeet-ctc-0.6b"),
        )

    def test_nim_streaming_backend_availability_requires_websocket_client(self):
        with mock.patch.object(
            streaming_stt, "nim_realtime_available", return_value=True
        ):
            self.assertTrue(
                streaming_stt.streaming_model_available("parakeet-ctc-1.1b-nim")
            )
            self.assertTrue(
                streaming_stt.streaming_model_available("nemotron-asr-streaming-nim")
            )

    def test_nim_install_hint_mentions_voice_nim_url(self):
        hint = streaming_stt.streaming_model_install_hint(
            "nemotron-asr-streaming-nim"
        )
        self.assertIn("websocket-client", hint)
        self.assertIn("VOICE_NIM_URL", hint)
        self.assertIn("Nemotron", hint)


class ParserDefaultTests(unittest.TestCase):
    def test_defaults_use_nemotron_streaming_and_optional_correction(self):
        module = load_main_module()
        self.assertEqual(
            module.CONFIG_DEFAULTS["streaming_model"], "nemotron-asr-streaming-nim"
        )
        self.assertFalse(module.CONFIG_DEFAULTS["post_commit_correction"])
        self.assertEqual(
            module.CONFIG_DEFAULTS["correction_model"], "parakeet-tdt-0.6b-v2"
        )

    def test_parser_accepts_native_moonshine_streaming_models(self):
        module = load_main_module()
        parser = module._build_parser(module._build_defaults({}))
        streaming_action = next(
            action for action in parser._actions if action.dest == "streaming_model"
        )
        self.assertIn("moonshine-medium-streaming-en", streaming_action.choices)
        self.assertIn("moonshine-small-streaming-en", streaming_action.choices)
        self.assertIn("moonshine-tiny-streaming-en", streaming_action.choices)
        self.assertIn("parakeet-ctc-0.6b", streaming_action.choices)
        self.assertIn("nemotron-asr-streaming-nim", streaming_action.choices)
        self.assertIn("parakeet-ctc-1.1b-nim", streaming_action.choices)

    def test_build_defaults_maps_legacy_refinement_keys(self):
        module = load_main_module()
        defaults = module._build_defaults(
            {"refinement": True, "refinement_model": "parakeet-tdt-0.6b-v2"}
        )
        self.assertTrue(defaults["post_commit_correction"])
        self.assertEqual(defaults["correction_model"], "parakeet-tdt-0.6b-v2")

    def test_parser_supports_post_commit_correction_and_legacy_aliases(self):
        module = load_main_module()
        parser = module._build_parser(module._build_defaults({}))

        args = parser.parse_args(
            ["--post-commit-correction", "--correction-model", "parakeet-tdt-0.6b-v2"]
        )
        self.assertTrue(args.post_commit_correction)
        self.assertEqual(args.correction_model, "parakeet-tdt-0.6b-v2")

        legacy = parser.parse_args(
            ["--refinement", "--refinement-model", "parakeet-tdt-0.6b-v2"]
        )
        self.assertTrue(legacy.post_commit_correction)
        self.assertEqual(legacy.correction_model, "parakeet-tdt-0.6b-v2")

    def test_parser_supports_output_backend_and_remote_modes(self):
        module = load_main_module()
        parser = module._build_parser(module._build_defaults({}))

        defaults = parser.parse_args([])
        self.assertEqual(defaults.output_backend, "auto")
        self.assertEqual(defaults.remote_mode, "auto")

        explicit = parser.parse_args(
            ["--output-backend", "clipboard-paste", "--remote-mode", "live-keys"]
        )
        self.assertEqual(explicit.output_backend, "clipboard-paste")
        self.assertEqual(explicit.remote_mode, "live-keys")

        shortcut = parser.parse_args(["--remote-live-keys"])
        self.assertEqual(shortcut.remote_mode, "live-keys")

        disabled = parser.parse_args(["--no-remote-auto"])
        self.assertEqual(disabled.remote_mode, "off")

    def test_env_overrides_output_backend_and_remote_mode(self):
        module = load_main_module()
        with mock.patch.dict(
            os.environ,
            {
                "VOICE_OUTPUT_BACKEND": "keys",
                "VOICE_REMOTE_MODE": "live-keys",
            },
        ):
            config = module._apply_env_overrides({})
        self.assertEqual(config["output_backend"], "keys")
        self.assertEqual(config["remote_mode"], "live-keys")


class RemoteDesktopOutputTests(unittest.TestCase):
    def test_remote_window_identity_matches_known_clients(self):
        module = load_main_module()
        self.assertTrue(
            module.VoiceTyping._is_remote_window_identity("class=RustDesk")
        )
        self.assertTrue(
            module.VoiceTyping._is_remote_window_identity("Remmina remote desktop")
        )
        self.assertTrue(
            module.VoiceTyping._is_remote_window_identity("xfreerdp /v:server")
        )
        self.assertFalse(module.VoiceTyping._is_remote_window_identity("Firefox"))

    def test_auto_backend_uses_clipboard_paste_for_remote_focus(self):
        module = load_main_module()
        vt = module.VoiceTyping.__new__(module.VoiceTyping)
        vt.output_backend = "auto"
        vt.remote_mode = "auto"
        with mock.patch.object(vt, "_focused_remote_desktop", return_value=True):
            self.assertEqual(vt._effective_output_backend(), "clipboard-paste")
            self.assertTrue(vt._defer_streaming_partials())

    def test_auto_backend_can_use_live_keys_for_remote_focus(self):
        module = load_main_module()
        vt = module.VoiceTyping.__new__(module.VoiceTyping)
        vt.output_backend = "auto"
        vt.remote_mode = "live-keys"
        with mock.patch.object(vt, "_focused_remote_desktop", return_value=True):
            self.assertEqual(vt._effective_output_backend(), "keys")
            self.assertFalse(vt._defer_streaming_partials())

    def test_auto_backend_uses_ibus_for_local_or_disabled_remote(self):
        module = load_main_module()
        vt = module.VoiceTyping.__new__(module.VoiceTyping)
        vt.output_backend = "auto"
        vt.remote_mode = "off"
        with mock.patch.object(vt, "_focused_remote_desktop", return_value=True):
            self.assertEqual(vt._effective_output_backend(), "ibus")

        vt.remote_mode = "auto"
        with mock.patch.object(vt, "_focused_remote_desktop", return_value=False):
            self.assertEqual(vt._effective_output_backend(), "ibus")

    def test_forced_output_backends_bypass_remote_detection(self):
        module = load_main_module()
        vt = module.VoiceTyping.__new__(module.VoiceTyping)
        vt.remote_mode = "auto"
        for backend in ("ibus", "keys", "clipboard-paste"):
            vt.output_backend = backend
            with mock.patch.object(vt, "_focused_remote_desktop") as focused:
                self.assertEqual(vt._effective_output_backend(), backend)
                focused.assert_not_called()


class IBusClientTests(unittest.TestCase):
    def test_is_available_rejects_stale_socket_path(self):
        module = load_main_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = pathlib.Path(temp_dir) / "voice-typing-ibus.sock"
            socket_path.touch()
            with mock.patch.object(module.IBusClient, "SOCKET_PATH", str(socket_path)):
                client = module.IBusClient()
                self.assertFalse(client.is_available)

    def test_is_available_accepts_live_socket(self):
        module = load_main_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = pathlib.Path(temp_dir) / "voice-typing-ibus.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(socket_path))
            server.listen(1)
            try:
                with mock.patch.object(
                    module.IBusClient, "SOCKET_PATH", str(socket_path)
                ):
                    client = module.IBusClient()
                    self.assertTrue(client.is_available)
                    client.close()
            finally:
                server.close()


class IBusSocketParsingTests(unittest.TestCase):
    def test_handle_client_preserves_payload_whitespace(self):
        module = load_ibus_module()
        received = []
        module._handle_socket_command = received.append
        parent, child = socket.socketpair()
        worker = threading.Thread(target=module._handle_client, args=(child,))
        worker.start()
        try:
            parent.sendall(b"commit: to \nreplace:4: now.\n")
            parent.shutdown(socket.SHUT_WR)
            worker.join(timeout=2)
        finally:
            parent.close()
            child.close()

        self.assertEqual(received, ["commit: to ", "replace:4: now."])


class StreamingFallbackBehaviorTests(unittest.TestCase):
    @staticmethod
    def _voice_typing_stub(module):
        vt = module.VoiceTyping.__new__(module.VoiceTyping)
        vt.streaming_lock = threading.Lock()
        vt.current_streaming_text = ""
        vt.visible_streaming_text = ""
        vt.ibus_client = types.SimpleNamespace(
            is_available=False,
            send_preedit=mock.Mock(return_value=False),
            send_commit=mock.Mock(return_value=False),
            supports_surrounding_text=False,
        )
        vt._type_raw = mock.Mock()
        vt._replace_typed_text = mock.Mock(return_value=0)
        vt._enqueue_transcription = mock.Mock()
        vt._log = mock.Mock()
        vt.typing_history = []
        vt.max_history = 5
        vt.previous_text = ""
        vt.refinement_enabled = False
        vt.streaming_preview_tail_words = 3
        vt.streaming_preview_max_replace_chars = 24
        vt.streaming_use_ibus_preedit = False
        return vt

    def test_no_ibus_streaming_partial_commits_stable_prefix(self):
        module = load_main_module()
        vt = self._voice_typing_stub(module)
        vt.streaming_preview_tail_words = 2

        vt._type_streaming_partial("Testing testing one two")

        self.assertEqual(vt.current_streaming_text, "testing testing one two")
        vt._type_raw.assert_called_once_with("testing testing ")
        vt._replace_typed_text.assert_not_called()
        self.assertEqual(vt.visible_streaming_text, "testing testing ")

    def test_missing_preedit_flag_defaults_to_stable_partial_mode(self):
        module = load_main_module()
        vt = self._voice_typing_stub(module)
        del vt.streaming_use_ibus_preedit
        vt.ibus_client = types.SimpleNamespace(
            is_available=True,
            supports_surrounding_text=True,
            send_preedit=mock.Mock(return_value=True),
            send_commit=mock.Mock(return_value=False),
        )
        vt.streaming_preview_tail_words = 2

        vt._type_streaming_partial("Testing testing one two")

        vt.ibus_client.send_preedit.assert_not_called()
        vt._type_raw.assert_called_once_with("testing testing ")
        vt._replace_typed_text.assert_not_called()

    def test_stable_partial_mode_only_appends_monotonic_suffix(self):
        module = load_main_module()
        vt = self._voice_typing_stub(module)
        vt.streaming_preview_tail_words = 1
        vt.visible_streaming_text = "testing "

        vt._type_streaming_partial("Testing one two")

        vt._type_raw.assert_called_once_with("one ")
        vt._replace_typed_text.assert_not_called()
        self.assertEqual(vt.visible_streaming_text, "testing one ")

    def test_stable_partial_mode_skips_non_monotonic_rewrites(self):
        module = load_main_module()
        vt = self._voice_typing_stub(module)
        vt.streaming_preview_tail_words = 1
        vt.visible_streaming_text = "testing "

        vt._type_streaming_partial("Tested one two")

        vt._type_raw.assert_not_called()
        vt._replace_typed_text.assert_not_called()
        vt._log.assert_called()
        self.assertEqual(vt.visible_streaming_text, "testing ")

    def test_ibus_surrounding_text_allows_bounded_non_monotonic_replace(self):
        module = load_main_module()
        vt = self._voice_typing_stub(module)
        vt.streaming_preview_tail_words = 1
        vt.visible_streaming_text = "testing "
        vt.ibus_client = types.SimpleNamespace(
            is_available=True,
            supports_surrounding_text=True,
            send_preedit=mock.Mock(return_value=False),
            send_commit=mock.Mock(return_value=False),
        )

        vt._type_streaming_partial("Tested one two")

        vt._replace_typed_text.assert_called_once_with("testing ", "tested one ")
        vt._type_raw.assert_not_called()
        self.assertEqual(vt.visible_streaming_text, "tested one ")

    def test_large_non_monotonic_replace_still_skips_mid_utterance(self):
        module = load_main_module()
        vt = self._voice_typing_stub(module)
        vt.streaming_preview_tail_words = 1
        vt.streaming_preview_max_replace_chars = 3
        vt.visible_streaming_text = "testing "
        vt.ibus_client = types.SimpleNamespace(
            is_available=True,
            supports_surrounding_text=True,
            send_preedit=mock.Mock(return_value=False),
            send_commit=mock.Mock(return_value=False),
        )

        vt._type_streaming_partial("Tested one two")

        vt._replace_typed_text.assert_not_called()
        vt._type_raw.assert_not_called()
        vt._log.assert_called()
        self.assertEqual(vt.visible_streaming_text, "testing ")

    def test_ibus_surrounding_text_still_uses_preedit(self):
        module = load_main_module()
        vt = self._voice_typing_stub(module)
        vt.streaming_use_ibus_preedit = True
        vt.ibus_client = types.SimpleNamespace(
            is_available=True,
            supports_surrounding_text=True,
            send_preedit=mock.Mock(return_value=True),
            send_commit=mock.Mock(return_value=False),
        )

        vt._type_streaming_partial("Testing testing one two")

        vt.ibus_client.send_preedit.assert_called_once_with("testing testing one two")
        vt._replace_typed_text.assert_not_called()
        self.assertEqual(vt.visible_streaming_text, "")

    def test_no_ibus_finalize_types_once_without_queueing_correction(self):
        module = load_main_module()
        vt = self._voice_typing_stub(module)

        text = "testing, testing, one, two, three."
        audio_buffer = [np.array([1, 2, 3], dtype=np.int16)]
        vt._finalize_streaming_utterance(
            text,
            audio_buffer,
            log_event="streaming_endpoint",
        )

        vt._type_raw.assert_called_once_with(text)
        vt._enqueue_transcription.assert_not_called()
        self.assertEqual(vt.typing_history, [(text, len(text))])
        self.assertIn(text, vt.previous_text)

    def test_finalize_replaces_visible_prefix_before_committing_full_text(self):
        module = load_main_module()
        vt = self._voice_typing_stub(module)
        vt.visible_streaming_text = "testing testing "

        vt._finalize_streaming_utterance(
            "testing testing one two three",
            [np.array([1, 2], dtype=np.int16)],
            log_event="streaming_endpoint",
        )

        vt._replace_typed_text.assert_called_once_with(
            "testing testing ", "testing testing one two three"
        )
        vt._type_raw.assert_not_called()


class MoonshineNativeStreamingTests(unittest.TestCase):
    @staticmethod
    def _fake_moonshine_module():
        module = types.ModuleType("moonshine_voice")

        class ModelArch:
            TINY_STREAMING = "tiny-streaming-arch"
            SMALL_STREAMING = "small-streaming-arch"
            MEDIUM_STREAMING = "medium-streaming-arch"

        class LineTextChanged:
            def __init__(self, text):
                self.line = types.SimpleNamespace(text=text, line_id=1)

        class LineCompleted:
            def __init__(self, text):
                self.line = types.SimpleNamespace(text=text, line_id=1)

        class Error:
            def __init__(self, error):
                self.error = error
                self.line = None

        class FakeStream:
            def __init__(self, update_interval):
                self.update_interval = update_interval
                self.listeners = []
                self.started = False
                self.closed = False
                self.audio_calls = []
                self.pending_events = []

            def add_listener(self, listener):
                self.listeners.append(listener)

            def start(self):
                self.started = True

            def add_audio(self, audio_data, sample_rate):
                self.audio_calls.append((len(audio_data), sample_rate))
                while self.pending_events:
                    event = self.pending_events.pop(0)
                    for listener in self.listeners:
                        listener(event)

            def stop(self):
                return None

            def close(self):
                self.closed = True

        class FakeTranscriber:
            created = []

            def __init__(
                self, model_path, model_arch, update_interval=0.5, options=None
            ):
                self.model_path = model_path
                self.model_arch = model_arch
                self.update_interval = update_interval
                self.options = options
                self.streams = []
                self.closed = False
                FakeTranscriber.created.append(self)

            def create_stream(self, update_interval=None):
                stream = FakeStream(update_interval)
                self.streams.append(stream)
                return stream

            def close(self):
                self.closed = True

        def string_to_model_arch(name):
            mapping = {
                "tiny-streaming": ModelArch.TINY_STREAMING,
                "small-streaming": ModelArch.SMALL_STREAMING,
                "medium-streaming": ModelArch.MEDIUM_STREAMING,
            }
            return mapping[name]

        def get_model_for_language(wanted_language="en", wanted_model_arch=None):
            return "/fake/moonshine", wanted_model_arch

        module.ModelArch = ModelArch
        module.Transcriber = FakeTranscriber
        module.LineTextChanged = LineTextChanged
        module.LineCompleted = LineCompleted
        module.Error = Error
        module.string_to_model_arch = string_to_model_arch
        module.get_model_for_language = get_model_for_language
        module.FakeTranscriber = FakeTranscriber
        return module

    def test_native_moonshine_stream_emits_partials_and_completions(self):
        fake_module = self._fake_moonshine_module()
        with mock.patch.dict(sys.modules, {"moonshine_voice": fake_module}):
            stt = streaming_stt.StreamingSTT(
                model_name="moonshine-medium-streaming-en"
            )
            stt.create_recognizer()

            stream = fake_module.FakeTranscriber.created[-1].streams[-1]
            stream.pending_events.append(fake_module.LineTextChanged("hello"))
            partial = stt.feed_chunk(np.array([0, 1000], dtype=np.int16))
            self.assertEqual(partial, "hello")

            stream.pending_events.append(fake_module.LineCompleted("hello world"))
            stt.feed_chunk(np.array([0, 1000], dtype=np.int16))
            self.assertEqual(stt.check_endpoint(), (True, "hello world"))
            self.assertEqual(stt.check_endpoint(), (False, ""))

    def test_native_moonshine_reset_recreates_stream(self):
        fake_module = self._fake_moonshine_module()
        with mock.patch.dict(sys.modules, {"moonshine_voice": fake_module}):
            stt = streaming_stt.StreamingSTT(
                model_name="moonshine-medium-streaming-en"
            )
            stt.create_recognizer()

            transcriber = fake_module.FakeTranscriber.created[-1]
            first_stream = transcriber.streams[-1]
            stt.reset()

            self.assertTrue(first_stream.closed)
            self.assertEqual(len(transcriber.streams), 2)
            self.assertTrue(transcriber.streams[-1].started)


class ParakeetCTCStreamingTests(unittest.TestCase):
    @staticmethod
    def _fake_torch_module():
        module = types.ModuleType("torch")
        module.float16 = "float16"
        module.float32 = "float32"

        class _InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        module.inference_mode = lambda: _InferenceMode()
        module.cuda = types.SimpleNamespace(is_available=lambda: False)
        return module

    @staticmethod
    def _fake_transformers_module():
        module = types.ModuleType("transformers")

        class FakeInputs(dict):
            def to(self, *_args, **_kwargs):
                return self

        class FakeProcessor:
            def __init__(self):
                self.calls = []

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def __call__(self, audio, sampling_rate, return_tensors):
                self.calls.append((len(audio), sampling_rate, return_tensors))
                return FakeInputs(input_values=audio)

            def batch_decode(self, generated_ids, skip_special_tokens=True):
                return list(generated_ids)

        class FakeModel:
            created = []

            def __init__(self):
                self.device = "cpu"
                self.dtype = "float32"
                self.outputs = []
                FakeModel.created.append(self)

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def to(self, device):
                self.device = device
                return self

            def eval(self):
                return self

            def generate(self, **_kwargs):
                if self.outputs:
                    return [self.outputs.pop(0)]
                return [""]

        module.AutoProcessor = FakeProcessor
        module.AutoModelForCTC = FakeModel
        module.FakeModel = FakeModel
        return module

    def test_buffered_parakeet_ctc_emits_partials_and_endpoint(self):
        fake_torch = self._fake_torch_module()
        fake_transformers = self._fake_transformers_module()

        with (
            mock.patch.dict(
                sys.modules,
                {"torch": fake_torch, "transformers": fake_transformers},
            ),
            mock.patch.object(
                streaming_stt, "parakeet_ctc_available", return_value=True
            ),
        ):
            stt = streaming_stt.StreamingSTT(model_name="parakeet-ctc-0.6b")
            stt.model_config["update_interval_seconds"] = 0.02
            stt.model_config["min_decode_seconds"] = 0.02
            stt.model_config["endpoint_silence_seconds"] = 0.04
            stt.create_recognizer()

            fake_model = fake_transformers.FakeModel.created[-1]
            fake_model.outputs = ["hello", "hello world", "hello world"]

            chunk = np.ones(320, dtype=np.int16)
            partial = stt.feed_chunk(chunk, is_speech=True)
            self.assertEqual(partial, "hello")

            stt.feed_chunk(chunk, is_speech=False)
            stt.feed_chunk(chunk, is_speech=False)
            self.assertEqual(stt.check_endpoint(), (True, "hello world"))
            self.assertEqual(stt.check_endpoint(), (False, ""))


class NimRealtimeStreamingTests(unittest.TestCase):
    @staticmethod
    def _fake_websocket_module():
        module = types.ModuleType("websocket")

        class WebSocketTimeoutException(Exception):
            pass

        class FakeConnection:
            def __init__(self, incoming_events):
                self.incoming = list(incoming_events)
                self.sent = []
                self.closed = False
                self.timeout = None

            def send(self, payload):
                self.sent.append(payload)

            def recv(self):
                if self.incoming:
                    event = self.incoming.pop(0)
                    if isinstance(event, BaseException):
                        raise event
                    return event
                raise WebSocketTimeoutException()

            def settimeout(self, timeout):
                self.timeout = timeout

            def close(self):
                self.closed = True

        module.WebSocketTimeoutException = WebSocketTimeoutException
        module.created = []

        def create_connection(_url, header=None, timeout=None, sslopt=None):
            conn = FakeConnection(module.pending_events)
            conn.header = header or []
            conn.timeout = timeout
            conn.sslopt = sslopt
            module.created.append(conn)
            return conn

        module.create_connection = create_connection
        module.pending_events = []
        return module

    @staticmethod
    def _fake_http_response(payload: dict):
        class FakeHTTPResponse:
            def __init__(self, body):
                self.body = json.dumps(body).encode("utf-8")

            def read(self):
                return self.body

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return FakeHTTPResponse(payload)

    def test_nim_realtime_session_updates_and_streams_partials(self):
        fake_websocket = self._fake_websocket_module()
        fake_websocket.pending_events = [
            json.dumps({"type": "conversation.created"}),
            json.dumps(
                {
                    "type": "transcription_session.updated",
                    "session": {
                        "input_audio_transcription": {"model": "placeholder"},
                        "input_audio_params": {},
                        "recognition_config": {},
                    },
                }
            ),
        ]
        fake_session = {
            "input_audio_transcription": {},
            "input_audio_params": {},
            "recognition_config": {},
        }

        with (
            mock.patch.dict(sys.modules, {"websocket": fake_websocket}),
            mock.patch.object(
                streaming_stt, "nim_realtime_available", return_value=True
            ),
            mock.patch.object(
                streaming_stt.urllib.request,
                "urlopen",
                return_value=self._fake_http_response(fake_session),
            ),
        ):
            stt = streaming_stt.StreamingSTT(model_name="parakeet-ctc-1.1b-nim")
            stt.create_recognizer()

            conn = fake_websocket.created[-1]
            update = json.loads(conn.sent[0])
            self.assertEqual(update["type"], "transcription_session.update")
            self.assertEqual(
                update["session"]["input_audio_transcription"]["model"],
                "parakeet-1-1b-ctc-en-us",
            )
            self.assertEqual(
                update["session"]["input_audio_params"]["sample_rate_hz"],
                16000,
            )
            self.assertTrue(
                update["session"]["recognition_config"][
                    "enable_automatic_punctuation"
                ]
            )

            conn.incoming.append(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "delta": "testing one",
                    }
                )
            )

            partial = stt.feed_chunk(np.array([0, 1000], dtype=np.int16))
            self.assertEqual(partial, "testing one")

            conn.incoming.append(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": "testing one two",
                        "is_last_result": False,
                    }
                )
            )
            self.assertEqual(stt.check_endpoint(), (True, "testing one two"))
            self.assertEqual(stt.check_endpoint(), (False, ""))

            sent_types = [json.loads(payload)["type"] for payload in conn.sent]
            self.assertIn("input_audio_buffer.append", sent_types)
            self.assertIn("input_audio_buffer.commit", sent_types)
            self.assertNotIn("input_audio_buffer.clear", sent_types)

            sent_count_before_reset = len(conn.sent)
            stt.reset()
            self.assertEqual(len(conn.sent), sent_count_before_reset)
            stt.close()
            self.assertTrue(conn.closed)

    def test_nim_realtime_session_can_use_server_default_model(self):
        fake_websocket = self._fake_websocket_module()
        fake_websocket.pending_events = [
            json.dumps({"type": "conversation.created"}),
            json.dumps(
                {
                    "type": "transcription_session.updated",
                    "session": {
                        "input_audio_transcription": {"model": "conformer"},
                        "input_audio_params": {},
                        "recognition_config": {},
                    },
                }
            ),
        ]
        fake_session = {
            "input_audio_transcription": {"model": "conformer"},
            "input_audio_params": {},
            "recognition_config": {},
        }

        with (
            mock.patch.dict(sys.modules, {"websocket": fake_websocket}),
            mock.patch.object(
                streaming_stt, "nim_realtime_available", return_value=True
            ),
            mock.patch.object(
                streaming_stt.urllib.request,
                "urlopen",
                return_value=self._fake_http_response(fake_session),
            ),
        ):
            stt = streaming_stt.StreamingSTT(model_name="nemotron-asr-streaming-nim")
            stt.create_recognizer()

            conn = fake_websocket.created[-1]
            update = json.loads(conn.sent[0])
            self.assertEqual(update["type"], "transcription_session.update")
            self.assertEqual(
                update["session"]["input_audio_transcription"]["model"],
                "conformer",
            )
            self.assertEqual(
                update["session"]["input_audio_params"]["sample_rate_hz"],
                16000,
            )
            self.assertTrue(
                update["session"]["recognition_config"][
                    "enable_automatic_punctuation"
                ]
            )

    def test_nim_nonblocking_receive_treats_eagain_as_no_event(self):
        fake_websocket = self._fake_websocket_module()
        fake_websocket.pending_events = [
            json.dumps({"type": "conversation.created"}),
            json.dumps(
                {
                    "type": "transcription_session.updated",
                    "session": {
                        "input_audio_transcription": {"model": "conformer"},
                        "input_audio_params": {},
                        "recognition_config": {},
                    },
                }
            ),
        ]
        fake_session = {
            "input_audio_transcription": {"model": "conformer"},
            "input_audio_params": {},
            "recognition_config": {},
        }

        with (
            mock.patch.dict(sys.modules, {"websocket": fake_websocket}),
            mock.patch.object(
                streaming_stt, "nim_realtime_available", return_value=True
            ),
            mock.patch.object(
                streaming_stt.urllib.request,
                "urlopen",
                return_value=self._fake_http_response(fake_session),
            ),
        ):
            stt = streaming_stt.StreamingSTT(model_name="nemotron-asr-streaming-nim")
            stt.create_recognizer()

            conn = fake_websocket.created[-1]
            conn.incoming.append(BlockingIOError(errno.EAGAIN, "try again"))

            partial = stt.feed_chunk(np.array([0, 1000], dtype=np.int16))
            self.assertEqual(partial, "")
            self.assertIsNone(stt.stream_error)


class OfflineSherpaTests(unittest.TestCase):
    def test_offline_transcribe_reads_decoded_text_from_stream_result(self):
        class FakeResult:
            def __init__(self, text):
                self.text = text

        class FakeStream:
            def __init__(self):
                self.result = FakeResult("refined text")
                self.accepted = None
                self.finished = False

            def accept_waveform(self, sample_rate, samples):
                self.accepted = (sample_rate, samples)

            def input_finished(self):
                self.finished = True

        class FakeRecognizer:
            def __init__(self):
                self.stream = FakeStream()
                self.decoded_stream = None

            def create_stream(self):
                return self.stream

            def decode_stream(self, stream):
                self.decoded_stream = stream

        stt = streaming_stt.OfflineSTT(model_name="parakeet-tdt-0.6b-v2")
        stt.recognizer = FakeRecognizer()

        text = stt.transcribe(np.array([0, 32767], dtype=np.int16))

        self.assertEqual(text, "refined text")
        self.assertIs(stt.recognizer.decoded_stream, stt.recognizer.stream)
        self.assertEqual(stt.recognizer.stream.accepted[0], stt.sample_rate)
        self.assertTrue(stt.recognizer.stream.finished)


class PostCommitCorrectionTests(unittest.TestCase):
    def test_replace_typed_text_prefers_ibus_replace(self):
        module = load_main_module()
        replace_calls = []

        vt = module.VoiceTyping.__new__(module.VoiceTyping)
        vt.ibus_client = types.SimpleNamespace(
            is_available=True,
            supports_surrounding_text=True,
            send_replace=lambda count, text: replace_calls.append((count, text))
            or True,
        )
        vt.key_injector = None
        vt._send_backspaces = lambda _count: self.fail("unexpected backspace path")
        vt._type_raw = lambda _text: self.fail("unexpected raw typing path")

        chars_to_delete = vt._replace_typed_text("hello world", "hello world!")

        self.assertEqual(chars_to_delete, 0)
        self.assertEqual(replace_calls, [(0, "!")])


if __name__ == "__main__":
    unittest.main()
