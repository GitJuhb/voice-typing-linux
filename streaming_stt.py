"""
Streaming speech-to-text using sherpa-onnx OnlineRecognizer.

Wraps a sherpa-onnx transducer model (streaming zipformer) to provide
real-time partial transcription results as audio chunks arrive.

Thread safety: NOT thread-safe. All methods must be called from a single
thread (the streaming worker thread). No locking is needed since the
OnlineRecognizer and stream are single-consumer.
"""

import os
import sys
import tarfile
import urllib.request
import numpy as np

# Model configurations: name -> (repo, tarball, subdir with model files)
MODELS = {
    "zipformer-en": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        "dir": "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "encoder": "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "decoder": "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "joiner": "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
        "tokens": "tokens.txt",
        "size_mb": 80,
    },
    "zipformer-en-20M": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2",
        "dir": "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        "encoder": "encoder-epoch-99-avg-1.onnx",
        "decoder": "decoder-epoch-99-avg-1.onnx",
        "joiner": "joiner-epoch-99-avg-1.onnx",
        "tokens": "tokens.txt",
        "size_mb": 20,
    },
}

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/sherpa-onnx")


class StreamingSTT:
    """Streaming speech-to-text using sherpa-onnx OnlineRecognizer."""

    def __init__(
        self,
        model_name: str = "zipformer-en",
        cache_dir: str = DEFAULT_CACHE_DIR,
        sample_rate: int = 16000,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.sample_rate = sample_rate
        self.recognizer = None
        self.stream = None

        if model_name not in MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}"
            )

        self.model_config = MODELS[model_name]

    @classmethod
    def download_model(cls, model_name: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
        """Download streaming model if not already cached.

        Returns the path to the model directory.
        """
        if model_name not in MODELS:
            raise ValueError(f"Unknown model '{model_name}'")

        config = MODELS[model_name]
        model_dir = os.path.join(cache_dir, config["dir"])

        # Check if already downloaded (look for tokens.txt as sentinel)
        tokens_path = os.path.join(model_dir, config["tokens"])
        if os.path.exists(tokens_path):
            return model_dir

        os.makedirs(cache_dir, exist_ok=True)
        url = config["url"]
        tarball_path = os.path.join(cache_dir, os.path.basename(url))

        print(f"Downloading streaming model '{model_name}' (~{config['size_mb']}MB)...")
        print(f"  URL: {url}")

        def _progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(
                    f"\r  Progress: {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)"
                )
                sys.stdout.flush()

        try:
            urllib.request.urlretrieve(url, tarball_path, reporthook=_progress_hook)
            print()  # newline after progress
        except Exception as e:
            # Clean up partial download
            if os.path.exists(tarball_path):
                os.remove(tarball_path)
            raise RuntimeError(f"Failed to download model: {e}") from e

        # Extract
        print(f"  Extracting to {cache_dir}...")
        try:
            with tarfile.open(tarball_path, "r:bz2") as tar:
                tar.extractall(path=cache_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to extract model: {e}") from e
        finally:
            # Clean up tarball
            if os.path.exists(tarball_path):
                os.remove(tarball_path)

        if not os.path.exists(tokens_path):
            raise RuntimeError(
                f"Model extraction succeeded but {config['tokens']} not found in {model_dir}"
            )

        print(f"  Model ready: {model_dir}")
        return model_dir

    def create_recognizer(self):
        """Initialize the sherpa-onnx OnlineRecognizer with the downloaded model."""
        import sherpa_onnx

        model_dir = self.download_model(self.model_name, self.cache_dir)
        config = self.model_config

        encoder_path = os.path.join(model_dir, config["encoder"])
        decoder_path = os.path.join(model_dir, config["decoder"])
        joiner_path = os.path.join(model_dir, config["joiner"])
        tokens_path = os.path.join(model_dir, config["tokens"])

        # Verify files exist
        for path, label in [
            (encoder_path, "encoder"),
            (decoder_path, "decoder"),
            (joiner_path, "joiner"),
            (tokens_path, "tokens"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {label} at {path}")

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            tokens=tokens_path,
            num_threads=2,
            sample_rate=self.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            # Endpoint detection rules
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,  # No speech detected for 2.4s
            rule2_min_trailing_silence=1.2,  # Speech detected, then 1.2s silence
            rule3_min_utterance_length=20.0,  # Force endpoint after 20s
        )

        self.stream = self.recognizer.create_stream()
        print(f"Streaming recognizer initialized ({self.model_name})")

    def feed_chunk(self, chunk: np.ndarray) -> str:
        """Feed an int16 audio chunk and return the current partial text.

        Args:
            chunk: numpy int16 audio data (single channel, 16kHz)

        Returns:
            Current partial transcription text
        """
        if self.recognizer is None or self.stream is None:
            return ""

        # Convert int16 to float32 normalized to [-1, 1]
        samples = chunk.astype(np.float32) / 32768.0

        self.stream.accept_waveform(self.sample_rate, samples)

        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        result = self.recognizer.get_result(self.stream)
        return self._extract_text(result)

    @staticmethod
    def _extract_text(result) -> str:
        """Extract text from result (handles both str and object with .text)."""
        if isinstance(result, str):
            return result.strip()
        if hasattr(result, "text"):
            return result.text.strip() if result.text else ""
        return str(result).strip() if result else ""

    def check_endpoint(self) -> tuple[bool, str]:
        """Check if an endpoint (end of utterance) was detected.

        Returns:
            (is_endpoint, final_text) - if is_endpoint is True, final_text
            contains the complete utterance and the stream has been reset.
        """
        if self.recognizer is None or self.stream is None:
            return False, ""

        if self.recognizer.is_endpoint(self.stream):
            result = self.recognizer.get_result(self.stream)
            final_text = self._extract_text(result)
            self.recognizer.reset(self.stream)
            return True, final_text

        return False, ""

    def reset(self):
        """Reset the stream for a fresh utterance."""
        if self.recognizer is not None:
            self.stream = self.recognizer.create_stream()
