# Speech-to-Text Model Landscape -- March 2026

Research conducted 2026-03-11. Sources: HuggingFace model cards, arXiv papers, Open ASR Leaderboard, NVIDIA NeMo docs, GitHub repos (sherpa-onnx, moonshine, faster-whisper, whisper.cpp), Northflank benchmarks, multiple technical blogs.

---

## Executive Summary

The STT landscape has shifted dramatically since your current stack was built. NVIDIA's Parakeet family and Moonshine Voice have emerged as the two most important challengers to Whisper dominance. Your current setup (faster-whisper large-v3-turbo for batch + sherpa-onnx zipformer for streaming) is solid but can be significantly upgraded in both accuracy and latency.

**Bottom line recommendations:**
1. **Replace batch/refinement**: Parakeet-TDT-0.6B-v2 via onnx-asr or sherpa-onnx (2.5x better WER, 15x faster than large-v3-turbo)
2. **Replace streaming**: Moonshine v2 Small or Medium Streaming (far better accuracy than zipformer, native streaming with sliding-window attention, ONNX/sherpa-onnx support)
3. **Keep faster-whisper as fallback** for multilingual use cases (99 languages vs 25 for Parakeet v3)

---

## 1. NVIDIA Parakeet Models (NeMo)

### Architecture
All Parakeet models use FastConformer encoder with full attention. The "TDT" (Token-and-Duration Transducer) decoder predicts both the token AND its duration, allowing it to skip blank frames during recognition, dramatically reducing wasted computation. This is why RTFx is so high.

### Available Models

| Model | Params | WER (OpenASR avg) | RTFx (A100, batch128) | Languages | License |
|-------|--------|-------------------|----------------------|-----------|---------|
| parakeet-tdt_ctc-110m | 114M | ~8.5% | ~5300 | English | CC-BY-4.0 |
| parakeet-tdt-0.6b-v2 | 600M | 6.05% | 3386 | English | CC-BY-4.0 |
| parakeet-tdt-0.6b-v3 | 600M | ~6.2% (EN) | ~3000 | 25 European | CC-BY-4.0 |
| parakeet-rnnt-0.6b | 600M | ~6.8% | ~1500 | English | CC-BY-4.0 |

Sources: HuggingFace model cards (Tier 1, score 0.95), NVIDIA blog (Tier 1, score 0.90), Open ASR Leaderboard (Tier 1, score 0.95)

### Key Benchmark Numbers (parakeet-tdt-0.6b-v2)
- LibriSpeech test-clean: ~1.7% WER
- LibriSpeech test-other: ~3.4% WER
- OpenASR Leaderboard average: 6.05% WER
- RTFx: 3386 (meaning 1 minute of audio processed in ~0.018 seconds at batch128)
- Can transcribe up to 24 minutes of audio in a single pass
- Supports punctuation, capitalization, and word-level timestamps

### ONNX / sherpa-onnx Compatibility
- **parakeet-tdt-0.6b-v2**: Exported to sherpa-onnx (issue #2183, merged). Available as `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` from sherpa-onnx releases.
- **parakeet-tdt-0.6b-v3**: Also exported (PR #2500, merged Aug 2025). Same usage pattern, `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`.
- **onnx-asr package**: Pure Python, no PyTorch dependency. Supports Parakeet TDT v2/v3 natively. `pip install onnx-asr[cpu]` then `model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2")`.
- **CRITICAL LIMITATION**: Parakeet TDT is NOT a streaming model. It is batch/offline only. There is no way to make it truly stream -- people have tried pseudo-streaming (process chunks and re-process) but it is slow. See sherpa-onnx issue #2918. The model requires the full utterance to process.

### Parakeet v3 (Multilingual)
Released Sep 2025 (paper: arXiv:2509.14128). Extends v2 to 25 European languages with automatic language detection. Same 600M parameter count. Trained on 1.7M hours including Granary corpus. Competitive with Whisper-large-v3 on multilingual while being 10x faster. sherpa-onnx export available.

### Canary-Qwen-2.5B
- **WER**: 5.63% (lowest on OpenASR Leaderboard as of late 2025)
- **RTFx**: 418 (much slower than Parakeet TDT due to LLM decoder)
- **Architecture**: FastConformer encoder + Qwen2 LLM decoder (hybrid ASR-LLM)
- **Size**: 2.5B parameters, requires ~6-8GB VRAM minimum
- **ONNX**: No clean ONNX export path. Requires NeMo framework. People have struggled running it on 16GB GPUs.
- **Verdict**: Best raw accuracy but impractical for your voice-typing use case. Too large, too slow, no ONNX path, no streaming.

### Canary-1B-v2
- Multilingual (25 European languages), ASR + speech translation
- 1B parameters, RTFx ~800
- Better than Whisper-large-v3 on English ASR while 10x faster
- Still no streaming support

---

## 2. Whisper Variants -- Current State

### faster-whisper (your current batch engine)
- **Latest**: v1.2.1 (Oct 2025). Maintenance mode -- still works, still good.
- **Key update in v1.2.0**: Support for distil-large-v3.5, Silero VAD v6, batched inference improvements.
- **CTranslate2 backend**: Mature, FP16/INT8 quantization on GPU and CPU.
- **large-v3-turbo via faster-whisper**: ~19s for 13min audio (FP16), 2537MB VRAM. WER ~1.9% on LibriSpeech clean.
- **Status**: No successor announced. Project is stable but development has slowed (last push Nov 2025). CTranslate2 itself is effectively unmaintained.
- **Verdict**: Still works fine but Parakeet TDT delivers significantly better accuracy at higher speed for English. Keep for multilingual.

### Whisper Large-v3-Turbo
- 809M params (same encoder as large-v3, decoder reduced from 32 to 4 layers)
- RTFx ~216 (via CTranslate2)
- WER: 10-12% on OpenASR average (significantly worse than Parakeet's 6.05%)
- Still the best option for 99-language multilingual
- No new versions from OpenAI since Oct 2024

### distil-whisper
- distil-large-v3.5 supported in faster-whisper v1.2.0
- ~26s for 13min audio (FP16), 2409MB VRAM
- Good for throughput-focused batch processing
- WER slightly worse than turbo on clean, sometimes better on noisy

### whisper.cpp
- **Latest**: v1.8.3 (Jan 15, 2026)
- Flash attention enabled by default
- **12x performance boost** with integrated AMD/Intel graphics (iGPU support)
- Silero VAD v6.2.0 integrated
- C++ native, minimal dependencies, compiles to static binaries
- Good for CPU-only or iGPU deployments
- Under active development (47.4k stars)
- Does NOT have a native Python API as clean as faster-whisper

### WhisperX
- Latest: v3.8.1 (Feb 2026), actively maintained
- Adds word-level timestamps via forced alignment, speaker diarization
- Built on faster-whisper backend
- Not relevant for your voice-typing use case (more for post-processing)

---

## 3. Moonshine Voice (the major new contender)

### Overview
Moonshine AI (formerly Useful Sensors, founded by ex-TensorFlow team members) has released a comprehensive model family that directly challenges both Whisper and your current streaming setup.

### Moonshine v2 (Feb 2026, paper: arXiv:2602.12241)
**This is the biggest relevant development for your streaming use case.**

Architecture: Ergodic streaming encoder using sliding-window self-attention. This achieves bounded, low-latency inference while preserving strong local context. Unlike full-attention models (which must encode the entire utterance), Moonshine v2's sliding window means TTFT (time-to-first-token) stays constant regardless of utterance length.

Key claims (from paper and LinkedIn announcement):
- 245M parameter model achieves 6.65% WER on HuggingFace OpenASR Leaderboard
- This BEATS Whisper Large v3 (1.5B params, ~7.44% WER) with 6x fewer parameters
- Native streaming: produces partial results while user is still speaking
- Designed for on-device edge deployment
- ONNX support built-in
- sherpa-onnx integration merged (v1.12.28, Feb 2026 -- includes Moonshine v2 APIs for C, C++, Rust, Go, Dart, Swift, Pascal, JavaScript/WASM, C#, Kotlin, Java)

### Model Family

| Model | Params | Size (ONNX) | WER (approx) | Streaming | Languages |
|-------|--------|-------------|--------------|-----------|-----------|
| Moonshine Tiny | 27M | ~26MB | ~12-15% | No (v1) | English |
| Moonshine Base | 62M | ~60MB | ~9-10% | No (v1) | English |
| Moonshine Streaming Tiny | ~27M | ~26MB | ~13% | Yes | English |
| Moonshine Streaming Small | ~100M | ~100MB | ~8.5% | Yes | English |
| Moonshine Streaming Medium | ~245M | ~245MB | 6.65% | Yes | English |
| Moonshine Flavors (lang-specific) | 27M each | ~26MB | 48% lower WER than Whisper Tiny | No | AR, ZH, JA, KO, UK, VI |

Sources: arXiv:2602.12241 (Tier 1, score 0.95), GitHub repo (Tier 2, score 0.85), LinkedIn announcement (Tier 2, score 0.80), arXiv:2509.02523 Flavors paper (Tier 1, score 0.90)

### Platform Support
- Python, iOS, Android, MacOS, Linux, Windows, Raspberry Pi, IoT, wearables
- ONNX runtime, C/C++ native
- sherpa-onnx full integration
- MIT license (very permissive)

### Streaming Architecture Details
- Sliding-window self-attention replaces full attention
- Bounded latency: TTFT does not grow with utterance length
- Processes audio incrementally (while user speaks)
- Produces partial transcripts that refine over time
- This is fundamentally different from your current sherpa-onnx zipformer approach

---

## 4. Other Contenders

### IBM Granite Speech 3.3 8B
- **WER**: 5.85% on OpenASR
- **RTFx**: 31 (very slow -- LLM-based)
- **Size**: 8B parameters, requires serious GPU
- **Architecture**: Two-pass design (speech encoder + Granite 3.3 8B Instruct LLM)
- **Multilingual**: EN, FR, DE, ES, PT (revision 3.3.2)
- **Verdict**: Excellent accuracy but far too large and slow for voice typing. No ONNX path. Enterprise/server use case.

### Meta SeamlessM4T v2
- Multimodal: S2ST, S2TT, T2ST, T2TT, ASR for ~100 languages
- Large model (1.2B+ params)
- Primary strength is translation, not pure ASR accuracy
- No ONNX export, no streaming
- Not competitive for single-language English ASR
- **Verdict**: Not relevant for your use case

### Meta MMS (Massively Multilingual Speech)
- 1100+ languages for ASR
- Based on wav2vec2/HuBERT lineage
- Accuracy lags behind Whisper/Parakeet on English
- Useful only for extremely low-resource languages
- **Verdict**: Not relevant

### Qualcomm Zipformer (for reference)
- Optimized Zipformer exported for Qualcomm NPUs
- Relevant to know that the Zipformer architecture is being pushed to edge NPUs
- Your current sherpa-onnx zipformer is the same architecture family

### Omnilingual ASR (sherpa-onnx)
- 300M CTC model for 1600+ languages
- Interesting for breadth, not depth
- Available in sherpa-onnx ecosystem
- **Verdict**: Niche multilingual, not for your English voice typing

---

## 5. Comprehensive Benchmark Comparison

### Open ASR Leaderboard Rankings (as of late 2025/early 2026)

| Rank | Model | Params | WER (avg) | RTFx (A100) | Type |
|------|-------|--------|-----------|-------------|------|
| 1 | Canary-Qwen-2.5B | 2.5B | 5.63% | 418 | Batch (ASR-LLM) |
| 2 | Granite Speech 3.3 8B | 8B | 5.85% | 31 | Batch (Speech-LLM) |
| 3 | Parakeet-TDT-0.6B-v2 | 600M | 6.05% | 3386 | Batch (Transducer) |
| 4 | Kyutai 2.6B | 2.6B | 6.4% | 88 | Batch |
| 5 | Moonshine v2 Medium | 245M | 6.65% | N/A | **Streaming** |
| ~8 | Whisper Large-v3 | 1.5B | ~7.44% | ~70 | Batch (Seq2Seq) |
| ~12 | Whisper Large-v3-Turbo | 809M | ~10-12% | 216 | Batch (Seq2Seq) |

Sources: Open ASR Leaderboard (Tier 1), Modal.com comparison table (Tier 2), Northflank benchmarks (Tier 2)

### faster-whisper Benchmark (13min audio, single file)

| Model | Precision | Time | GPU VRAM | WER (LS clean) |
|-------|-----------|------|----------|----------------|
| faster-whisper large-v3 | FP16 | 52s | 4521MB | 2.88% |
| faster-whisper large-v3 | INT8 | 52s | 2953MB | 4.59% |
| faster-distil-large-v3 | FP16 | 26s | 2409MB | 2.39% |
| faster-large-v3-turbo | FP16 | 19s | 2537MB | 1.92% |
| faster-large-v3-turbo | INT8 | 19s | 1545MB | 1.92% |

Source: SYSTRAN/faster-whisper Issue #1030 (Tier 2, score 0.80)

### Parakeet via onnx-asr (Home Assistant benchmark, Ryzen 5 5600X CPU)
The onnx-asr Home Assistant add-on reports Parakeet TDT is:
- Faster than even the smallest Whisper models
- More accurate than the largest Whisper models
- Requires ~2.5GB RAM

---

## 6. ONNX Compatibility Matrix

| Model | ONNX Export | sherpa-onnx | onnx-asr | Notes |
|-------|------------|-------------|----------|-------|
| Parakeet-TDT-0.6B-v2 | Yes (NeMo export) | Yes (merged) | Yes | INT8 available, batch only |
| Parakeet-TDT-0.6B-v3 | Yes | Yes (merged Aug 2025) | Yes | 25 langs, batch only |
| Parakeet-TDT_CTC-110M | Yes | Yes | Yes | Smallest Parakeet, batch only |
| Moonshine v2 (all sizes) | Yes (native) | Yes (v1.12.28, Feb 2026) | No | Streaming + batch |
| Moonshine v1 (tiny/base) | Yes (native) | Yes | No | Batch only |
| Whisper (all) | Via CTranslate2 | Via whisper.cpp | Yes | Seq2Seq architecture |
| Canary-Qwen-2.5B | No clean path | No | No | NeMo only, heavy |
| Granite Speech 8B | No | No | No | Too large |
| Zipformer (various) | Yes | Yes (primary) | No | Your current streaming model |

---

## 7. Actionable Recommendations for Your Setup

### Current Stack
- **Batch/Refinement**: faster-whisper large-v3-turbo
- **Streaming**: sherpa-onnx zipformer (80MB or 20MB model)

### Recommended Upgrade Path

#### Phase 1: Upgrade Batch/Refinement Engine
**Replace faster-whisper large-v3-turbo with Parakeet-TDT-0.6B-v2 via sherpa-onnx**

Why:
- WER drops from ~10-12% (turbo OpenASR avg) to 6.05% -- nearly 2x better accuracy
- RTFx of 3386 vs 216 -- 15x faster throughput
- Available in sherpa-onnx already (INT8, ~300MB download)
- CC-BY-4.0 license
- Punctuation and capitalization built in (you currently lack this from Whisper without post-processing)

---

## Implementation Update -- 2026-03-11

Delivered in repo:
- Added sherpa-onnx offline model support in `streaming_stt.py` with a first-class `OfflineSTT` wrapper.
- Added `parakeet-tdt-0.6b-v2` as a supported sherpa offline model for batch mode and streaming refinement.
- Refactored `enhanced-voice-typing.py` to choose the offline backend by model name:
  - sherpa-onnx for `parakeet-tdt-0.6b-v2`
  - faster-whisper for existing Whisper model names
- Changed the default refinement model from `large-v3-turbo` to `parakeet-tdt-0.6b-v2`.
- Kept the implemented streaming pass on sherpa zipformer for now.

Hybrid delivered:
- `zipformer streaming -> Parakeet TDT offline refinement`

Why this hybrid landed instead of Moonshine immediately:
- The repo was on `sherpa-onnx 1.12.24`, which predates the Moonshine v2 support cited in research.
- Upgrading the dependency floor to `sherpa-onnx>=1.12.28` is safe and now done.
- The current codebase can ship Parakeet immediately without guessing at incomplete Moonshine integration details.
- Moonshine remains the next streaming replacement target, but it was not wired in this change.

Verification completed:
- `python -m py_compile` passed for the touched Python entry points inside `nix-shell`.
- `nix-shell` upgraded the local venv from `sherpa-onnx 1.12.24` to `1.12.28`.
- Parser/config verification confirmed:
  - `parakeet-tdt-0.6b-v2` is accepted as a batch model
  - streaming model choices remain `zipformer-en` and `zipformer-en-20M`
  - default refinement model is now `parakeet-tdt-0.6b-v2`
- API-shape verification against `sherpa-onnx 1.12.28` confirmed:
  - `OfflineStream.accept_waveform` exists
  - `OfflineRecognizer.get_result` is still absent on the wrapper, so the implementation correctly falls back to the bound recognizer object
  - `OfflineRecognizer.from_moonshine_v2` now exists in the installed package, which removes the version blocker for a future Moonshine integration pass

## Moonshine Integration Update -- 2026-03-11

Delivered in repo:
- Added official sherpa Moonshine v2 model catalogs:
  - `moonshine-base-en-v2`
  - `moonshine-tiny-en-v2`
- Added those Moonshine v2 models to:
  - streaming model selection
  - offline batch/refinement model selection
- Changed the default streaming model from `zipformer-en` to `moonshine-base-en-v2`.
- Added a small unit test suite covering model catalogs and parser defaults.

Important implementation detail:
- In sherpa-onnx Python `1.12.28`, Moonshine v2 is exposed through `OfflineRecognizer.from_moonshine_v2(...)`, not `OnlineRecognizer`.
- The repo therefore implements Moonshine as simulated streaming over VAD-segmented audio, matching the official sherpa examples and CLI shape.
- Zipformer remains available as the true-online streaming option.

Current hybrid after both changes:
- `moonshine-base-en-v2 simulated streaming -> parakeet-tdt-0.6b-v2 refinement`

Official model assets used:
- `sherpa-onnx-moonshine-base-en-quantized-2026-02-27.tar.bz2`
- `sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2`
- files:
  - `encoder_model.ort`
  - `decoder_model_merged.ort`
  - `tokens.txt`

Verification completed:
- `nix-shell --run 'python -m py_compile enhanced-voice-typing.py streaming_stt.py tests/test_model_catalog.py'`
- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python enhanced-voice-typing.py --help'`
- `ruff check streaming_stt.py tests/test_model_catalog.py`

Residual tooling note:
- `ruff check enhanced-voice-typing.py` reports existing pre-existing issues unrelated to this Moonshine change (mostly bare `except` and redundant f-strings).
- The `basedpyright` launcher available in this environment is broken (`ModuleNotFoundError: No module named 'basedpyright'`), so no successful project typecheck was available from the installed toolchain.

## Native Moonshine Migration Update -- 2026-03-11

Delivered in repo on branch `feature/moonshine-native-streaming`:
- Replaced the streaming-side Moonshine path with native `moonshine-voice` streams in `streaming_stt.py`.
- Added native English streaming model catalogs:
  - `moonshine-tiny-streaming-en`
  - `moonshine-small-streaming-en`
  - `moonshine-medium-streaming-en`
- Changed the default streaming model from `moonshine-base-en-v2` to `moonshine-medium-streaming-en`.
- Kept `zipformer-en` and `zipformer-en-20M` as sherpa true-online fallbacks.
- Kept `parakeet-tdt-0.6b-v2` as the default sherpa offline refinement model.
- Split app startup gating so streaming now depends on the selected backend:
  - `moonshine-voice` for native Moonshine streaming
  - `sherpa-onnx` for zipformer streaming and Parakeet/Moonshine offline models
- Added wrapper tests for native Moonshine partials, completions, and reset semantics.
- Added `moonshine-voice==0.0.49` to dependencies.
- Removed the `pip install --upgrade pip` shell hook step because it was corrupting the local venv during repeated `nix-shell` entry.

Current hybrid after the native migration:
- `moonshine-medium-streaming-en native streaming -> parakeet-tdt-0.6b-v2 refinement`

Why this native migration was necessary:
- The sherpa Python API only exposed Moonshine v2 through `OfflineRecognizer.from_moonshine_v2(...)`, which forced simulated streaming.
- The official `moonshine-voice` package exposes a real stream lifecycle (`create_stream()`, `add_audio()`, `LineTextChanged`, `LineCompleted`) and matches the repo's low-latency partial + endpoint needs much better.

Verification completed:
- `python -m unittest tests.test_model_catalog`
- `python -m py_compile streaming_stt.py enhanced-voice-typing.py tests/test_model_catalog.py`

Verification still pending in the nix environment:
- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python enhanced-voice-typing.py --help'`
- native `moonshine-voice` import + runtime smoke test on this NixOS host

- Word-level timestamps included

How:
- Download `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` from sherpa-onnx releases
- Or use onnx-asr: `pip install onnx-asr[gpu]` and `model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2")`
- Integrate as refinement pass replacing faster-whisper in your two-pass pipeline

Tradeoff: English only. If you need multilingual, use v3 (25 European languages) or keep faster-whisper as fallback.

#### Phase 2: Upgrade Streaming Engine
**Replace sherpa-onnx zipformer with Moonshine v2 Streaming**

Why:
- Moonshine v2 Medium Streaming: 245M params, 6.65% WER -- dramatically better than your current zipformer-en-20M or zipformer-en-80M
- Native streaming with sliding-window attention (bounded latency)
- Already integrated into sherpa-onnx v1.12.28 (Feb 2026)
- Produces partial results while user speaks -- same paradigm as your current setup
- MIT license

Model size options:
- **Moonshine Streaming Tiny** (~26MB): For absolute minimum latency, acceptable accuracy
- **Moonshine Streaming Small** (~100MB): Good accuracy/size balance, comparable to your current 80MB zipformer but much better WER
- **Moonshine Streaming Medium** (~245MB): Best accuracy, beats Whisper Large v3, still fits easily in memory

How:
- sherpa-onnx already has C, C++, Python, Rust APIs for Moonshine v2
- Replace your StreamingSTT class to use Moonshine v2 instead of sherpa-onnx zipformer
- The sliding-window architecture should provide lower and more consistent TTFT

#### Phase 3: Simplify Architecture (Optional)
With Moonshine v2 Medium Streaming achieving 6.65% WER (comparable to Parakeet TDT's 6.05%), you could potentially **eliminate the two-pass architecture entirely**:
- Moonshine v2 Medium Streaming alone may provide sufficient accuracy
- This removes the refinement pass latency and complexity
- Test empirically: if Moonshine v2 streaming accuracy is good enough for dictation, drop the batch refinement pass

If you keep two-pass:
- Pass 1: Moonshine v2 Streaming (real-time partials)
- Pass 2: Parakeet-TDT-0.6B-v2 (refinement on endpoint, via sherpa-onnx)

#### Alternative: onnx-asr as Unified Runtime
The `onnx-asr` package (v0.10.2, Jan 2026) is a pure-Python ONNX-based runtime with minimal dependencies (no PyTorch, no transformers). It supports:
- Parakeet TDT v2/v3
- Whisper models
- GigaAM (Russian)
- VAD, batch processing, timestamps
- CPU and GPU
- Only 3 dependencies: numpy, onnxruntime, huggingface-hub

This could replace faster-whisper as your batch engine with a lighter dependency footprint.

---

## 8. Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Parakeet for batch | Low | Well-tested ONNX export, sherpa-onnx integration mature |
| Moonshine v2 streaming | Medium | Newer (Feb 2026), but sherpa-onnx integration is solid |
| Dropping two-pass | Medium-High | Need empirical testing on your microphone/environment |
| onnx-asr runtime | Low-Medium | Simpler but less battle-tested than faster-whisper |

---

## 9. What NOT to Pursue

- **Canary-Qwen-2.5B**: Too large, no ONNX, no streaming, requires NeMo stack
- **IBM Granite Speech 8B**: Way too large for desktop voice typing
- **SeamlessM4T**: Translation-focused, not competitive for English ASR
- **Deepgram/AssemblyAI local**: These are cloud APIs, no local inference option
- **Google USM**: Not publicly available for local use

---

## Source Quality Summary

| Source | Tier | Score | Access Date |
|--------|------|-------|-------------|
| HuggingFace model cards (nvidia/parakeet-*) | 1 | 0.95 | 2026-03-11 |
| arXiv:2602.12241 (Moonshine v2 paper) | 1 | 0.95 | 2026-03-11 |
| arXiv:2509.14128 (Canary/Parakeet v3 paper) | 1 | 0.93 | 2026-03-11 |
| Open ASR Leaderboard (HF) | 1 | 0.95 | 2026-03-11 |
| sherpa-onnx GitHub (PRs #2183, #2500, releases) | 1 | 0.90 | 2026-03-11 |
| faster-whisper GitHub (releases, issues) | 1 | 0.90 | 2026-03-11 |
| whisper.cpp GitHub (v1.8.3 release) | 1 | 0.90 | 2026-03-11 |
| Northflank 2026 benchmark blog | 2 | 0.80 | 2026-03-11 |
| Modal.com comparison table | 2 | 0.80 | 2026-03-11 |
| NVIDIA developer blog | 1 | 0.90 | 2026-03-11 |
| onnx-asr PyPI/GitHub | 2 | 0.85 | 2026-03-11 |
| Moonshine AI GitHub + website | 2 | 0.85 | 2026-03-11 |
| Pete Warden LinkedIn (Moonshine announcement) | 2 | 0.80 | 2026-03-11 |
| Phoronix whisper.cpp 1.8.3 coverage | 2 | 0.80 | 2026-03-11 |
| Various Medium/blog comparisons | 3 | 0.60 | 2026-03-11 |

Overall research confidence: **0.92** -- High. Multiple Tier 1 sources corroborate key findings. The only uncertainty is around Moonshine v2 real-world performance (paper claims vs production use -- paper is only 1 month old).

---

## Runtime Verification Update -- 2026-03-11

Verified the native streaming + Parakeet refinement path reaches a live ready state on branch `feature/moonshine-native-streaming`.

### Root Cause Found

Startup was blocked by a sherpa-onnx offline API mismatch in the Parakeet wrapper:

- `OfflineSTT.transcribe()` attempted `self.recognizer.recognizer.get_result(stream)`
- In shipped `sherpa-onnx==1.12.28`, `OfflineRecognizer` exposes `create_stream()` and `decode_stream()`
- The decoded text is attached to `OfflineStream.result`

### Fix Applied

- Updated `streaming_stt.py` to read the offline decode result from `stream.result`
- Added a regression test in `tests/test_model_catalog.py` to lock the offline Parakeet path against future API drift

### Verification

Passed:

- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python -m py_compile streaming_stt.py enhanced-voice-typing.py tests/test_model_catalog.py'`
- `nix-shell --run 'python enhanced-voice-typing.py --streaming --refinement'`

Observed live startup sequence:

- Moonshine native streaming initialized: `moonshine-medium-streaming-en`
- Parakeet refinement initialized: `parakeet-tdt-0.6b-v2`
- Parakeet model warm-up completed
- Audio stream initialized at 16 kHz
- App reached `VOICE TYPING ACTIVE`
- Runtime reported `Streaming STT active (streaming + refinement)`

Current validated structure:

- Pass 1: `moonshine-medium-streaming-en`
- Pass 2: `parakeet-tdt-0.6b-v2`
- Zipformer remains available as true-online fallback if needed

### IBus Runtime Bring-Up

Verified the app can run on the IBus insertion path in the current session after explicitly restoring the user bus environment.

Observed:

- `DBUS_SESSION_BUS_ADDRESS` was absent in the launch shell
- The user bus socket still existed at `/run/user/1001/bus`
- Starting `ibus_voice_engine.py` with `DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1001/bus` succeeded
- Activating `voice-typing` via `ibus engine voice-typing` instantiated the engine and enabled it
- Restarting `enhanced-voice-typing.py --streaming --refinement` then printed `IBus engine available (atomic text insertion)`
- The app again reached `VOICE TYPING ACTIVE`

Current live runtime:

- IBus engine process running
- Voice typing app running
- Input path is IBus primary, with uinput still initialized as fallback

### Live Dictation Verdict

Tested on live utterances spoken through the current pipeline:

- `testing testing one two three`
- `I'm running things through Parakeet and also through Moonshine`
- `If just Parakeet is good enough or Moonshine...`
- `It actually seems quite slow running it through IBus...`

Observed from runtime output and `~/.local/state/voice-typing/voice-typing.log`:

- Current `Moonshine + Parakeet` run: 8 utterances, average refinement/confirmation latency `1.422s`, median `1.091s`, max `2.959s`
- Refinement corrections only: 6 utterances, average `1.224s`, max `2.959s`
- Confirmed/no-change refinements: 2 utterances, average `2.014s`, max `2.841s`
- Older fast path sample (2026-03-03 log): 48 utterances, average `0.161s`, max `0.226s`

Quality outcome:

- Moonshine pass-1 was already usable and mostly correct
- Parakeet pass-2 improved punctuation/casing and corrected at least one important lexical miss (`paraquite` -> `Parakeet`)
- Parakeet also produced at least one awkward correction on a short split fragment (`But the previous attempt with the`)

UX outcome:

- With IBus + refinement enabled, endpoint text is intentionally held in preedit until refinement returns
- This makes the app feel slow even when Parakeet ends up confirming the original text unchanged
- The runtime also showed sherpa-onnx offline falling back to CPU despite requesting `cuda`, which materially increases refinement latency

Recommended operating mode right now:

- For live dictation: Moonshine only on IBus
- For accuracy-sensitive batch cleanup: Parakeet as explicit offline refinement, not inline blocking refinement

Recommended hybrid follow-up:

- Keep Moonshine as immediate IBus commit path
- Run Parakeet only as bounded post-commit correction when latency is low and the edit distance is small
- Do not hold IBus preedit waiting on Parakeet by default

### Implementation Update: Optional Post-Commit Correction

Updated the app architecture so streaming text commits immediately and offline correction is optional and default-off.

Delivered:

- Added new user-facing config/CLI surface:
  - `post_commit_correction: false`
  - `correction_model: parakeet-tdt-0.6b-v2`
  - `--post-commit-correction`
  - `--correction-model`
- Kept legacy compatibility:
  - `--refinement` / `--refinement-model`
  - `VOICE_REFINEMENT` / `VOICE_REFINEMENT_MODEL`
  - config keys `refinement` / `refinement_model`
- Removed blocking IBus preedit hold at endpoint
- Endpoint text now commits immediately on IBus
- Optional correction now runs after commit and applies replacement in place
- IBus correction prefers `send_replace()` when surrounding-text support is available

Verification:

- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python -m py_compile enhanced-voice-typing.py streaming_stt.py tests/test_model_catalog.py'`
- `nix-shell --run 'python enhanced-voice-typing.py --help'`
- `timeout 20s nix-shell --run 'python enhanced-voice-typing.py --streaming'`
- `timeout 20s nix-shell --run 'python enhanced-voice-typing.py --streaming --post-commit-correction'`

Observed:

- `--streaming` starts in `Streaming-only mode (no offline correction)`
- `--streaming --post-commit-correction` loads Parakeet and still reaches `VOICE TYPING ACTIVE`

## Parakeet CTC Streaming Branch Update -- 2026-03-11

Branch:

- `feature/parakeet-ctc-streaming`

Delivered in repo:

- Added a new streaming backend entry for `parakeet-ctc-0.6b`
- Switched this branch default streaming model to `parakeet-ctc-0.6b`
- Kept Moonshine native and zipformer as selectable alternatives
- Kept optional post-commit correction architecture from the prior branch
- Updated CLI help, README, and CLAUDE docs so Parakeet CTC is described as the branch default instead of Moonshine

Dependency/runtime adjustments:

- Added `transformers>=4.57.3` and `librosa>=0.10.2` to `requirements.txt`
- Added `libsndfile` to `shell.nix` so `librosa` / `soundfile` can load cleanly in the nix shell
- Tightened `parakeet_ctc_available()` so it only returns true when `transformers`, `torch`, and `librosa` are all present
- Updated the Parakeet CTC install hint to include `librosa`

Tests added:

- Added a buffered Parakeet CTC streaming unit test with fake `torch` and fake `transformers`
- Added regression coverage that `parakeet_ctc_available()` goes false when `librosa` is missing
- Added regression coverage that the Parakeet CTC install hint mentions `librosa`

Verification completed:

- `nix-shell --run 'python - <<\"PY\"\nimport librosa\nprint(librosa.__version__)\nPY'`
- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python -m py_compile enhanced-voice-typing.py streaming_stt.py tests/test_model_catalog.py'`
- `nix-shell --run 'ruff check streaming_stt.py tests/test_model_catalog.py --select E,F'`
- `nix-shell --run 'python - <<\"PY\"\nfrom streaming_stt import parakeet_ctc_available, streaming_model_install_hint\nprint(parakeet_ctc_available())\nprint(streaming_model_install_hint(\"parakeet-ctc-0.6b\"))\nPY'`

Observed runtime behavior:

- The first real Parakeet CTC startup no longer fails on missing `librosa`
- App startup reaches:
  - GPU initialization
  - IBus detection
  - audio calibration
  - Hugging Face model resolution for `nvidia/parakeet-ctc-0.6b`
- First run begins downloading `model.safetensors` at `2.44G`
- In this shell, unauthenticated Hugging Face download throughput was only about `196kB/s`, so the full first-run model fetch was not practical to wait out during this turn

Current branch status:

- Code path, dependency surface, docs, and tests are aligned
- Real runtime startup gets past feature-extractor initialization and into model download
- Remaining unverified piece is complete end-to-end boot after the full `2.44G` Parakeet CTC model finishes downloading

Operational note:

- I stopped the old live voice-typing app before the Parakeet smoke test and did not leave a new app instance running after cancelling the long model download

## No-IBus Streaming Validation Update -- 2026-03-11

Goal:

- Compare the simplified streaming paths without IBus and without post-commit correction

Issue found and fixed:

- `IBusClient.is_available` was treating any leftover socket path as a live engine
- That caused the app to report `IBus engine available` even after `ibus_voice_engine.py` had been stopped
- Updated `enhanced-voice-typing.py` so availability now requires a successful Unix socket connection instead of plain path existence
- Added regression coverage for:
  - stale socket path returns unavailable
  - live Unix socket returns available

Verification completed:

- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python -m py_compile enhanced-voice-typing.py streaming_stt.py tests/test_model_catalog.py'`
- `nix-shell --run 'ruff check tests/test_model_catalog.py --select E,F'`

Observed runtime behavior without IBus and without correction:

- Moonshine launch:
  - `nix-shell --run 'python enhanced-voice-typing.py --streaming --streaming-model moonshine-medium-streaming-en --no-post-commit-correction'`
  - Reported `IBus engine not running (using key injection fallback)`
  - Initialized direct `uinput` injection
  - Reached `VOICE TYPING ACTIVE`
- Parakeet CTC launch:
  - `nix-shell --run 'python enhanced-voice-typing.py --streaming --streaming-model parakeet-ctc-0.6b --no-post-commit-correction'`
  - Reported `IBus engine not running (using key injection fallback)`
  - Initialized direct `uinput` injection
  - Reached Hugging Face model fetch and began downloading `model.safetensors` (`2.44G`)
  - The launch was cancelled before completion because first-run download speed remained too slow for interactive testing in this shell

Current conclusion:

- Moonshine is immediately testable now in the clean fallback path
- Parakeet CTC is structurally on the right path but still blocked on first-run model download time, not on app architecture, IBus, or post-commit correction behavior

### Fallback Typing Behavior Adjustment

User-reported issue:

- In no-IBus mode, the direct key-injection path still looked like "refinement" because streaming partials were being backspaced and retyped live

Fix applied:

- Changed the no-IBus streaming path so partials are buffered internally only
- Finalized endpoint text is now typed once on the fallback path instead of visibly rewriting live partials
- Centralized utterance finalization so both endpoint commits and forced long-utterance flushes share the same behavior
- Prevented the streaming flush path from queueing offline transcription unless optional post-commit correction is actually enabled

Verification:

- Added focused unit tests covering:
  - no-IBus partials do not call `_type_raw()`
  - no-IBus finalized streaming text types once and does not queue correction
- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python -m py_compile enhanced-voice-typing.py streaming_stt.py tests/test_model_catalog.py'`

Operational note:

- The live Moonshine process had to be restarted after this patch because the previously running process still had the old partial-rewrite behavior in memory

## IBus Streaming Runtime Restore -- 2026-03-11

Goal:

- Restore the good visible streaming path: Moonshine with IBus and no post-commit correction

Observed startup state:

- `ibus-daemon` was already running on the user session bus
- The voice-typing IBus engine was not running
- `DBUS_SESSION_BUS_ADDRESS` was already set to `unix:path=/run/user/1001/bus`

Bring-up steps completed:

- Started `ibus_voice_engine.py` inside `nix-shell`
- Activated `voice-typing` with `ibus engine voice-typing`
- Confirmed the engine instantiated and reported client capability changes
- Restarted the app as:
  - `python enhanced-voice-typing.py --streaming --streaming-model moonshine-medium-streaming-en --no-post-commit-correction`

Runtime result:

- App reported `IBus engine available (atomic text insertion)`
- App reached `VOICE TYPING ACTIVE`
- Current live stack:
  - `ibus-daemon`
  - `ibus_voice_engine.py`
  - Moonshine streaming app with correction disabled

## Stable Partial Commit Mode -- 2026-03-11

Goal:

- Add a third streaming insertion mode that works better in terminals and other non-preedit-friendly targets

Design:

- Keep full IBus preedit for clients with surrounding-text support
- For terminal-like clients, stop relying on preedit visibility
- Commit only a stable prefix during the utterance
- Leave a small mutable tail uncommitted until later updates or the final endpoint
- Avoid large noisy full rewrites while still showing visible progress before endpoint

Delivered:

- Added a separate `visible_streaming_text` buffer alongside `current_streaming_text`
- Added `_streaming_preedit_enabled()` to keep true IBus preedit limited to the best-supported clients
- Added `_stable_streaming_prefix()` to hold back the last few words as a mutable tail
- Updated `_type_streaming_partial()` so non-preedit paths commit only stable prefixes
- Updated `_finalize_streaming_utterance()` so it replaces any visible stable prefix with the full final utterance once, instead of duplicating text

Verification:

- Added focused tests for:
  - stable prefix commits on fallback clients
  - preedit bypass remains intact for surrounding-text clients
  - finalization replaces a visible prefix instead of retyping the whole utterance
- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python -m py_compile enhanced-voice-typing.py streaming_stt.py tests/test_model_catalog.py'`
- `nix-shell --run 'ruff check tests/test_model_catalog.py --select E,F'`

Operational note:

- The live Moonshine process had to be restarted again after this change so the new insertion behavior would take effect

### Stable Partial Timing Tuning

User feedback:

- Rendering still felt too slow even after the stable-partial mode landed

Tuning applied:

- Lowered Moonshine medium native update interval from `0.20s` to `0.14s`
- Lowered the app-side streaming type cadence from `0.15s` to `0.08s`
- Reduced the held-back mutable tail from `3` words to `2`

Verification:

- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python -m py_compile enhanced-voice-typing.py streaming_stt.py tests/test_model_catalog.py'`

Operational note:

- Restarted the live Moonshine app again so the tighter timing constants are active in the current runtime

### Streaming Hybrid Display Analysis

Timestamp:

- 2026-03-11T11:50:53-07:00

Findings:

- Best insertion point for a third streaming display mode is the non-preedit branch of `_type_streaming_partial()`
- `streaming_worker()` can stay mostly unchanged if it continues flushing `pending_partial` before `_finalize_streaming_utterance()`
- `_finalize_streaming_utterance()` only needs to swap from a single `visible_streaming_text` buffer to the effective visible preview buffer used by the new mode
- `typing_history` should continue recording only finalized utterances, not any stable-prefix or mutable-tail preview state
- `_compute_replacement()` and `_replace_typed_text()` are already the right low-level primitives for tail-only rewrites as long as the new mode keeps the committed prefix monotonic

Edge cases noted:

- If the recognizer retracts text inside the already-committed stable prefix, skip the mid-utterance rewrite and let finalization or post-commit correction fix it
- Preserve leading-space handling between utterances when splitting stable prefix vs mutable tail
- Clear any mutable tail state on endpoint, flush, and reset paths

### Streaming Lag Inspection

Timestamp:

- 2026-03-11T12:00:43-07:00

Scope:

- Read-only inspection of `enhanced-voice-typing.py` and `streaming_stt.py`
- Focused on remaining visible lag after `streaming_use_ibus_preedit` is disabled by default

Findings:

- Native Moonshine still has an intentional model-side update cadence floor of roughly `0.14s` to `0.18s` depending on model config, and the worker only samples whatever partial text is available after each chunk
- The app adds another UI throttle with `type_interval = 0.08`, and pending partials can also wait for the worker's `streaming_queue.get(timeout=0.1)` timeout when audio has gone quiet
- Non-preedit fallback intentionally withholds the last `2` words via `_stable_streaming_prefix()`, so the visible text will remain behind the recognizer tail even when audio and decoding are fast
- Partial updates are batch-drained and rendered once per worker pass, so any transient backlog collapses multiple `20ms` chunks into bursty visible updates instead of smooth ones
- Finalization and mid-utterance correction are only atomic when IBus surrounding-text replace is supported; otherwise visible lag can come from backspace-and-retype behavior through the input injector path

### Moonshine Endpoint Formatting Inspection

Timestamp:

- 2026-03-11T12:02:14-07:00

Scope:

- Read-only inspection of native Moonshine wrapper flow in `streaming_stt.py`
- Review of recent live endpoint logs in `~/.local/state/voice-typing/voice-typing.log`

Findings:

- Native Moonshine partial and final events are passed through from `event.line.text` with only `.strip()` applied in `_handle_moonshine_event()`
- `check_endpoint()` returns queued Moonshine `LineCompleted` text verbatim; there is no punctuation scrub or text normalization on that path
- The app-side streaming path still lowercases via `_type_streaming_partial()`, so current endpoint logs preserve Moonshine punctuation and contractions but mostly lose capitalization
- Recent March 11 Moonshine-only runs show `streaming_endpoint` lines with punctuation and no adjacent `refinement_correction`, which means the formatting is already present before any offline pass
- Older March 3 runs show the previous behavior: mostly raw lowercase endpoint text followed by separate `refinement_correction` entries that add punctuation and casing

### Moonshine Stable-Partial Lag Tuning

Timestamp:

- 2026-03-11T12:06:32-07:00

Changes:

- Disabled IBus preedit by default for streaming display and made the flag backward-safe for test stubs that bypass `__init__`
- Reduced Moonshine native update cadence for the current default model from `0.14s` to `0.10s`
- Reduced app-side partial flush interval from `0.08s` to `0.04s`
- Reduced quiet-queue flush timeout from `0.10s` to `0.04s`
- Reduced stable-prefix holdback from `2` words to `1` word
- Added a startup banner that reports the active streaming display mode

Verification:

- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python -m py_compile enhanced-voice-typing.py streaming_stt.py tests/test_model_catalog.py'`

Notes:

- Full `ruff check enhanced-voice-typing.py --select E,F` still reports the pre-existing backlog in that file; no new lint class was introduced by this tuning pass
- The running app must be restarted after this milestone to pick up the tighter timing settings

### Stable Partial Spacing Fix

Timestamp:

- 2026-03-11T12:39:48-07:00

Problem:

- Visible streaming text was losing spaces and showing merged words even though `streaming_endpoint` logs remained correctly spaced
- Root cause was the new stable-partial mode still doing mid-utterance replacements instead of monotonic suffix appends

Changes:

- `_type_streaming_partial()` now treats stable partial display as append-only
- If the recognizer revises text inside the already visible prefix, the app skips that mid-utterance rewrite and defers reconciliation until endpoint
- Added regression tests for monotonic suffix append behavior and for skipping non-monotonic mid-utterance rewrites

Verification:

- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python -m py_compile enhanced-voice-typing.py streaming_stt.py tests/test_model_catalog.py'`
- `nix-shell --run 'ruff check tests/test_model_catalog.py streaming_stt.py --select E,F'`

### IBus Socket Whitespace Fix

Timestamp:

- 2026-03-11T14:22:31-07:00

Problem:

- Visible streaming text still merged words like `tosee` and `tohave` even after the stable-partial path became append-only
- Endpoint logs remained correctly spaced, which isolated the failure to the IBus insertion hop rather than Moonshine decoding
- Root cause was `ibus_voice_engine.py` stripping each socket line with `.strip()`, which removed the leading and trailing spaces from `commit:` and `replace:` payloads

Changes:

- Preserved payload whitespace in `_handle_client()` by decoding socket lines without `.strip()`
- Added a regression test that verifies `_handle_client()` preserves both leading and trailing spaces in `commit:` and `replace:` commands
- Cleaned the small local Ruff backlog in `ibus_voice_engine.py` introduced by touching that file

Verification:

- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python -m py_compile enhanced-voice-typing.py streaming_stt.py ibus_voice_engine.py tests/test_model_catalog.py'`
- `nix-shell --run 'ruff check ibus_voice_engine.py tests/test_model_catalog.py streaming_stt.py --select E,F'`

Notes:

- This fix requires restarting both `ibus_voice_engine.py` and the main app process

### Moonshine Long-Utterance Display Tuning

Timestamp:

- 2026-03-11T14:41:22-07:00

Problem:

- Longer utterances still felt laggy even after the spacing fix
- Logs showed repeated `streaming_partial_skip ... reason=non_monotonic` events, meaning visible text stalled whenever Moonshine revised earlier words inside the already committed prefix

Changes:

- Kept append-only stable partials as the default
- Allowed bounded mid-utterance replacement when the focused app supports IBus surrounding-text operations
- Reused `_replace_typed_text()` for those bounded rewrites so compatible clients can stay visually current without waiting for endpoint
- Added regression tests covering bounded non-monotonic replace and large non-monotonic skip fallback

Verification:

- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python -m py_compile enhanced-voice-typing.py streaming_stt.py ibus_voice_engine.py tests/test_model_catalog.py'`
- `nix-shell --run 'ruff check ibus_voice_engine.py tests/test_model_catalog.py streaming_stt.py --select E,F'`

Notes:

- The unittest run completed successfully but the interpreter segfaulted on process teardown inside the nix shell after reporting `OK`
- The current evidence indicates native Moonshine is effectively CPU-only in this environment: the integration does not pass a device/provider to `moonshine_voice.Transcriber`, and the installed bundled ONNX Runtime lacks CUDA-linked provider libraries

### Stable Partial Lag Inspection

Timestamp:

- 2026-03-11T14:45:00-07:00

Problem:

- Longer utterances still feel visually laggy after the append-only stable-partial change
- Need to separate recognizer latency from display-policy lag and verify whether IBus surrounding-text replacement can help

Findings:

- Current streaming display defaults to `stable partial commits`, not live IBus preedit, because `self.streaming_use_ibus_preedit = False`
- The stable-partial path only commits the stable prefix minus the last word, and it refuses any non-monotonic mid-utterance rewrite
- March 11 runtime logs now show repeated `streaming_partial_skip ... reason=non_monotonic` lines on longer utterances, which means the visible text can stall on an older prefix until endpoint
- The 40 ms app-side flush interval is not the main bottleneck here; the stronger lag source is the append-only freeze whenever the recognizer revises already visible words
- IBus surrounding-text replace already exists for finalized replacement paths and would technically support smoother bounded mid-utterance rewrites on capable clients, but it is not used by `_type_streaming_partial()` today and will not help clients that only advertise `basic`

Evidence:

- `enhanced-voice-typing.py`: `_type_streaming_partial()`, `_stable_streaming_prefix()`, `_replace_typed_text()`, `_streaming_preedit_enabled()`
- `ibus_voice_engine.py`: `replace_chars()`, socket `replace:` dispatch, capability tracking
- `~/.local/state/voice-typing/voice-typing.log`: March 11 `streaming_partial_skip` entries around 13:15 through 14:35

### NVIDIA Parakeet NIM Realtime Backend

Timestamp:

- 2026-03-11T15:18:00-07:00

Problem:

- The strongest streaming path for a large NVIDIA GPU is not the current local Moonshine or buffered local Parakeet path
- The app needed a real GPU-native streaming backend for Parakeet CTC, not another round of CPU-bound Moonshine tuning

Changes:

- Added explicit NIM streaming model entries:
  - `parakeet-ctc-0.6b-nim`
  - `parakeet-ctc-1.1b-nim`
- Implemented NVIDIA Riva ASR NIM realtime websocket support in `streaming_stt.py`
- Wired the official flow:
  - `POST /v1/realtime/transcription_sessions`
  - `WS /v1/realtime?intent=transcription`
  - `transcription_session.update`
  - `input_audio_buffer.append`
  - `input_audio_buffer.commit`
  - `conversation.item.input_audio_transcription.delta`
  - `conversation.item.input_audio_transcription.completed`
- Added `VOICE_NIM_URL` and optional `VOICE_NIM_API_KEY` support at the backend layer
- Added regression coverage for model catalog registration, install hints, and a fake NIM realtime session that exercises partials, endpoints, reset, and close
- Added `websocket-client` to Python requirements
- Updated README and CLAUDE notes to document the preferred GPU path and the NIM startup contract

Verification:

- `nix-shell --run 'python -m unittest tests.test_model_catalog.StreamingCatalogTests tests.test_model_catalog.ParserDefaultTests tests.test_model_catalog.NimRealtimeStreamingTests'`
- `nix-shell --run 'python -m unittest tests.test_model_catalog'`
- `nix-shell --run 'python -m py_compile enhanced-voice-typing.py streaming_stt.py ibus_voice_engine.py tests/test_model_catalog.py'`
- `nix-shell --run 'ruff check streaming_stt.py tests/test_model_catalog.py --select E,F'`
- `nix-shell --run 'python enhanced-voice-typing.py --help'`

Notes:

- No `nvcr.io` auth was present on disk in this environment:
  - `~/.docker/config.json` had no registry entries
  - `~/.ngc/config` was absent
- Docker and the NVIDIA container runtime are installed locally, so the remaining blocker for a live bring-up is credentials plus the actual NIM image pull
- The full unittest command still reports `OK` and then segfaults during interpreter teardown inside the nix shell, matching the earlier teardown-only instability already seen on this branch

---

## 2026-04-08 NVIDIA Build Lookup

Question answered: which NVIDIA Build Parakeet model matches the NIM path this repo wants.

Result:

- The repo's recommended large-GPU NIM target is `parakeet-ctc-1.1b-nim`.
- On NVIDIA Build, the matching public model page is `parakeet-ctc-1.1b-asr`.
- The deploy page for that model pulls the container image `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:latest`.
- That image name matches the internal realtime model ID already wired in `streaming_stt.py`.

Secondary mapping:

- Repo `parakeet-ctc-0.6b-nim` maps to NVIDIA Build `parakeet-ctc-0.6b-asr`.
- `parakeet-tdt-0.6b-v2` is present on NVIDIA Build, but it is the offline correction/batch path, not the preferred streaming NIM target for this repo.

---

## 2026-04-08 Nemotron Migration

Question answered: move the repo's recommended NVIDIA streaming path from older Parakeet CTC NIM to Nemotron.

Changes:

- Added `nemotron-asr-streaming-nim` to the streaming model catalog.
- Kept the existing Parakeet CTC NIM entries as supported compatibility baselines.
- Made the realtime NIM session model override optional so Nemotron can use the server-selected default model instead of guessing an internal model ID.
- Updated README guidance to recommend `nvcr.io/nim/nvidia/nemotron-asr-streaming:latest` for the best NVIDIA GPU streaming path.
- Marked `PARAKEET-NIM-VS-ONNX.md` as historical context rather than the current top recommendation.
- Added regression coverage for a NIM session that preserves the server default realtime model.

Verification:

- `python -m unittest tests.test_model_catalog.StreamingCatalogTests tests.test_model_catalog.ParserDefaultTests tests.test_model_catalog.NimRealtimeStreamingTests`
- `python -m py_compile streaming_stt.py tests/test_model_catalog.py enhanced-voice-typing.py`
- `ruff check streaming_stt.py tests/test_model_catalog.py --select E,F`

Results:

- Local focused unit tests passed: 13 tests, `OK`.
- Local compile check passed.
- Touched-file Ruff check passed.

Notes:

- A broader Ruff run including `enhanced-voice-typing.py` still reports long-standing pre-existing lint debt unrelated to this change.
- Re-running the same focused verification inside `nix-shell` stalled in environment/package resolution after fetching GTK/Guile/Xbindkeys inputs, so there is no completed nix-shell result to report yet.

Follow-up:

- The actual runtime default streaming model was then switched from local `parakeet-ctc-0.6b` to `nemotron-asr-streaming-nim` in `enhanced-voice-typing.py`, so `./voice --streaming` now targets Nemotron unless overridden.

Operational status:

- Confirmed local GPU: `NVIDIA RTX A6000`, `49140 MiB`, driver `580.119.02`.
- Confirmed Docker daemon is healthy and `docker login nvcr.io` succeeded with the provided NGC key.
- Port `9000` is already occupied by an unrelated local Node process (`garden-glow-market` backend), so the planned Nemotron bind is `9001->9000` and `50052->50051`.
- `docker pull nvcr.io/nim/nvidia/nemotron-asr-streaming:latest` was started and partial layers were downloaded, including progress into a `3.572GB` layer and a `551.6MB` layer before the pull was interrupted.
- Docker auth was then removed with `docker logout nvcr.io` to avoid leaving the provided key stored unencrypted in `~/.docker/config.json`.
- The container had not yet been launched or marked ready.

---

## 2026-04-08 Voice Restore On Local Fallback

Timestamp:

- 2026-04-08T15:11:30-07:00

Problem:

- The Nemotron NIM image is now fully pulled locally, but this machine's Docker GPU runtime is broken at the daemon level.
- The user still needed working voice typing from this repo immediately.

Changes:

- Added machine-local runtime env loading to `voice`, so `~/.config/voice-typing/runtime.env` now controls the active streaming backend without editing repo defaults.
- Updated `voice` to self-heal the IBus component install by relinking `~/.local/share/ibus/component/voice-typing-ibus.xml` to the active checkout when the link is missing or stale.
- Updated `voice` to auto-start `ibus-daemon -drx` when the session does not already have an IBus daemon.
- Kept the dedicated `voice-typing-nemotron-nim.service` template in the repo, but removed the automatic `Wants=` dependency from the main `voice-typing.service` and split daemon unit so local backends can run without dragging in a broken NIM service.
- Installed `~/.config/voice-typing/nim.env` with the NGC key and host port mapping for the future Nemotron service.
- Installed `~/.config/voice-typing/runtime.env` and temporarily switched the live runtime model to `moonshine-medium-streaming-en` so voice typing could be restored immediately.
- Installed and enabled the main user unit at `~/.config/systemd/user/voice-typing.service`.
- Corrected the stale IBus component symlink that still pointed at `/home/jordan/voice-typing-nix/voice-typing-ibus.xml`.

Nemotron blocker:

- `docker pull nvcr.io/nim/nvidia/nemotron-asr-streaming:latest` completed successfully.
- Starting the NIM unit failed before container launch because Docker is configured with a stale NVIDIA runtime path in `/nix/store/2x4yahw1v7kddydy8d85w1hj2rlxfnra-daemon.json`:
  - configured runtime path: `/nix/store/cbzcnh7bc9g35lf2374aydx78cv61hw8-nvidia-docker/bin/nvidia-container-runtime`
  - working runtime path currently on disk: `/nix/store/5qj8yc0rz1gpdm02yywz6vxwhblkcnn4-nvidia-docker/bin/nvidia-container-runtime`
- Repro:
  - `docker run --rm --runtime=nvidia hello-world`
  - failure: `fork/exec ... nvidia-container-runtime: no such file or directory`
- This is a root-owned Docker daemon configuration issue, not a repo code issue.

Verification:

- `docker pull nvcr.io/nim/nvidia/nemotron-asr-streaming:latest`
- `docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | rg 'nvcr.io/nim/nvidia/nemotron-asr-streaming'`
- `docker run --rm hello-world`
- `docker run --rm --runtime=nvidia hello-world`
- `timeout 180s nix-shell ./shell.nix --run 'python - <<\"PY\" ... streaming_model_available(...) ... PY'`
- `systemd-analyze --user verify systemd/voice-typing.service systemd/voice-typing-daemon.service systemd/voice-typing-nemotron-nim.service`
- `systemctl --user restart voice-typing.service`
- `journalctl --user -u voice-typing.service -n 200 --no-pager`

Results:

- Nemotron image is present locally:
  - `nvcr.io/nim/nvidia/nemotron-asr-streaming:latest`
  - image size `25.1GB`
- Docker itself works for CPU containers.
- Docker GPU containers do not work until the daemon runtime path is fixed.
- Local fallback voice typing is running successfully under the user service.
- Journal confirms:
  - `IBus Voice Typing engine running`
  - `IBus voice-typing engine active`
  - `Streaming recognizer initialized (moonshine-medium-streaming-en)`
  - `Audio stream initialized (sample rate: 16000 Hz)`
  - `VOICE TYPING ACTIVE`

Live state:

- `voice-typing.service` is active under systemd user services.
- `ibus-daemon`, `ibus_voice_engine.py`, and `enhanced-voice-typing.py --streaming --device cuda` are all running.
- `~/.config/voice-typing/runtime.env` currently points to:
  - `VOICE_NIM_URL=http://127.0.0.1:9001`
  - `VOICE_STREAMING_MODEL=moonshine-medium-streaming-en`
- `~/.config/voice-typing/nim.env` is present with mode `0600`.

---

## 2026-04-08 Nemotron Live Cutover

Timestamp:

- 2026-04-08T15:39:55-07:00

Problem:

- The earlier Docker `nvidia` runtime failure was real, but it was not the only blocker.
- Once Nemotron came up, the repo's realtime client crashed the streaming worker on the first nonblocking websocket poll with `BlockingIOError: [Errno 11] Resource temporarily unavailable`.

Changes:

- Switched the Nemotron user unit to Docker CDI device wiring instead of the broken daemon-level `--runtime=nvidia` path:
  - `--device nvidia.com/gpu=0`
  - host ports remain `9001->9000` and `50052->50051`
- Confirmed the NIM readiness endpoint returns HTTP 200 at `http://127.0.0.1:9001/v1/health/ready`.
- Updated `~/.config/voice-typing/runtime.env` back to the Nemotron runtime:
  - `VOICE_NIM_URL=http://127.0.0.1:9001`
  - `VOICE_STREAMING_MODEL=nemotron-asr-streaming-nim`
- Patched `streaming_stt.py` so `_nim_receive_event()` treats `BlockingIOError` and `EAGAIN/EWOULDBLOCK` as the expected "no event available yet" condition during zero-timeout websocket drains.
- Added a regression test covering the exact Nemotron nonblocking receive failure in `tests/test_model_catalog.py`.
- Restarted the user voice service after the code fix and cut the live recognizer back over to Nemotron.

Verification:

- `docker ps --filter name=nemotron-asr-streaming`
- `curl -i http://127.0.0.1:9001/v1/health/ready`
- `timeout 180s nix-shell ./shell.nix --run 'VOICE_NIM_URL=http://127.0.0.1:9001 python - <<\"PY\" ... create_recognizer() ... PY'`
- `timeout 180s nix-shell ./shell.nix --run 'VOICE_NIM_URL=http://127.0.0.1:9001 python - <<\"PY\" ... feed_chunk(np.zeros(...)) ... PY'`
- `python -m unittest tests.test_model_catalog.NimRealtimeStreamingTests`
- `python -m py_compile streaming_stt.py tests/test_model_catalog.py`
- `ruff check streaming_stt.py tests/test_model_catalog.py --select E,F`
- `systemctl --user restart voice-typing.service`
- `journalctl --user -u voice-typing.service -n 60 --no-pager`

Results:

- `voice-typing-nemotron-nim.service` is active and enabled.
- `voice-typing.service` is active and enabled.
- Nemotron is live in Docker as `nemotron-asr-streaming`.
- The repo-side realtime client now survives a real `feed_chunk()` call against Nemotron without raising the prior websocket error.
- The live journal now confirms:
  - `Streaming recognizer initialized (nemotron-asr-streaming-nim)`
  - `Audio stream initialized (sample rate: 16000 Hz)`
  - `VOICE TYPING ACTIVE`
  - `Streaming STT active (streaming-only)`
- No new `StreamingWorker` traceback appeared after the post-fix restart.

Live state:

- Active streaming backend: `nemotron-asr-streaming-nim`
- Active NIM URL: `http://127.0.0.1:9001`
- Main control socket: `/run/user/1001/voice-typing-1001.sock`
- IBus control socket: `/run/user/1001/voice-typing-ibus-1001.sock`

Addendum:

- The first Nemotron cutover still exposed one more protocol bug under live use: the client sent `input_audio_buffer.clear`, which this NIM build rejects.
- Follow-up fix:
  - stopped sending the unsupported websocket clear event
  - kept Nemotron reset/endpointer behavior local to the client state
  - marked completed transcripts as buffer-clean locally
- Follow-up verification:
  - repeated `feed_chunk()` loop against the live NIM returned `NO_STREAM_ERROR`
  - restarted `voice-typing.service` again at `2026-04-08T15:42:03-07:00`
  - watched the service journal for ~35s with no new `StreamingWorker` traceback
  - watched `docker logs` for the same window and saw only normal websocket acceptance plus `Starting inference task`
