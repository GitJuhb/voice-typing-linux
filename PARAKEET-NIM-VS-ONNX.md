# Why We Leaned NVIDIA Parakeet NIM Instead Of ONNX

Note dated 2026-04-08: this remains the historical explanation for choosing a first-party NVIDIA NIM path over local ONNX for streaming work. It is no longer the repo's current recommendation for the broader streaming ASR landscape; that role now goes to NVIDIA Nemotron ASR Streaming, with Parakeet CTC kept as an older supported NIM baseline.

## Short Answer

We were not choosing `Parakeet` versus `ONNX`.

We were choosing how to run a streaming-capable model:

- `Parakeet` is the model family
- `ONNX` is one runtime/deployment path
- `NVIDIA NIM` is another runtime/deployment path

For the specific goal of `best, most powerful, true streaming on a 48 GB GPU`, the NVIDIA Parakeet NIM path looked stronger than the local ONNX-style path.

## Why NIM Looked Better For That Goal

1. The local ONNX path in this repo already had clear limits.
   - `Parakeet TDT` in the sherpa/ONNX path is offline only, not true streaming.
   - Moonshine was working locally, but in this environment its native path was effectively CPU-bound.

2. The strongest verified Parakeet streaming path was NVIDIA's own streaming stack.
   - NVIDIA officially exposes streaming Parakeet CTC profiles through Riva ASR NIM.
   - That made `Parakeet 1.1b CTC` the best fit for a large-GPU, low-latency streaming target.

3. A 48 GB GPU changes the optimization target.
   - We were no longer optimizing for minimum setup friction.
   - We were optimizing for maximum streaming performance and headroom.

## Why We Did Not Just Say "Use ONNX"

The local ONNX-style route is simpler operationally, but it was not the best match for the requirement we were optimizing around.

If the priority had been:

- fully local with no registry login
- minimal setup friction
- easiest reproducibility

then the ONNX-first path would have been the better choice.

That was not the constraint at the time. The constraint was effectively:

`give me the strongest streaming path available`

Under that constraint, first-party NVIDIA streaming support for Parakeet CTC was the better bet.

## Tradeoff Summary

### NIM / NVIDIA Parakeet

Best when you want:

- strongest GPU-native streaming path
- first-party streaming support for Parakeet CTC
- maximum performance over minimum setup friction

Costs:

- NVIDIA registry auth
- container setup
- more operational overhead

### Local ONNX / sherpa-style path

Best when you want:

- simpler local setup
- no private registry login
- fewer moving parts

Costs:

- weaker fit for the "best possible streaming on a huge GPU" goal
- local streaming options in this repo had already shown practical limits

## Repo-Specific Outcome

The repo was left in a conservative state:

- local `parakeet-ctc-0.6b` remained the default streaming model
- NVIDIA NIM-backed models were added as explicit options
- the NIM path was not made the default because it depends on NVIDIA registry access to pull the container image

So the decision was:

- `NIM` for the strongest target path
- `local ONNX-style backend` as the lower-friction local fallback

## Bottom Line

We leaned NVIDIA Parakeet NIM because the requirement was not "simplest local runtime."

It was:

`what is the best, most powerful streaming path if we are willing to do whatever it takes`

For that question, NVIDIA Parakeet CTC on NIM was the stronger answer than the ONNX-style route.
