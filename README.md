# NVFP4 Quantization Recovery — QAD vs AutoRound vs AWQ

Compare three accuracy-recovery methods after quantizing a trained LLM to **NVFP4**
(NVIDIA's Blackwell-native 4-bit float format): **QAD**, **AutoRound**, and **AWQ**.
All three target the *identical* number format ("Option B"), so any difference reflects
the recovery method, not the format.

## TL;DR

The original ask was to compare **QAD, AutoRound, AWQ, NVFP4, and bnb**. Those five
are not peers — they sit on two different axes:

| Axis | Members |
|---|---|
| **Number format** (where the bits go) | NVFP4, NF4 (bnb), INT4, FP8, … |
| **Recovery method** (how we minimize accuracy loss) | QAD, AutoRound, AWQ, RTN, QLoRA, … |

The meaningful experiment is: **fix one format, vary the recovery method**.
We fixed the format to **NVFP4** (because that's what Blackwell accelerates) and
ran QAD vs AutoRound vs AWQ. We did **not** run bnb experiments — bnb is a different
number format (NF4) and would only matter if Blackwell-native deployment isn't an option.

**Headline finding (Qwen3-0.6B mental-model run, 8 prompts × ~80 tokens):**

| Method (best config) | Post KL ↓ | % Recovered ↑ | Time |
|---|---:|---:|---:|
| **AR  iters=100** | **6.02** | **90.1%** | **25.8 s** ← best efficiency |
| AR  iters=500 | 5.22 | 91.4% | 106.1 s |
| QAD steps=500 T=4.0 | 18.38 | 70.3% | 60.2 s |
| AWQ awq_full | 50.16 | 19.4% | 11.6 s |

AR dominated the Pareto frontier at 0.6B. Whether QAD's "more headroom" advantage
reasserts itself at 4B+ is the open question that `combined_experiment_production.py`
is built to answer.

## What is NVFP4

NVIDIA's 4-bit floating-point format, native to Blackwell Tensor Cores:
- **E2M1**: 1 sign + 2 exponent + 1 mantissa = 16 representable values per number
- **Block size 16**: every 16 weights share one FP8 (E4M3) scale
- **Two-level scaling**: per-block FP8 scales themselves scaled by one FP32 per-tensor scale
- **W4A4**: weights AND activations both quantized

Naively rounding to NVFP4 (RTN) destroys accuracy. The three recovery methods exist
to claw back as much of the gap to the BF16 teacher as possible.

## Method comparison

| Axis | **QAD** | **AutoRound** | **AWQ** |
|---|---|---|---|
| Recovery mechanism | KL distillation into quantized weights | Sign-grad rounding direction opt | Per-channel α scaling search |
| Backprop? | Full (through STE) | Sign-grad only | None — grid search |
| Teacher needed? | Yes (BF16 frozen) | No | No |
| Calibration data | Token sequences | Token sequences | Token sequences |
| Knobs | STEPS, LR, TEMPERATURE | ITERS, SEQLEN, LR | ALGORITHM (lite/clip/full) |
| Theoretical headroom | Highest | Medium | Lowest |
| Cost | Slowest | Medium | Fastest |

## How recovery is measured

KL divergence measures how different two probability distributions are. We compare
each NVFP4 model against the **BF16 teacher** because that's the behavior we want
to preserve.

| Number | What it is | Role |
|---|---|---|
| **Oracle KL** | BF16 teacher vs itself | Floor — perfect agreement target (≈ 0) |
| **RTN KL** | BF16 teacher vs NVFP4 with no recovery | Ceiling — worst case any method should beat |
| **Post-method KL** | BF16 teacher vs NVFP4 after QAD/AR/AWQ | The scoreline — lower is better |

```
% recovered = (RTN_KL − Post_KL) / (RTN_KL − Oracle_KL)
```

100% means the recovery method completely undid quantization damage; 0% means it
did nothing.

## Repository layout

| File | Purpose |
|---|---|
| `mental_model.py` | End-to-end SFT/RL/QAD reference pipeline (Qwen3-0.6B) — explains the 5 training rules and how QAD slots in after SFT+RL |
| `qad_experiment.py` | Standalone QAD experiment + KL measurement |
| `qad_ablation.py` | QAD parameter sweep (steps, lr, temperature) |
| `autoround_experiment.py` | Standalone AutoRound experiment |
| `autoround_ablation.py` | AutoRound parameter sweep (iters, lr, seqlen) |
| `awq_experiment.py` | Standalone AWQ experiment |
| `awq_ablation.py` | AWQ algorithm-variant sweep |
| `combined_experiment.py` | All three methods, head-to-head, on Qwen3-0.6B |
| `combined_experiment_production.py` | Production sweep on Qwen3-4B with logit caching + 8-bit AdamW + gradient checkpointing |
| `smoke_test.py` | 1-min sanity check — verifies all imports, CUDA, and a forward+backward step |
| `Dockerfile` | Reproducible environment (PyTorch 2.7 + CUDA 12.8 + nvidia-modelopt[hf] + auto-round + bitsandbytes) |

## Hardware requirements

| Model | Minimum VRAM | Notes |
|---|---|---|
| Qwen3-0.6B (mental model + 0.6B experiments) | 8 GB | Any modern NVIDIA GPU |
| Qwen3-4B (`combined_experiment_production.py`) | 32 GB (RTX 5090) | Requires logit caching + 8-bit AdamW + grad checkpoint |
| Qwen3-14B | ~80 GB (A100/H100) | QAD doesn't fit on 5090 even with all tricks |

## How to run

### Build the Docker image

```bash
docker build -t quantization-recovery .
```

### Smoke test (always run this first, ~1 min)

```bash
docker run --gpus all --rm \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  quantization-recovery python /workspace/smoke_test.py
```

### Mental-model run (Qwen3-0.6B, fits any GPU, ~3-5 min)

```bash
docker run --gpus all --rm \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  quantization-recovery python /workspace/combined_experiment.py
```

### Production run (Qwen3-4B on 5090, ~3-5 hours)

```bash
# Use tmux so SSH disconnects don't kill the run
tmux new -s prod-sweep
docker run --gpus all --rm --shm-size=8g \
  -e HF_TOKEN \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  quantization-recovery python /workspace/combined_experiment_production.py
# Ctrl-b d to detach; tmux attach -t prod-sweep to reattach
```

Results land in `runs/results_<timestamp>/results.json` and `summary.csv`,
written incrementally after every sweep config (so a mid-run crash still
leaves usable data).

## Why we excluded bnb

`bitsandbytes` (bnb) is a 4-bit format (NF4), not a recovery method. We didn't
run a bnb experiment because for Blackwell deployment it's dominated on three
axes versus NVFP4 + (QAD/AR/AWQ):

1. **Throughput**: NVFP4 runs natively on Blackwell Tensor Cores (~2.35× INT4 throughput per published benchmarks). NF4 has to dequantize to BF16 around every matmul on every GPU.
2. **Accuracy at 4-bit**: published comparisons rank AWQ/AutoRound > bnb-NF4 on perplexity. NVFP4 paired with QAD/AR/AWQ compounds the win.
3. **Inference complexity**: bnb's typical recovery (QLoRA) leaves a BF16 LoRA on top of an NF4 base, creating an adapter/base mismatch at serving. NVFP4 + QAD trains the served weights directly via STE.

bnb would only be a candidate for prototyping on Ampere/Ada hardware where NVFP4
isn't supported.

## Open questions (TODO)

- Validate AR-beats-QAD finding at 4B and 8B scale (run `combined_experiment_production.py`)
- Test AWQ → QAD warm-start pipeline (use AWQ-quantized model as QAD initialization)
- Add downstream task evaluation (perplexity, MMLU) alongside KL divergence
- Compare NVFP4 vs INT4 (AWQ/AutoRound) vs FP8 on the same hardware
