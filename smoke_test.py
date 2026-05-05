"""
Smoke test for NVFP4 migration — run this first on the 5090 server before
any of the real experiments. Catches install/version/config issues fast.

Tests, in order:
  1. Imports — all required packages load
  2. CUDA   — GPU available, capability check
  3. Model  — Qwen3-0.6B loads in BF16
  4. ModelOpt NVFP4 — fake quantization + calibration runs
  5. AutoRound FP4  — iters=0 RTN runs with data_type="fp"
  6. Forward + backward — STE backward pass produces gradients

Each step prints PASS/FAIL with a short reason. On first failure, exits.
Total runtime: ~1 minute on a 5090.
"""

import sys
import traceback


def step(name: str):
    """Decorator to wrap each test with PASS/FAIL output and fail-fast."""
    def deco(fn):
        def wrapper(*args, **kwargs):
            print(f"\n[{name}] ...", flush=True)
            try:
                result = fn(*args, **kwargs)
                print(f"[{name}] PASS")
                return result
            except Exception as e:
                print(f"[{name}] FAIL — {type(e).__name__}: {e}")
                traceback.print_exc()
                sys.exit(1)
        return wrapper
    return deco


@step("1. imports")
def test_imports():
    import torch
    import transformers
    import peft
    import auto_round
    import modelopt.torch.quantization as mtq
    print(f"  torch={torch.__version__}  transformers={transformers.__version__}")
    print(f"  peft={peft.__version__}  auto_round={auto_round.__version__}")
    # ModelOpt version
    try:
        import modelopt
        print(f"  nvidia-modelopt={modelopt.__version__}")
    except AttributeError:
        print(f"  nvidia-modelopt=(version attr missing)")
    return torch, mtq


@step("2. cuda")
def test_cuda(torch):
    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  device: {name}  sm_{cap[0]}{cap[1]}  vram={vram:.1f} GB")
    if cap[0] >= 10:
        print(f"  Blackwell detected — NVFP4 hardware path available")
    elif cap[0] >= 9:
        print(f"  Hopper detected — NVFP4 will run via emulation")
    else:
        print(f"  WARNING: pre-Hopper GPU, NVFP4 may not work")


@step("3. model load")
def test_model_load(torch):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  loaded {name}  ({n_params:.0f}M params, BF16)")
    return tok, model


@step("4. modelopt nvfp4")
def test_modelopt_nvfp4(torch, mtq, tok, model):
    # Use ModelOpt's canonical NVFP4 config — this is the W4A4 deployment recipe.
    NVFP4_CFG = mtq.NVFP4_DEFAULT_CFG

    # Tiny calibration: one short sequence
    ids = tok.encode("Hello world, this is a calibration sequence.",
                     return_tensors="pt").to("cuda")

    def calibration_forward(m):
        with torch.no_grad():
            m(input_ids=ids)

    mtq.quantize(model, config=NVFP4_CFG, forward_loop=calibration_forward)
    # Verify some weight quantizers were actually inserted
    n_quantizers = sum(1 for n, _ in model.named_modules() if "weight_quantizer" in n)
    print(f"  inserted {n_quantizers} weight quantizers")
    assert n_quantizers > 0, "No weight quantizers inserted — NVFP4 config didn't match any layers"
    return ids


@step("5. autoround fp4")
def test_autoround_fp4(torch, tok):
    from transformers import AutoModelForCausalLM
    from auto_round import AutoRound
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda",
    )
    cal_texts = ["Hello world, this is a short calibration sequence for AutoRound."]
    ar = AutoRound(
        model=model,
        tokenizer=tok,
        bits=4,
        group_size=16,      # NVFP4 spec: 16 weights per block (enforced by auto_round)
        iters=0,            # RTN only — fastest
        dataset=cal_texts,
        seqlen=32,          # short for smoke test
        data_type="nv_fp4", # true NVFP4 (matches ModelOpt's format exactly)
        device_map="cuda",
    )
    ar.quantize()
    print(f"  AutoRound FP4 RTN (iters=0) completed")
    del model
    torch.cuda.empty_cache()


@step("6. forward + STE backward")
def test_ste_backward(torch, model, ids):
    import torch.nn.functional as F
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Use the model itself as a fake teacher (just to get a target distribution)
    with torch.no_grad():
        teacher_logits = model(input_ids=ids).logits.detach().clamp(-100, 100)

    student_logits = model(input_ids=ids).logits.clamp(-100, 100)
    loss = F.kl_div(
        F.log_softmax(student_logits / 2.0, dim=-1),
        F.softmax(teacher_logits / 2.0, dim=-1),
        reduction="batchmean",
    ) * 4.0
    print(f"  KL loss (model vs itself): {loss.item():.6f}")
    assert not torch.isnan(loss), "loss is nan"

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print(f"  backward + step OK, grad_norm={grad_norm.item():.4f}")
    assert not torch.isnan(grad_norm), "grad norm is nan"


if __name__ == "__main__":
    torch, mtq = test_imports()
    test_cuda(torch)
    tok, model = test_model_load(torch)
    ids = test_modelopt_nvfp4(torch, mtq, tok, model)
    test_autoround_fp4(torch, tok)
    test_ste_backward(torch, model, ids)
    print("\n" + "=" * 50)
    print("All smoke tests passed.")
    print("Safe to run qad_experiment.py / combined_experiment.py.")
    print("=" * 50)
