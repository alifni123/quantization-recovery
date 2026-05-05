FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# nvidia-modelopt[hf] pins transformers to a tested version (avoids "transformers
# X is not tested with nvidia-modelopt" warnings + potential ModelOpt key/attr errors).
# bitsandbytes>=0.43.0 ships pre-built wheels for CUDA 12.x — needed because the
# base image is the runtime variant (no nvcc for source builds).
RUN pip install peft auto-round "nvidia-modelopt[hf]" "bitsandbytes>=0.43.0" -q

WORKDIR /workspace
