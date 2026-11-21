# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Set to "false" when building for 24 GB cards (L4/A10) so ComfyUI can offload
# the text encoder instead of keeping everything resident in VRAM.
ARG GPU_ONLY=true

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Performance
ENV TRITON_CACHE_DIR=/tmp/.triton-cache
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/.inductor-cache
ENV SAFETENSORS_FAST_GPU=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    openssh-server \
    build-essential \
    ninja-build \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install uv and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI (Krea 2 needs a recent version; consider pinning a known-good
# release via --build-arg COMFYUI_VERSION=x.y.z for reproducible builds)
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install \
        --version "${COMFYUI_VERSION}" \
        --cuda-version "${CUDA_VERSION_FOR_COMFY}" \
        --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install \
        --version "${COMFYUI_VERSION}" \
        --nvidia; \
    fi

# Optional PyTorch upgrade
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio \
        --index-url "${PYTORCH_INDEX_URL}"; \
    fi

# Custom nodes needed by FaceDetailer
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

RUN comfy-node-install comfyui-impact-pack comfyui-impact-subpack

# Runtime dependencies
RUN uv pip install -r /comfyui/requirements.txt \
    && for r in /comfyui/custom_nodes/*/requirements.txt; do \
         [ -f "$r" ] && uv pip install -r "$r" || true; \
       done \
    && uv pip install "transformers>=4.50.3,<5" "huggingface-hub<1.0"

# Build-time smoke test
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

WORKDIR /comfyui

# Network volume configuration
ADD src/extra_model_paths.yaml ./

WORKDIR /

# RunPod/runtime dependencies
# (Not needed if you only run on Modal, which launches main.py directly.
#  Safe to remove this block, handler.py and start.sh in that case.)
RUN uv pip install runpod requests websocket-client

# Application code
ADD src/start.sh src/network_volume.py handler.py test_input.json ./
RUN chmod +x /start.sh

# Optionally guarantee --gpu-only (controlled by the GPU_ONLY build arg)
RUN if [ "${GPU_ONLY}" = "true" ]; then \
      if grep -q -- '--gpu-only' /start.sh; then \
        echo "start.sh: --gpu-only already present, leaving untouched"; \
      else \
        sed -i 's/main\.py/main.py --gpu-only/' /start.sh; \
        echo "start.sh: injected --gpu-only"; \
      fi \
      && grep -q -- '--gpu-only' /start.sh; \
    else \
      echo "start.sh: GPU_ONLY=false, not injecting --gpu-only"; \
    fi

ENV PIP_NO_INPUT=1

# ComfyUI Manager network mode helper
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

CMD ["/start.sh"]


# ============================================================================
# Stage 2: Download Krea 2 models
# ============================================================================
FROM base AS downloader

ARG CIVITAI_TOKEN

# Hugging Face revision for Comfy-Org/Krea-2. "main" moves over time; for
# reproducible builds pass a commit hash:
#   --build-arg KREA_REV=<commit-sha>
ARG KREA_REV=main

WORKDIR /comfyui

# Directory structure (only what is actually used)
RUN mkdir -p \
    models/vae \
    models/text_encoders \
    models/diffusion_models \
    models/upscale_models \
    models/ultralytics/bbox \
    models/loras


# ============================================================================
# KREA 2 TURBO
# ============================================================================

# Krea 2 Turbo FP8
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/diffusion_models/krea2_turbo_fp8_scaled.safetensors \
      "https://huggingface.co/Comfy-Org/Krea-2/resolve/${KREA_REV}/diffusion_models/krea2_turbo_fp8_scaled.safetensors"

# Qwen3-VL 4B FP8 text encoder
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors \
      "https://huggingface.co/Comfy-Org/Krea-2/resolve/${KREA_REV}/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"

# Qwen Image VAE
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/vae/qwen_image_vae.safetensors \
      "https://huggingface.co/Comfy-Org/Krea-2/resolve/${KREA_REV}/vae/qwen_image_vae.safetensors"


# ============================================================================
# KREA 2 LoRAs
# ============================================================================

# BeMyHero - NylaX
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/BeMyHero_NylaX_Krea2.safetensors \
      "https://civitai.com/api/download/models/3206740?fileId=3088255&token=${CIVITAI_TOKEN}"

# Asian Thick Ai influencer | Consistent Character | Instagram Influencer
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/asig2_Krea2.safetensors \
      "https://civitai.com/api/download/models/3075077?fileId=2954123&token=${CIVITAI_TOKEN}"

# SNOFS
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/SNOFS_Krea2.safetensors \
      "https://civitai.com/api/download/models/3290120?fileId=3174557&token=${CIVITAI_TOKEN}"

# Bubble Butt
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/Bubble_Butt_Krea2.safetensors \
      "https://civitai.com/api/download/models/3194194?fileId=3075137&token=${CIVITAI_TOKEN}"

# Krea 2 TextFusion / refusal-reduction LoRA
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/Krea2_TextFusion_Refusal_Reduction.safetensors \
      "https://huggingface.co/Quiho/Krea2_TextFusion_Refusal-Reduction_LoRA_v1.0_lora/resolve/main/Krea2_TextFusion_Refusal_Reduction.safetensors"

# Photo/Detail Slider for SNOFS (Ashen3 v2)
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/snofs_photoSlider14B.safetensors \
      "https://civitai.com/api/download/models/3337902?token=${CIVITAI_TOKEN}"

# Candid Slider (Loraholic)
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/candid_krea2_loraholic.safetensors \
      "https://civitai.com/api/download/models/3123867?token=${CIVITAI_TOKEN}"

# Haley
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/Haley.safetensors \
      "https://civitai.com/api/download/models/3154756?fileId=3035549&token=${CIVITAI_TOKEN}"

# Anal Helper (Loraholic)
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/anal_helper_krea2_loraholic.safetensors \
      "https://civitai.com/api/download/models/3105253?fileId=2985077&token=${CIVITAI_TOKEN}"

# ============================================================================
# FaceDetailer / Detection Models
# ============================================================================

# 4x UltraSharp
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/upscale_models/4x-UltraSharp.pth \
      "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"

# YOLOv8 face detector
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/ultralytics/bbox/face_yolov8m.pt \
      "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt"

# Hand detector
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/ultralytics/bbox/hand_yolov8s.pt \
      "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt"

# Person segmentation
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/ultralytics/bbox/person_yolov8m-seg.pt \
      "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt"


# ============================================================================
# Stage 3: Final image
# ============================================================================
FROM base AS final

# Copy models into final image
COPY --from=downloader /comfyui/models /comfyui/models
