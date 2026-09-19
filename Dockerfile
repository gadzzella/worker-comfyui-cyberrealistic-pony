# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

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

# Install ComfyUI
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
RUN uv pip install runpod requests websocket-client

# Application code
ADD src/start.sh src/network_volume.py handler.py test_input.json ./
RUN chmod +x /start.sh

# Guarantee --gpu-only
RUN if grep -q -- '--gpu-only' /start.sh; then \
      echo "start.sh: --gpu-only already present, leaving untouched"; \
    else \
      sed -i 's/main\.py/main.py --gpu-only/' /start.sh; \
      echo "start.sh: injected --gpu-only"; \
    fi \
    && grep -q -- '--gpu-only' /start.sh

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

WORKDIR /comfyui

# Krea 2 directory structure
RUN mkdir -p \
    models/checkpoints \
    models/vae \
    models/unet \
    models/clip \
    models/text_encoders \
    models/diffusion_models \
    models/upscale_models \
    models/ultralytics/bbox \
    models/loras


# ============================================================================
# KREA 2 TURBO
#
# 48 GB GPU:
# - Use the official ComfyUI FP8-scaled Turbo model.
# - It is ~13.1 GB and leaves substantial VRAM for the Qwen encoder,
#   VAE, LoRAs, ComfyUI, and FaceDetailer.
#
# Do NOT download FLUX.1-dev, CLIP-L, T5-XXL, or the FLUX VAE.
# ============================================================================

# Krea 2 Turbo FP8
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/diffusion_models/krea2_turbo_fp8_scaled.safetensors \
      "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors"

# Qwen3-VL 4B FP8 text encoder
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors \
      "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"

# Qwen Image VAE
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/vae/qwen_image_vae.safetensors \
      "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors"


# ============================================================================
# KREA 2 LoRAs
#
# These are kept from the original project because they were specifically
# selected for the Krea 2 workflow.
# ============================================================================

# Realistic Snapshot Krea 2
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/RealisticSnapshotKrea2.safetensors \
      "https://civitai.com/api/download/models/3084537?fileId=2963911&token=${CIVITAI_TOKEN}"

# BeMyHero - NylaX
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/BeMyHero_NylaX_Krea2.safetensors \
      "https://civitai.com/api/download/models/3206740?fileId=3088255&token=${CIVITAI_TOKEN}"

# RLY Thot Shot - Aspen
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/RLY_Thot_Shot_Aspen_Krea2.safetensors \
      "https://civitai.com/api/download/models/3071791?fileId=2951279&token=${CIVITAI_TOKEN}"

# SNOFS
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/SNOFS_Krea2.safetensors \
      "https://civitai.com/api/download/models/3290120?fileId=3174557&token=${CIVITAI_TOKEN}"

# Ass & Thighs Slider
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/Ass_Thighs_Slider_Krea2.safetensors \
      "https://civitai.red/api/download/models/3072964?fileId=2951934&token=${CIVITAI_TOKEN}"

# Krea 2 TextFusion / refusal-reduction LoRA
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/Krea2_TextFusion_Refusal_Reduction.safetensors \
      "https://civitai.com/api/download/models/3125118?token=${CIVITAI_TOKEN}"


# ============================================================================
# OPTIONAL OFFICIAL KREA 2 STYLE LoRAs
#
# Uncomment any of these if you want the official Krea 2 artistic styles.
# ============================================================================

# RUN curl -f --retry 3 --retry-delay 5 -L \
#       -o models/loras/krea2_darkbrush.safetensors \
#       "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_darkbrush.safetensors"

# RUN curl -f --retry 3 --retry-delay 5 -L \
#       -o models/loras/krea2_dotmatrix.safetensors \
#       "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_dotmatrix.safetensors"

# RUN curl -f --retry 3 --retry-delay 5 -L \
#       -o models/loras/krea2_kidsdrawing.safetensors \
#       "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_kidsdrawing.safetensors"

# RUN curl -f --retry 3 --retry-delay 5 -L \
#       -o models/loras/krea2_neondrip.safetensors \
#       "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_neondrip.safetensors"

# RUN curl -f --retry 3 --retry-delay 5 -L \
#       -o models/loras/krea2_rainywindow.safetensors \
#       "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_rainywindow.safetensors"

# RUN curl -f --retry 3 --retry-delay 5 -L \
#       -o models/loras/krea2_retroanime.safetensors \
#       "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_retroanime.safetensors"

# RUN curl -f --retry 3 --retry-delay 5 -L \
#       -o models/loras/krea2_softwatercolor.safetensors \
#       "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_softwatercolor.safetensors"

# RUN curl -f --retry 3 --retry-delay 5 -L \
#       -o models/loras/krea2_sunsetblur.safetensors \
#       "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_sunsetblur.safetensors"

# RUN curl -f --retry 3 --retry-delay 5 -L \
#       -o models/loras/krea2_vintagetarot.safetensors \
#       "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_vintagetarot.safetensors"


# ============================================================================
# FaceDetailer / Hi-Res Upscale
# ============================================================================

# 4x UltraSharp
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/upscale_models/4x-UltraSharp.pth \
      "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"

# YOLOv8 face detector for FaceDetailer
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/ultralytics/bbox/face_yolov8m.pt \
      "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt"


# ============================================================================
# Stage 3: Final image
# ============================================================================
FROM base AS final

# Copy Krea 2 models into final image
COPY --from=downloader /comfyui/models /comfyui/models
