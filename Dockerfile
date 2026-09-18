# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

# Build arguments for this stage with sensible defaults for standalone builds
ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# ---------------------------------------------------------------------------
# Performance precautions
# ---------------------------------------------------------------------------
ENV TRITON_CACHE_DIR=/tmp/.triton-cache
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/.inductor-cache
ENV SAFETENSORS_FAST_GPU=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    wget \
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
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli + dependencies needed by it to install ComfyUI
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Upgrade PyTorch if needed (for newer CUDA versions)
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

# Install custom nodes needed for FaceDetailer
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install
RUN comfy-node-install comfyui-impact-pack comfyui-impact-subpack

# Install runtime dependencies for ComfyUI and custom nodes
RUN uv pip install -r /comfyui/requirements.txt \
    && for r in /comfyui/custom_nodes/*/requirements.txt; do \
         [ -f "$r" ] && uv pip install -r "$r" || true; \
       done \
    && uv pip install "transformers>=4.50.3,<5" "huggingface-hub<1.0"

# Build-time smoke test
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client

# Add application code and scripts
ADD src/start.sh src/network_volume.py handler.py test_input.json ./
RUN chmod +x /start.sh

# Guarantee ComfyUI is launched with --gpu-only
RUN if grep -q -- '--gpu-only' /start.sh; then \
      echo "start.sh: --gpu-only already present, leaving untouched"; \
    else \
      sed -i 's/main\.py/main.py --gpu-only/' /start.sh; \
      echo "start.sh: injected --gpu-only"; \
    fi \
    && grep -q -- '--gpu-only' /start.sh

ENV PIP_NO_INPUT=1

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG CIVITAI_TOKEN

# Install curl
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /comfyui

# Create necessary directories
RUN mkdir -p models/checkpoints models/vae models/unet models/clip models/text_encoders models/diffusion_models models/upscale_models models/ultralytics/bbox models/loras

# -------------------------------------------------------------
# Krea 2 Turbo FP8 / Native Standard Checkpoints
# -------------------------------------------------------------
# Krea 2 Turbo Diffusion Model (FP8 precision for high quality & smooth VRAM usage)
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" \
      -o models/diffusion_models/krea-2-turbo-fp8.safetensors \
      "https://huggingface.co/Comfy-Org/Krea-2-Turbo/resolve/main/split_files/diffusion_models/krea-2-turbo-fp8.safetensors" || \
    curl -f --retry 3 --retry-delay 5 -L \
      -o models/checkpoints/krea-2-turbo-fp8.safetensors \
      "https://huggingface.co/Comfy-Org/Krea-2-Turbo/resolve/main/krea-2-turbo-fp8.safetensors"

# High-Precision Text Encoder (Qwen3-VL 4B / T5-XXL standard for Krea 2)
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/text_encoders/qwen_3_4b_fp8.safetensors \
      "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"

# Official VAE
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/vae/ae.safetensors \
      "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors"

# -------------------------------------------------------------
# Krea 2 LoRAs (Fully compatible with FP8 standard weights)
# -------------------------------------------------------------
# Realism / snapshot style
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

# Refusal / censorship reduction
RUN curl -f --retry 3 --retry-delay 5 -L \
      --header "User-Agent: Mozilla/5.0" \
      -o models/loras/Krea2_TextFusion_Refusal_Reduction.safetensors \
      "https://civitai.com/api/download/models/3125118?token=${CIVITAI_TOKEN}"

# Upscale model for FaceDetailer / hires pass
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/upscale_models/4x-UltraSharp.pth \
      "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"

# Face detection model for FaceDetailer
RUN curl -f --retry 3 --retry-delay 5 -L \
      -o models/ultralytics/bbox/face_yolov8m.pt \
      "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt"

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models
