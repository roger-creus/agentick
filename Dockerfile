# ============================================================================
# Agentick — GPU-ready container with all dependencies
#
# Build:
#   docker build -t agentick:latest .
#
# Run (GPU):
#   docker run --gpus all -it agentick:latest bash
#
# Run (CPU only):
#   docker run -it agentick:latest bash
#
# Convert to Singularity:
#   singularity build agentick.sif docker://agentick:latest
#   # or from a local Docker image:
#   singularity build agentick.sif docker-daemon://agentick:latest
#
# Run with Singularity on SLURM:
#   singularity exec --nv agentick.sif uv run python -m agentick.experiments.run --config config.yaml
# ============================================================================

# --- Base: NVIDIA CUDA 12.4 + Ubuntu 22.04 ---
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python build deps
    software-properties-common \
    build-essential \
    git \
    curl \
    wget \
    # Python 3.12 (deadsnakes PPA)
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    # SDL/pygame deps (needed for gymnasium rendering)
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    # Pillow deps
    libjpeg-dev \
    libpng-dev \
    # Video encoding (for experiment video recording)
    ffmpeg \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# --- Install uv ---
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# --- Set up working directory ---
WORKDIR /opt/agentick

# --- Copy project files ---
# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock* ./

# Copy the full project
COPY . .

# --- Install all dependencies ---
# Install with all extras + dev tools
# Use system Python (no managed Python in container)
ENV UV_PYTHON=/usr/bin/python3.12
ENV UV_SYSTEM_PYTHON=1

# Sync all extras (rl, llm, vllm, finetune, tracking, viz, webapp, docs) + dev
RUN uv sync --extra all --group dev --no-progress




# --- Verify installation ---
RUN uv run python -c "\
import agentick; \
env = agentick.make('GoToGoal-v0'); \
env.reset(); \
env.close(); \
print('agentick OK'); \
import torch; \
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); \
import stable_baselines3; \
print(f'stable-baselines3 {stable_baselines3.__version__}'); \
from agentick.agents import BaseAgent, MarkovianZeroShot; \
print('agents OK'); \
from agentick.training import EvalCallback; \
print('training OK'); \
print('All checks passed!'); \
"

# --- Environment variables ---
# Make agentick importable everywhere
ENV PYTHONPATH="/opt/agentick"
# Disable pygame welcome message
ENV PYGAME_HIDE_SUPPORT_PROMPT=1
# Default to headless rendering (no display needed)
ENV SDL_VIDEODRIVER=dummy
ENV SDL_AUDIODRIVER=dummy

# --- Default entrypoint ---
# Drop into bash so users can run any command
CMD ["/bin/bash"]
