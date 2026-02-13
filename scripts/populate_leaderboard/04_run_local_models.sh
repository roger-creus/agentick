#!/bin/bash
# 04_run_local_models.sh - Run local HuggingFace models on GPU
#
# Prerequisites: GPU with 16GB+ VRAM, HF_TOKEN in .env (optional)
# Estimated time: ~4 hours
# Estimated cost: $0 (compute only)
# Output: leaderboard_data/entries/llama_3_8b_v1.json, qwen_7b_v1.json

set -e  # Exit on error

echo "========================================"
echo "Running Local HuggingFace Models"
echo "========================================"
echo ""
echo "This script evaluates:"
echo "  - Llama 3 8B (text observations)"
echo "  - Qwen 7B (text observations)"
echo ""
echo "Time: ~4 hours"
echo "Cost: $0 (GPU compute)"
echo ""
echo "⚠️  Requires:"
echo "  - GPU with 16GB+ VRAM"
echo "  - Models will be downloaded (~15GB)"
echo ""

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Create output directories
mkdir -p leaderboard_data/entries
mkdir -p logs

# Timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/04_local_models_${TIMESTAMP}.log"

echo "Logs will be saved to: $LOG_FILE"
echo ""

# Run evaluations
echo "1/2: Evaluating Llama 3 8B..."
uv run agentick evaluate \
    --submission leaderboard_data/submissions/llama3_8b.yaml \
    --suite agentick-core-v1 \
    --output leaderboard_data/entries/ \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "2/2: Evaluating Qwen 7B..."
uv run agentick evaluate \
    --submission leaderboard_data/submissions/qwen_7b.yaml \
    --suite agentick-core-v1 \
    --output leaderboard_data/entries/ \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "========================================"
echo "✓ Local model evaluations complete!"
echo "========================================"
echo ""
echo "Results saved to: leaderboard_data/entries/"
ls -lh leaderboard_data/entries/*_8b*.json leaderboard_data/entries/*_7b*.json 2>/dev/null || echo "No results found"
echo ""
echo "Next step: Run 05_train_rl.sh (train RL agents, ~8 hours)"
echo "  or skip to step 7 for leaderboard generation:"
echo "  python scripts/populate_leaderboard/07_generate_leaderboard.py"
