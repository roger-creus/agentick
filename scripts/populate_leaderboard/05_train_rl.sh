#!/bin/bash
# 05_train_rl.sh - Train RL agents (PPO + DQN)
#
# Prerequisites: GPU recommended (works on CPU but slower)
# Estimated time: ~8 hours
# Estimated cost: $0 (compute only)
# Output: checkpoints/ppo_leaderboard/, checkpoints/dqn_leaderboard/

set -e  # Exit on error

echo "========================================"
echo "Training RL Agents"
echo "========================================"
echo ""
echo "This script trains:"
echo "  - PPO agent (500k steps)"
echo "  - DQN agent (500k steps)"
echo ""
echo "Time: ~8 hours (4 hours per agent)"
echo "Cost: $0 (GPU compute)"
echo ""

# Check for dependencies
if ! uv run python -c "import torch; import stable_baselines3" 2>/dev/null; then
    echo "❌ Error: RL dependencies not installed"
    echo "Run: uv sync --extra rl"
    exit 1
fi

# Create output directories
mkdir -p checkpoints
mkdir -p logs

# Timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/05_train_rl_${TIMESTAMP}.log"

echo "Logs will be saved to: $LOG_FILE"
echo ""

# Train PPO
echo "1/2: Training PPO agent..."
echo "  - Total timesteps: 500,000"
echo "  - Estimated time: ~4 hours"
echo ""

uv run python examples/rl/sb3_ppo.py \
    --total-timesteps 500000 \
    --checkpoint-dir checkpoints/ppo_leaderboard \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "✓ PPO training complete!"
echo ""

# Train DQN
echo "2/2: Training DQN agent..."
echo "  - Total timesteps: 500,000"
echo "  - Estimated time: ~4 hours"
echo ""

uv run python examples/rl/sb3_dqn.py \
    --total-timesteps 500000 \
    --checkpoint-dir checkpoints/dqn_leaderboard \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "========================================"
echo "✓ RL training complete!"
echo "========================================"
echo ""
echo "Models saved to:"
ls -lh checkpoints/ppo_leaderboard/*.zip 2>/dev/null || echo "  - No PPO model found"
ls -lh checkpoints/dqn_leaderboard/*.zip 2>/dev/null || echo "  - No DQN model found"
echo ""
echo "Next step: Run 06_run_rl_eval.sh (evaluate trained agents, ~30 min)"
