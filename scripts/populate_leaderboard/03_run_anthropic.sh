#!/bin/bash
# 03_run_anthropic.sh - Run Anthropic agents (Claude text + vision)
#
# Prerequisites: ANTHROPIC_API_KEY in .env
# Estimated time: ~2 hours
# Estimated cost: ~$20 (depending on API pricing)
# Output: leaderboard_data/entries/claude_text_v1.json, claude_vision_v1.json

set -e  # Exit on error

echo "========================================"
echo "Running Anthropic Agents"
echo "========================================"
echo ""
echo "This script evaluates:"
echo "  - Claude Sonnet text agent (text observations)"
echo "  - Claude Sonnet vision agent (image observations)"
echo ""
echo "Time: ~2 hours"
echo "Cost: ~$20"
echo ""

# Check for API key
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Create .env with: ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

if ! grep -q "ANTHROPIC_API_KEY" .env; then
    echo "❌ Error: ANTHROPIC_API_KEY not found in .env"
    echo "Add to .env: ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

# Create output directories
mkdir -p leaderboard_data/entries
mkdir -p logs

# Timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/03_anthropic_${TIMESTAMP}.log"

echo "Logs will be saved to: $LOG_FILE"
echo ""

# Run evaluations
echo "1/2: Evaluating Claude Text agent..."
uv run agentick evaluate \
    --submission leaderboard_data/submissions/anthropic_text.yaml \
    --suite agentick-core-v1 \
    --output leaderboard_data/entries/ \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "2/2: Evaluating Claude Vision agent..."
uv run agentick evaluate \
    --submission leaderboard_data/submissions/anthropic_vision.yaml \
    --suite agentick-core-v1 \
    --output leaderboard_data/entries/ \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "========================================"
echo "✓ Anthropic evaluations complete!"
echo "========================================"
echo ""
echo "Results saved to: leaderboard_data/entries/"
ls -lh leaderboard_data/entries/claude*.json 2>/dev/null || echo "No results found"
echo ""
echo "Next step: Run 04_run_local_models.sh (requires GPU, ~4 hours)"
echo "  or skip to step 7 for leaderboard generation:"
echo "  python scripts/populate_leaderboard/07_generate_leaderboard.py"
