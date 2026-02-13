#!/bin/bash
# 02_run_openai.sh - Run OpenAI agents (GPT-4o text + vision)
#
# Prerequisites: OPENAI_API_KEY in .env
# Estimated time: ~2 hours
# Estimated cost: ~$20 (depending on API pricing)
# Output: leaderboard_data/entries/gpt4o_text_v1.json, gpt4o_vision_v1.json

set -e  # Exit on error

echo "========================================"
echo "Running OpenAI Agents"
echo "========================================"
echo ""
echo "This script evaluates:"
echo "  - GPT-4o text agent (text observations)"
echo "  - GPT-4o vision agent (image observations)"
echo ""
echo "Time: ~2 hours"
echo "Cost: ~$20"
echo ""

# Check for API key
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Create .env with: OPENAI_API_KEY=sk-..."
    exit 1
fi

if ! grep -q "OPENAI_API_KEY" .env; then
    echo "❌ Error: OPENAI_API_KEY not found in .env"
    echo "Add to .env: OPENAI_API_KEY=sk-..."
    exit 1
fi

# Create output directories
mkdir -p leaderboard_data/entries
mkdir -p logs

# Timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/02_openai_${TIMESTAMP}.log"

echo "Logs will be saved to: $LOG_FILE"
echo ""

# Run evaluations
echo "1/2: Evaluating GPT-4o Text agent..."
uv run agentick evaluate \
    --submission leaderboard_data/submissions/openai_text.yaml \
    --suite agentick-core-v1 \
    --output leaderboard_data/entries/ \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "2/2: Evaluating GPT-4o Vision agent..."
uv run agentick evaluate \
    --submission leaderboard_data/submissions/openai_vision.yaml \
    --suite agentick-core-v1 \
    --output leaderboard_data/entries/ \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "========================================"
echo "✓ OpenAI evaluations complete!"
echo "========================================"
echo ""
echo "Results saved to: leaderboard_data/entries/"
ls -lh leaderboard_data/entries/gpt4o*.json 2>/dev/null || echo "No results found"
echo ""
echo "Next step: Run 03_run_anthropic.sh (requires ANTHROPIC_API_KEY, ~$20)"
echo "  or skip to step 7 for leaderboard generation:"
echo "  python scripts/populate_leaderboard/07_generate_leaderboard.py"
