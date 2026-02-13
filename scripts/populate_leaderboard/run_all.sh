#!/bin/bash
# run_all.sh - Run complete leaderboard population pipeline
#
# WARNING: This takes ~17 hours and costs ~$40
# Make sure you have:
# - .env file with OPENAI_API_KEY and ANTHROPIC_API_KEY
# - GPU with 8GB+ VRAM (or cloud GPU access)
# - 100GB+ free disk space

set -e

echo "========================================"
echo "Complete Leaderboard Population"
echo "========================================"
echo ""
echo "This will run ALL evaluation scripts:"
echo "  1. Baselines (~30 min, $0)"
echo "  2. OpenAI (~2 hours, ~$20)"
echo "  3. Anthropic (~2 hours, ~$20)"
echo "  4. Local Models (~4 hours, GPU)"
echo "  5. Train RL (~8 hours, GPU)"
echo "  6. Eval RL (~30 min)"
echo "  7. Generate Data"
echo "  8. Build Site"
echo ""
echo "Total: ~17 hours, ~$40 + GPU costs"
echo ""

read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Set start time
START_TIME=$(date +%s)

# Run all scripts
echo "Starting pipeline..."
echo ""

bash scripts/populate_leaderboard/01_run_baselines.sh
bash scripts/populate_leaderboard/02_run_openai.sh
bash scripts/populate_leaderboard/03_run_anthropic.sh
bash scripts/populate_leaderboard/04_run_local_models.sh
bash scripts/populate_leaderboard/05_train_rl.sh
bash scripts/populate_leaderboard/06_run_rl_eval.sh

python scripts/populate_leaderboard/07_generate_leaderboard.py
bash scripts/populate_leaderboard/08_generate_site.sh

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "========================================"
echo "✓ Complete pipeline finished!"
echo "========================================"
echo ""
echo "Time elapsed: ${HOURS}h ${MINUTES}m"
echo "Results: leaderboard_data/entries/"
echo "Leaderboard site: leaderboard_site/"
echo ""
echo "To preview leaderboard:"
echo "  cd leaderboard_site && python -m http.server 8080"
echo "  Visit: http://localhost:8080"
