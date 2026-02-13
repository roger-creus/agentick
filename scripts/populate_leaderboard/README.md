# Leaderboard Population Scripts

These scripts run the full benchmark suite to populate the leaderboard with baseline and state-of-the-art results.

## Execution Order

Run scripts in order for complete leaderboard:

1. **01_run_baselines.sh** - Random, Greedy, Oracle agents (~30 min, no cost, no GPU)
2. **02_run_openai.sh** - GPT-4o text + vision (~$20, ~2 hours)
3. **03_run_anthropic.sh** - Claude text + vision (~$20, ~2 hours)
4. **04_run_local_models.sh** - HuggingFace models (~4 hours, requires GPU)
5. **05_train_rl.sh** - Train PPO + DQN agents (~8 hours, requires GPU)
6. **06_run_rl_eval.sh** - Evaluate trained RL agents (~30 min)
7. **07_generate_leaderboard.py** - Compile all results into leaderboard data
8. **08_generate_site.sh** - Build the static leaderboard website

Or run everything: `bash run_all.sh`

## Prerequisites

### Required
- `uv` package manager installed
- `.env` file with API keys (for scripts 2-3)
- At least 100GB free disk space (for model weights and results)

### Optional
- CUDA GPU with 8GB+ VRAM (for scripts 4-5, otherwise CPU)
- `wandb` account for tracking (optional, can disable)

## Cost Estimates

| Script | Cost | Time | GPU Required |
|--------|------|------|--------------|
| 01 - Baselines | $0 | 30 min | No |
| 02 - OpenAI | ~$20 | 2 hours | No |
| 03 - Anthropic | ~$20 | 2 hours | No |
| 04 - Local Models | $0 | 4 hours | Yes (8GB+ VRAM) |
| 05 - Train RL | $0 | 8 hours | Yes (8GB+ VRAM) |
| 06 - RL Eval | $0 | 30 min | No |

**Total**: ~$40, ~17 hours wall time (or ~$40 + GPU cloud costs)

## Output

All scripts save results to `leaderboard_data/entries/`:
```
leaderboard_data/
└── entries/
    ├── random_agent_v1.json
    ├── greedy_agent_v1.json
    ├── oracle_agent_v1.json
    ├── gpt4o_text_v1.json
    ├── gpt4o_vision_v1.json
    ├── claude_text_v1.json
    ├── claude_vision_v1.json
    ├── qwen_local_v1.json
    ├── ppo_pixels_v1.json
    └── dqn_pixels_v1.json
```

## Usage

### Run Individual Script
```bash
# Ensure dependencies installed
uv sync --extra all

# Run baseline agents (no cost, quick)
bash scripts/populate_leaderboard/01_run_baselines.sh

# Check results
ls -lh leaderboard_data/entries/
```

### Run Full Pipeline
```bash
# WARNING: This will take ~17 hours and cost ~$40
# Make sure you have .env with API keys and enough disk space

bash scripts/populate_leaderboard/run_all.sh
```

### Generate Leaderboard Site
```bash
# After running evaluations
python scripts/populate_leaderboard/07_generate_leaderboard.py
bash scripts/populate_leaderboard/08_generate_site.sh

# Preview locally
cd leaderboard_site
python -m http.server 8080
# Visit http://localhost:8080
```

## Customization

### Run on Subset of Tasks
Edit the suite in each script. Change from `agentick-core-v1` to:
- `agentick-quick-v1` - 5 representative tasks (~1/8 the time)
- `agentick-navigation-v1` - Navigation tasks only
- Custom task list in script

### Reduce Episodes
Edit `--n-episodes` in scripts (default: 100 episodes per task/difficulty)
- Quick test: `--n-episodes 10` (~1/10 the time and cost)
- For leaderboard: `--n-episodes 100` (statistically robust)

### Skip Expensive Models
Comment out model lines in scripts 02-04 to skip expensive models

## Troubleshooting

### Script fails with "No API key"
Ensure `.env` file exists in project root with:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Out of memory (GPU)
Reduce batch size or use smaller models in scripts 04-05

### Results not appearing
Check `leaderboard_data/entries/` - scripts save JSON files there
Verify scripts completed successfully (exit code 0)

### Leaderboard site empty
Run `07_generate_leaderboard.py` to compile results
Then run `08_generate_site.sh` to build HTML

## Notes

- Scripts are idempotent - safe to re-run (will skip existing results)
- Each script creates a log file in `logs/` for debugging
- GPU scripts auto-detect CUDA and fall back to CPU if unavailable
- Intermediate checkpoints saved every 10 tasks (can resume from failure)
