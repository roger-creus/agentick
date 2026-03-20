#!/usr/bin/env python3
"""SLURM launcher for full benchmark experiments.

Splits each experiment config into one SLURM job per (task, difficulty) pair,
so all combinations run in parallel across the cluster. Use --no-split to
revert to one-job-per-config.

Usage:
    python examples/experiments/slurm/launch.py --dry-run
    python examples/experiments/slurm/launch.py
    python examples/experiments/slurm/launch.py --configs "ppo_*" --time 12:00:00
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # agentick repo root

API_KEY_VARS = [
    "OPENAI_API_KEY", "GEMINI_API_KEY", "HUGGING_FACE_HUB_TOKEN",
    "CLAUDE_API_KEY", "CLAUDE_ENDPOINT",
]

REQUIRED_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "huggingface_llm": "HUGGING_FACE_HUB_TOKEN",
    "huggingface_vlm": "HUGGING_FACE_HUB_TOKEN",
    "anthropic": "CLAUDE_API_KEY",
}

# Known API rate limits per model (free/default tier).
# {model_name: {rpm, tpm, max_concurrent_jobs}}
# max_concurrent_jobs: how many SLURM jobs can safely share the quota.
API_MODEL_LIMITS: dict[str, dict[str, int]] = {
    "gemini-2.5-flash-lite": {"rpm": 4000, "tpm": 4_000_000, "max_concurrent_jobs": 4},
    "gemini-2.5-flash": {"rpm": 1000, "tpm": 1_000_000, "max_concurrent_jobs": 2},
    "gemini-3.1-flash-lite-preview": {"rpm": 4000, "tpm": 4_000_000, "max_concurrent_jobs": 4},
}

# Suite name → task list mapping (kept in sync with agentick.leaderboard.suites)
SUITE_TASKS: dict[str, list[str]] = {}


def _load_suite_tasks() -> None:
    """Populate SUITE_TASKS by importing from agentick if available."""
    if SUITE_TASKS:
        return
    # Ensure agentick is importable even when running outside uv/conda
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from agentick.leaderboard.suites import (
            FULL_TASKS,
            GENERALIZATION_TASKS,
            MEMORY_TASKS,
            MULTIAGENT_TASKS,
            NAVIGATION_TASKS,
            PLANNING_TASKS,
            REASONING_TASKS,
        )
        from agentick.tasks.registry import list_tasks

        SUITE_TASKS["full"] = list_tasks()
        SUITE_TASKS["navigation"] = NAVIGATION_TASKS
        SUITE_TASKS["planning"] = PLANNING_TASKS
        SUITE_TASKS["reasoning"] = REASONING_TASKS
        SUITE_TASKS["memory"] = MEMORY_TASKS
        SUITE_TASKS["generalization"] = GENERALIZATION_TASKS
        SUITE_TASKS["multi_agent"] = MULTIAGENT_TASKS
        SUITE_TASKS["_full_leaderboard"] = FULL_TASKS
    except ImportError:
        print(
            "WARNING: Could not import agentick. Suite resolution (tasks: 'full') "
            "will not work. Install agentick or run with 'uv run'.",
            file=sys.stderr,
        )


def resolve_task_list(config: dict) -> list[str]:
    """Resolve the task list from a config's 'tasks' field."""
    tasks = config.get("tasks", [])
    if isinstance(tasks, list):
        return tasks
    if isinstance(tasks, str):
        _load_suite_tasks()
        if tasks in SUITE_TASKS:
            return SUITE_TASKS[tasks]
        # Might be a single task name
        if tasks.endswith("-v0"):
            return [tasks]
        print(
            f"WARNING: Unknown suite '{tasks}' and could not resolve. "
            f"Available: {list(SUITE_TASKS.keys())}",
            file=sys.stderr,
        )
        return []
    return []


def load_profiles(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_template(path: Path) -> str:
    with open(path) as f:
        return f.read()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def classify_config(config_stem: str, classification_rules: list[dict]) -> str | None:
    """Return the profile name for a config stem using first-match rules."""
    for rule in classification_rules:
        if fnmatch.fnmatch(config_stem, rule["pattern"]):
            return rule["profile"]
    return None


def detect_runner_type(config: dict) -> str:
    """Classify runner: 'ppo', 'module', or 'single'."""
    agent = config.get("agent")
    agent_type = config.get("agent_type")

    if agent and isinstance(agent, dict) and agent.get("type") == "ppo":
        if config.get("training") is not None:
            return "ppo"
    if agent_type == "ppo":
        return "ppo"
    if agent and isinstance(agent, dict) and "type" in agent:
        return "module"
    return "single"


def build_runner_command(
    runner_type: str,
    config_path: Path | str,
    task: str | None = None,
    output_dir: str | None = None,
    render_mode: str | None = None,
    difficulties: list[str] | None = None,
) -> str:
    """Build the runner command string (to be prefixed with 'uv run').

    Args:
        runner_type: 'ppo', 'module', or 'single'
        config_path: Path to the YAML config (relative to project root)
        task: If set, run only this single task
        output_dir: Optional output directory override
        render_mode: Optional render mode override (e.g. 'rgb_array')
        difficulties: Optional list of difficulties to run (PPO only)
    """
    if runner_type == "ppo":
        # PPO training: use train_and_eval_ppo.py
        cmd = f"python examples/experiments/train_and_eval_ppo.py --config {config_path}"
        if task:
            cmd += f" --tasks {task}"
        if difficulties:
            cmd += f" --difficulties {' '.join(difficulties)}"
        if output_dir:
            cmd += f" --output-dir {output_dir}"
        if render_mode:
            cmd += f" --render-mode {render_mode}"
        return cmd

    if runner_type == "module":
        # agentick.experiments.run has no --tasks CLI flag, so config_path
        # should already point to a per-task config if splitting
        cmd = f"python -m agentick.experiments.run --config {config_path}"
        if difficulties:
            cmd += f" --difficulties {' '.join(difficulties)}"
        if output_dir:
            cmd += f" --output-dir {output_dir}"
        return cmd

    # single (run_single_benchmark.py)
    cmd = f"python examples/experiments/run_single_benchmark.py {config_path}"
    if output_dir:
        cmd += f" --output-dir {output_dir}"
    return cmd


RUNNER_DISPLAY = {
    "ppo": "train_and_eval_ppo.py",
    "module": "agentick.experiments.run",
    "single": "run_single_benchmark.py",
}


def get_required_api_key(config: dict) -> str | None:
    """Return the env var name for the API key this config needs, or None."""
    agent = config.get("agent")
    if agent and isinstance(agent, dict):
        hp = agent.get("hyperparameters", {})
        backend = hp.get("backend", "")
        if backend in REQUIRED_KEYS:
            return REQUIRED_KEYS[backend]
        key_env = hp.get("api_key_env")
        if key_env:
            return key_env

    name = config.get("name", "")
    if "openai" in name.lower() or "gpt" in name.lower():
        return "OPENAI_API_KEY"
    return None


def get_api_model_limits(config: dict) -> tuple[str, dict[str, int]] | None:
    """Return (model_name, limits) if this config uses a rate-limited API model."""
    agent = config.get("agent")
    if not agent or not isinstance(agent, dict):
        return None
    hp = agent.get("hyperparameters", {})
    model = hp.get("model", "")
    if model in API_MODEL_LIMITS:
        return model, API_MODEL_LIMITS[model]
    return None


def inject_rate_limits(config: dict, rpm_limit: int, tpm_limit: int) -> dict:
    """Inject per-job rate limits into the config's agent hyperparameters."""
    config = copy.deepcopy(config)
    hp = config.get("agent", {}).get("hyperparameters", {})
    # Only inject if not already explicitly set by the user
    if "rpm_limit" not in hp:
        hp["rpm_limit"] = rpm_limit
    if "tpm_limit" not in hp:
        hp["tpm_limit"] = tpm_limit
    return config


def is_vllm_backend(config: dict) -> bool:
    """Return True if the config uses a vLLM backend (LLM or VLM)."""
    agent = config.get("agent")
    if agent and isinstance(agent, dict):
        backend = agent.get("hyperparameters", {}).get("backend", "")
        return backend in ("vllm_llm", "vllm_vlm")
    return False


def build_env_exports() -> str:
    """Build export lines for API keys that are set in the current environment."""
    lines = []
    for var in API_KEY_VARS:
        val = os.environ.get(var)
        if val:
            lines.append(f'export {var}="{val}"')
    return "\n".join(lines)


def generate_sbatch_script(
    template: str,
    *,
    job_name: str,
    partition: str,
    cpus: int,
    gres: str,
    mem: str,
    time: str,
    log_dir: str,
    modules: str,
    conda_env: str,
    env_exports: str,
    project_root: str,
    config_display: str,
    profile_name: str,
    runner_command: str,
    inline_config_yaml: str | None = None,
) -> str:
    gres_line = f"#SBATCH --gres={gres}" if gres else ""

    if inline_config_yaml:
        # Embed config YAML directly in the script so it's self-contained.
        # The heredoc writes a temp file that the runner reads.
        inline_config_block = (
            "_AGENTICK_CFG=$(mktemp /tmp/agentick_cfg_XXXXXX.yaml)\n"
            f"cat > \"$_AGENTICK_CFG\" <<'__AGENTICK_CONFIG_EOF__'\n"
            f"{inline_config_yaml}"
            "__AGENTICK_CONFIG_EOF__"
        )
        cleanup_block = 'rm -f "$_AGENTICK_CFG"'
    else:
        inline_config_block = "# Using config file directly (no inline config)"
        cleanup_block = ""

    return template.format(
        job_name=job_name,
        partition=partition,
        cpus=cpus,
        gres_line=gres_line,
        mem=mem,
        time=time,
        log_dir=log_dir,
        modules=modules,
        conda_env=conda_env,
        env_exports=env_exports,
        project_root=project_root,
        config_display=config_display,
        profile_name=profile_name,
        runner_command=runner_command,
        inline_config_block=inline_config_block,
        cleanup_block=cleanup_block,
    )


def make_per_task_config_yaml(
    base_config: dict,
    task_name: str,
    difficulty: str | None = None,
) -> str:
    """Generate per-task YAML config string derived from base_config.

    Overrides the 'tasks' field to a single-element list and optionally
    restricts difficulties to a single value.
    The config 'name' is preserved (not modified with task suffix) so that
    all per-task jobs share the same agent identity in results.
    Returns the YAML content as a string (not written to disk).
    """
    cfg = copy.deepcopy(base_config)
    cfg["tasks"] = [task_name]
    if difficulty:
        cfg["difficulties"] = [difficulty]
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False)


def resolve_configs(
    config_names: list[str] | None,
    exclude_names: list[str] | None,
) -> list[Path]:
    """Resolve config file paths from names/globs, applying exclusions."""
    all_configs = sorted(CONFIGS_DIR.glob("*.yaml"))

    if config_names:
        selected = []
        for pattern in config_names:
            if not pattern.endswith(".yaml"):
                pattern = pattern + ".yaml" if "*" not in pattern else pattern
            matches = [c for c in all_configs if fnmatch.fnmatch(c.name, pattern)]
            if not matches:
                matches = [
                    c
                    for c in all_configs
                    if fnmatch.fnmatch(c.stem, pattern.removesuffix(".yaml"))
                ]
            selected.extend(matches)
        seen = set()
        configs = []
        for c in selected:
            if c not in seen:
                seen.add(c)
                configs.append(c)
    else:
        configs = all_configs

    if exclude_names:
        for pattern in exclude_names:
            configs = [
                c
                for c in configs
                if not fnmatch.fnmatch(c.stem, pattern.removesuffix(".yaml"))
                and not fnmatch.fnmatch(c.name, pattern)
            ]

    return configs


def submit_job(script_path: Path, dependency: int | None = None) -> int | None:
    """Submit an sbatch script and return the job ID."""
    cmd = ["sbatch"]
    if dependency is not None:
        cmd.append(f"--dependency=afterany:{dependency}")
    cmd.append(str(script_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR submitting {script_path.name}: {result.stderr.strip()}")
        return None

    for word in result.stdout.strip().split():
        if word.isdigit():
            return int(word)
    print(f"  WARNING: could not parse job ID from: {result.stdout.strip()}")
    return None


# --- Manifest & relaunch ---

FAIL_STATES = {"FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "CANCELLED", "PREEMPTED"}


def save_manifest(jobs: list[dict], manifest_dir: Path, timestamp: str) -> Path:
    """Save a JSON manifest of submitted jobs for later relaunch."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for job in jobs:
        records.append({
            "config": job["config"],
            "profile": job["profile"],
            "runner": job["runner"],
            "gres": job["gres"],
            "time": job["time"],
            "script_path": str(job["script_path"]),
            "job_id": job["job_id"],
        })
    manifest = {"timestamp": timestamp, "jobs": records}

    path = manifest_dir / f"manifest_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Symlink as latest.json
    latest = manifest_dir / "latest.json"
    latest.unlink(missing_ok=True)
    latest.symlink_to(path.name)

    return path


def query_job_states(job_ids: list[int]) -> dict[int, str]:
    """Query sacct for the final state of each job ID."""
    if not job_ids:
        return {}
    ids_str = ",".join(str(j) for j in job_ids)
    result = subprocess.run(
        ["sacct", "-j", ids_str, "--format=JobID,State", "--noheader", "--parsable2"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: sacct failed: {result.stderr.strip()}")
        return {}

    states: dict[int, str] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        raw_id = parts[0].split(".")[0]  # strip ".batch" / ".extern" suffixes
        if not raw_id.isdigit():
            continue
        jid = int(raw_id)
        state = parts[1].split(" ")[0]  # strip "by <uid>" from CANCELLED
        # Keep the most specific state (prefer the batch step's non-COMPLETED state)
        if jid not in states or state != "COMPLETED":
            states[jid] = state
    return states


def do_relaunch(manifest_path: Path, max_concurrent: int) -> None:
    """Read a manifest, find failed/timed-out jobs, and resubmit them."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    records = manifest["jobs"]
    job_ids = [r["job_id"] for r in records if r["job_id"] is not None]
    states = query_job_states(job_ids)

    # Find jobs that need relaunching
    to_relaunch = []
    completed = 0
    running = 0
    pending = 0
    for rec in records:
        jid = rec["job_id"]
        if jid is None:
            to_relaunch.append(rec)  # never submitted successfully
            continue
        state = states.get(jid, "UNKNOWN")
        if state in FAIL_STATES:
            to_relaunch.append(rec)
        elif state == "COMPLETED":
            completed += 1
        elif state in ("RUNNING", "REQUEUED"):
            running += 1
        elif state in ("PENDING",):
            pending += 1
        # else: unknown state, skip

    print(f"Manifest: {manifest_path.name} ({len(records)} jobs)")
    print(f"  Completed: {completed}  |  Running: {running}  |  Pending: {pending}")
    print(f"  Failed/timed-out/crashed: {len(to_relaunch)}")

    if not to_relaunch:
        print("\nNothing to relaunch!")
        return

    print(f"\nRelaunching {len(to_relaunch)} jobs:\n")
    submitted_ids: list[int] = []

    for i, rec in enumerate(to_relaunch):
        script_path = Path(rec["script_path"])
        if not script_path.exists():
            print(f"  SKIP {rec['config']}: script {script_path} not found")
            continue

        old_state = states.get(rec["job_id"], "NEVER_SUBMITTED") if rec["job_id"] else "NEVER_SUBMITTED"

        dependency = None
        if max_concurrent > 0 and i >= max_concurrent:
            lane_prev = i - max_concurrent
            if lane_prev < len(submitted_ids) and submitted_ids[lane_prev]:
                dependency = submitted_ids[lane_prev]

        job_id = submit_job(script_path, dependency)
        rec["job_id"] = job_id  # update for new manifest
        submitted_ids.append(job_id)

        jid_str = str(job_id) if job_id else "FAILED"
        print(f"  {rec['config']:<45}  {old_state:<14} -> {jid_str}")

    # Save updated manifest
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = manifest_path.parent
    new_path = manifest_dir / f"manifest_{ts}_relaunch.json"
    manifest["timestamp"] = ts
    manifest["relaunched_from"] = manifest_path.name
    with open(new_path, "w") as f:
        json.dump(manifest, f, indent=2)
    latest = manifest_dir / "latest.json"
    latest.unlink(missing_ok=True)
    latest.symlink_to(new_path.name)

    ok = sum(1 for j in submitted_ids if j is not None)
    print(f"\nResubmitted {ok}/{len(to_relaunch)} jobs. Manifest: {new_path.name}")
    if submitted_ids:
        ids_str = " ".join(str(j) for j in submitted_ids if j is not None)
        print(f"Monitor: squeue -u $USER")
        print(f"Cancel:  scancel {ids_str}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch benchmark experiments as SLURM jobs (one job per task)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --dry-run                            Preview all jobs
  %(prog)s                                      Submit all experiments (1 job per task)
  %(prog)s --configs "ppo_*" --time 12:00:00    PPO with more time
  %(prog)s --exclude "ppo_*" "qwen*"            Skip GPU experiments
  %(prog)s --max-concurrent 50                  Limit parallel jobs
  %(prog)s --tasks SokobanPush-v0 TileSorting-v0  Only run specific tasks
  %(prog)s --no-split --configs random_agent     One job for entire config

Relaunch failed jobs:
  %(prog)s --relaunch                           Relaunch from latest manifest
  %(prog)s --relaunch path/to/manifest.json     Relaunch from specific manifest""",
    )
    parser.add_argument(
        "--configs", nargs="+", metavar="NAME",
        help="Specific configs or glob patterns (default: all)",
    )
    parser.add_argument(
        "--exclude", nargs="+", metavar="NAME",
        help="Configs to skip",
    )
    parser.add_argument("--partition", help="Override partition")
    parser.add_argument("--time", help="Override wall time (e.g. 12:00:00)")
    parser.add_argument("--mem", help="Override memory (e.g. 64G)")
    parser.add_argument("--cpus", type=int, help="Override CPUs")
    parser.add_argument("--gres", help="Override GPU spec (e.g. gpu:a100:1)")
    parser.add_argument(
        "--max-concurrent", type=int, default=0, metavar="N",
        help="Limit simultaneous jobs (default: unlimited)",
    )
    parser.add_argument("--conda-env", help="Override conda env name")
    parser.add_argument("--output-dir", help="Override results output directory")
    parser.add_argument(
        "--render-mode", help="Override render mode (e.g. rgb_array, rgb_array)",
    )
    parser.add_argument(
        "--tasks", nargs="+", metavar="TASK",
        help="Only launch these specific tasks (e.g. SokobanPush-v0 TileSorting-v0)",
    )
    parser.add_argument(
        "--difficulties", nargs="+", metavar="DIFF",
        help="Only run these difficulties (e.g. easy expert). PPO splits by difficulty.",
    )
    parser.add_argument(
        "--no-split", action="store_true",
        help="Don't split by task — one SLURM job per config (old behavior)",
    )
    parser.add_argument(
        "--relaunch", nargs="?", const="latest", metavar="MANIFEST",
        help="Relaunch failed jobs from a manifest (default: latest)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print job table and sample script without submitting",
    )
    args = parser.parse_args()

    # Handle --relaunch mode
    if args.relaunch:
        profiles_data = load_profiles(SCRIPT_DIR / "profiles.yaml")
        cluster = profiles_data["cluster"]
        manifest_dir = PROJECT_ROOT / cluster["log_dir"] / "manifests"
        if args.relaunch == "latest":
            manifest_path = manifest_dir / "latest.json"
        else:
            manifest_path = Path(args.relaunch)
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}")
            print(f"Available manifests in {manifest_dir}/:")
            if manifest_dir.exists():
                for f in sorted(manifest_dir.glob("manifest_*.json")):
                    print(f"  {f.name}")
            sys.exit(1)
        # Resolve symlinks
        manifest_path = manifest_path.resolve()
        do_relaunch(manifest_path, args.max_concurrent)
        return

    # Load profiles and template
    profiles_data = load_profiles(SCRIPT_DIR / "profiles.yaml")
    template = load_template(SCRIPT_DIR / "job_template.sh")
    cluster = profiles_data["cluster"]
    profiles = profiles_data["profiles"]
    classification = profiles_data["classification"]

    # Resolve configs
    configs = resolve_configs(args.configs, args.exclude)
    if not configs:
        print("No configs matched. Available configs:")
        for c in sorted(CONFIGS_DIR.glob("*.yaml")):
            print(f"  {c.name}")
        sys.exit(1)

    # Prepare directories
    log_dir = PROJECT_ROOT / cluster["log_dir"]
    scripts_dir = log_dir / "scripts"
    log_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    env_exports = build_env_exports()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    partition = args.partition or cluster["partition"]
    conda_env = args.conda_env or cluster["conda_env"]
    modules = " ".join(cluster["modules"])

    # --- Auto-detect API rate limits ---
    # Pre-scan configs to count total jobs sharing the same rate-limited API,
    # then auto-set max_concurrent and per-job rate limits.
    max_concurrent = args.max_concurrent
    api_per_job_rpm: int = 0
    api_per_job_tpm: int = 0

    api_total_jobs = 0
    api_limits: dict[str, int] | None = None
    api_model_name: str | None = None
    for cp in configs:
        cfg = load_config(cp)
        result = get_api_model_limits(cfg)
        if result is None:
            continue
        model_name, limits = result
        api_model_name = model_name
        api_limits = limits
        # Count how many jobs this config will produce
        if args.no_split:
            api_total_jobs += 1
        else:
            tl = resolve_task_list(cfg)
            if args.tasks and tl:
                tl = [t for t in tl if t in set(args.tasks)]
            api_total_jobs += max(len(tl), 1)

    if api_limits and api_total_jobs > 0:
        model_max_concurrent = api_limits["max_concurrent_jobs"]
        # Auto-set max_concurrent if user didn't specify
        if max_concurrent == 0:
            max_concurrent = min(api_total_jobs, model_max_concurrent)
        if max_concurrent > 0:
            effective_concurrent = min(max_concurrent, api_total_jobs)
        else:
            effective_concurrent = api_total_jobs
        # Compute per-job limits (90% of total / concurrent, for safety margin)
        api_per_job_rpm = int(api_limits["rpm"] * 0.9 / max(effective_concurrent, 1))
        api_per_job_tpm = int(api_limits["tpm"] * 0.9 / max(effective_concurrent, 1))
        print(
            f"[Rate limiting] {api_model_name}: {api_total_jobs} API jobs, "
            f"max {effective_concurrent} concurrent, "
            f"{api_per_job_rpm} RPM / {api_per_job_tpm} TPM per job"
        )

    # Build the job list
    jobs: list[dict] = []
    warnings: list[str] = []

    for config_path in configs:
        config = load_config(config_path)
        stem = config_path.stem

        # Classify profile
        profile_name = classify_config(stem, classification)
        if profile_name is None:
            warnings.append(
                f"WARNING: {config_path.name} matched no classification rule, "
                "using 'baseline' as fallback"
            )
            profile_name = "baseline"
        profile = profiles[profile_name]

        # Detect runner type
        runner_type = detect_runner_type(config)

        # Check API key
        required_key = get_required_api_key(config)
        if required_key and not os.environ.get(required_key):
            warnings.append(
                f"WARNING: {config_path.name} needs {required_key} but it is not set"
            )

        # Resolve overrides (per-profile partition overrides global)
        cpus = args.cpus if args.cpus is not None else profile["cpus"]
        gres = args.gres if args.gres is not None else profile["gres"]
        mem = args.mem or profile["mem"]
        time_limit = args.time or profile["time"]
        job_partition = args.partition or profile.get("partition") or partition

        # Determine task list for splitting
        if args.no_split:
            task_list = [None]  # None = run whole config as one job
        else:
            task_list = resolve_task_list(config)
            if not task_list:
                warnings.append(
                    f"WARNING: {config_path.name} has no resolvable tasks, "
                    "running as single job"
                )
                task_list = [None]

            # Filter to specific tasks if --tasks is provided
            if args.tasks and task_list != [None]:
                requested = set(args.tasks)
                task_list = [t for t in task_list if t in requested]
                if not task_list:
                    warnings.append(
                        f"WARNING: {config_path.name} has no matching tasks "
                        f"from --tasks filter, skipping"
                    )
                    continue

        # Determine difficulty list for per-difficulty splitting.
        # Each (task, difficulty) pair becomes a separate SLURM job.
        DEFAULT_DIFFICULTIES = ["easy", "medium", "hard", "expert"]
        if not args.no_split:
            if args.difficulties:
                difficulty_list = list(args.difficulties)
            else:
                difficulty_list = config.get("difficulties", DEFAULT_DIFFICULTIES)
        else:
            difficulty_list = [None]  # None = don't split by difficulty

        # Shared output directory for all per-task jobs of this config.
        # All tasks write into the same folder so results are in one place.
        if args.output_dir:
            shared_output = args.output_dir
        else:
            config_name = config.get("name", stem)
            base_output = config.get("output_dir", "results/ppo_benchmarks")
            shared_output = f"{base_output}/{config_name}_{timestamp}"

        for task in task_list:
          for difficulty in difficulty_list:
            # Build job name
            if task:
                task_short = task.removesuffix("-v0")
                if difficulty:
                    job_name = f"ag-{stem}-{task_short}-{difficulty}"
                else:
                    job_name = f"ag-{stem}-{task_short}"
                # Truncate job name if too long for SLURM (max ~128 chars)
                if len(job_name) > 100:
                    job_name = job_name[:100]
            else:
                job_name = f"agentick-{stem}"

            # Per-job difficulty list: single difficulty if splitting, else
            # whatever the user passed (or None for all).
            job_difficulties = [difficulty] if difficulty else args.difficulties

            # Build config path and runner command for this job.
            # Per-task configs are embedded inline in the sbatch script
            # so they're self-contained (no external file dependencies).
            inline_yaml: str | None = None

            if task is None:
                # Whole-config mode — use original config file directly
                rel_config = config_path.relative_to(PROJECT_ROOT)
                runner_cmd = build_runner_command(
                    runner_type, rel_config, output_dir=shared_output,
                    render_mode=args.render_mode, difficulties=job_difficulties,
                )
                display_config = config_path.name
            elif runner_type == "ppo":
                # PPO: use --tasks flag on original config
                rel_config = config_path.relative_to(PROJECT_ROOT)
                runner_cmd = build_runner_command(
                    runner_type, rel_config, task=task, output_dir=shared_output,
                    render_mode=args.render_mode, difficulties=job_difficulties,
                )
                diff_tag = f"/{difficulty}" if difficulty else ""
                display_config = f"{config_path.name} [{task}{diff_tag}]"
            else:
                # module / single: embed per-task config inline in the script
                # Inject auto-computed rate limits for API backends
                cfg_for_task = config
                if api_per_job_rpm > 0:
                    cfg_for_task = inject_rate_limits(config, api_per_job_rpm, api_per_job_tpm)
                inline_yaml = make_per_task_config_yaml(cfg_for_task, task, difficulty)
                # Runner will use $_AGENTICK_CFG (set by inline heredoc)
                runner_cmd = build_runner_command(
                    runner_type, "$_AGENTICK_CFG", output_dir=shared_output,
                    render_mode=args.render_mode, difficulties=job_difficulties,
                )
                diff_tag = f"/{difficulty}" if difficulty else ""
                display_config = f"{config_path.name} [{task}{diff_tag}]"

            script_content = generate_sbatch_script(
                template,
                job_name=job_name,
                partition=job_partition,
                cpus=cpus,
                gres=gres,
                mem=mem,
                time=time_limit,
                log_dir=str(log_dir),
                modules=modules,
                conda_env=conda_env,
                env_exports=env_exports,
                project_root=str(PROJECT_ROOT),
                config_display=display_config,
                profile_name=profile_name,
                runner_command=runner_cmd,
                inline_config_yaml=inline_yaml,
            )

            if task:
                task_short = task.removesuffix("-v0")
                if difficulty:
                    script_name = f"{stem}_{task_short}_{difficulty}_{timestamp}.sh"
                else:
                    script_name = f"{stem}_{task_short}_{timestamp}.sh"
            else:
                script_name = f"{stem}_{timestamp}.sh"
            script_path = scripts_dir / script_name
            script_path.write_text(script_content)
            script_path.chmod(0o755)

            jobs.append({
                "config": display_config,
                "profile": profile_name,
                "runner": RUNNER_DISPLAY[runner_type],
                "gres": gres,
                "time": time_limit,
                "script_path": script_path,
                "job_id": None,
            })

    # Print warnings (deduplicated)
    seen_warnings = set()
    for w in warnings:
        if w not in seen_warnings:
            seen_warnings.add(w)
            print(w)

    if args.dry_run:
        # Group by source config for readability
        concurrent_str = f" (max {max_concurrent} concurrent)" if max_concurrent > 0 else ""
        print(f"\n=== DRY RUN: {len(jobs)} jobs would be submitted{concurrent_str} ===\n")
        print(
            f"{'#':>4} | {'Config / Task':<45} | {'Profile':<10} | "
            f"{'Time':<9} | {'GPU':<8} | Runner"
        )
        print(
            f"{'----':>4}-+-{'-----':<45}-+-{'-------':<10}-+-"
            f"{'----':<9}-+-{'---':<8}-+--------"
        )
        for i, job in enumerate(jobs, 1):
            gpu = job["gres"] if job["gres"] else "-"
            print(
                f"{i:>4} | {job['config']:<45} | {job['profile']:<10} | "
                f"{job['time']:<9} | {gpu:<8} | {job['runner']}"
            )

        # Show a sample generated script
        print(f"\n--- Sample script ({jobs[0]['config']}) ---")
        print(jobs[0]["script_path"].read_text())
        print(f"Scripts: {scripts_dir}/")
        return

    # Submit jobs (max_concurrent was set earlier — auto-computed or from CLI)
    print(f"\nSubmitting {len(jobs)} jobs...\n")
    submitted_ids: list[int] = []

    for i, job in enumerate(jobs):
        dependency = None
        if max_concurrent > 0 and i >= max_concurrent:
            lane_prev = i - max_concurrent
            prev_id = jobs[lane_prev].get("job_id")
            if prev_id is not None:
                dependency = prev_id

        job_id = submit_job(job["script_path"], dependency)
        job["job_id"] = job_id
        if job_id is not None:
            submitted_ids.append(job_id)

    # Save manifest for relaunch
    manifest_dir = log_dir / "manifests"
    manifest_path = save_manifest(jobs, manifest_dir, timestamp)

    # Summary table
    success_count = sum(1 for j in jobs if j["job_id"] is not None)
    print(f"\nSubmitted {success_count}/{len(jobs)} jobs:")
    print(
        f"{'#':>4} | {'Config / Task':<45} | {'Profile':<10} | "
        f"{'Job ID':<8} | {'Time':<9} | GPU"
    )
    print(
        f"{'----':>4}-+-{'-----':<45}-+-{'-------':<10}-+-"
        f"{'------':<8}-+-{'----':<9}-+-----"
    )
    for i, job in enumerate(jobs, 1):
        job_id = str(job["job_id"]) if job["job_id"] else "FAILED"
        gpu = job["gres"] if job["gres"] else "-"
        print(
            f"{i:>4} | {job['config']:<45} | {job['profile']:<10} | "
            f"{job_id:<8} | {job['time']:<9} | {gpu}"
        )

    if submitted_ids:
        ids_str = " ".join(str(j) for j in submitted_ids)
        print(f"\nMonitor:  squeue -u $USER")
        print(f"Cancel:   scancel {ids_str}")
        print(f"Relaunch: python examples/experiments/slurm/launch.py --relaunch")
        print(f"Manifest: {manifest_path}")
        print(f"Scripts:  {scripts_dir}/")
        print(f"Logs:     {log_dir}/")


if __name__ == "__main__":
    main()
