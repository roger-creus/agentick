#!/usr/bin/env python3
"""Generate all 24 HuggingFace LLM/VLM benchmark config YAML files.

Matrix: 4 models x 2 obs modes x 3 harness presets = 24 configs.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).parent / "configs"

MODELS = [
    {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "short": "qwen3_4b",
        "type": "llm",
        "backend": "huggingface_llm",
        "obs_variants": {
            "lang": {"observation_modes": ["language"], "render_modes": ["language"]},
            "ascii": {"observation_modes": ["ascii"], "render_modes": ["ascii"]},
        },
        "tags_base": ["llm", "qwen3-4b", "benchmark"],
        "record_videos": True,
    },
    {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "short": "qwen3_30b_a3b",
        "type": "llm",
        "backend": "huggingface_llm",
        "obs_variants": {
            "lang": {"observation_modes": ["language"], "render_modes": ["language"]},
            "ascii": {"observation_modes": ["ascii"], "render_modes": ["ascii"]},
        },
        "tags_base": ["llm", "qwen3-30b-a3b", "benchmark"],
        "record_videos": True,
    },
    {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "short": "qwen3_vl4b",
        "type": "vlm",
        "backend": "huggingface_vlm",
        "obs_variants": {
            "lang": {
                "observation_modes": ["rgb_array", "language"],
                "render_modes": ["rgb_array"],
            },
            "ascii": {
                "observation_modes": ["rgb_array", "ascii"],
                "render_modes": ["rgb_array"],
            },
        },
        "tags_base": ["vlm", "qwen3-vl-4b", "benchmark"],
        "record_videos": True,
    },
    {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "short": "qwen3_vl8b",
        "type": "vlm",
        "backend": "huggingface_vlm",
        "obs_variants": {
            "lang": {
                "observation_modes": ["rgb_array", "language"],
                "render_modes": ["rgb_array"],
            },
            "ascii": {
                "observation_modes": ["rgb_array", "ascii"],
                "render_modes": ["rgb_array"],
            },
        },
        "tags_base": ["vlm", "qwen3-vl-8b", "benchmark"],
        "record_videos": True,
    },
]

HARNESSES = [
    {
        "name": "markovian_zero_shot",
        "short": "markov",
        "max_new_tokens": 50,
    },
    {
        "name": "non_markovian_zero_shot",
        "short": "nonmarkov",
        "max_new_tokens": 50,
    },
    {
        "name": "markovian_reasoner",
        "short": "reasoner",
        "max_new_tokens": 4096,
    },
]

HARNESS_LABELS = {
    "markov": "markovian zero-shot",
    "nonmarkov": "non-markovian zero-shot",
    "reasoner": "markovian reasoner (CoT)",
}

OBS_LABELS = {"lang": "language", "ascii": "ASCII"}


def generate_config(model: dict, obs_key: str, harness: dict) -> dict:
    """Generate a single config dict."""
    obs_variant = model["obs_variants"][obs_key]
    is_vlm = model["type"] == "vlm"

    obs_label = OBS_LABELS[obs_key]
    harness_label = HARNESS_LABELS[harness["short"]]
    model_short = model["model_id"].split("/")[-1]
    config_name = f"{model['short']}_{obs_key}_{harness['short']}"

    if is_vlm:
        desc = (
            f"{model_short} VLM, {harness_label}, "
            f"pixel+{obs_label} observations"
        )
    else:
        desc = f"{model_short} LLM, {harness_label}, {obs_label} observations"

    tags = list(model["tags_base"]) + [obs_key, harness["short"]]

    output_dir = "results/vlm_benchmarks" if is_vlm else "results/llm_benchmarks"

    config = {
        "name": config_name,
        "description": desc,
        "agent": {
            "type": model["type"],
            "hyperparameters": {
                "backend": model["backend"],
                "model": model["model_id"],
                "harness": harness["name"],
                "observation_modes": obs_variant["observation_modes"],
                "device": "auto",
                "dtype": "bfloat16",
                "max_new_tokens": harness["max_new_tokens"],
                "temperature": 0.0,
            },
        },
        "tasks": "full",
        "difficulties": ["easy", "medium", "hard", "expert"],
        "training": None,
        "n_episodes": 5,
        "n_seeds": 5,
        "render_modes": obs_variant["render_modes"],
        "record_trajectories": True,
        "record_videos": model["record_videos"],
        "output_dir": output_dir,
        "tags": tags,
        "metrics": [
            "mean_return",
            "success_rate",
            "mean_length",
            "mean_latency",
            "total_tokens",
        ],
    }

    return config_name, config


def main():
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    for model in MODELS:
        for obs_key in model["obs_variants"]:
            for harness in HARNESSES:
                config_name, config = generate_config(model, obs_key, harness)
                filepath = CONFIGS_DIR / f"{config_name}.yaml"
                with open(filepath, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                generated.append(config_name)
                print(f"  {filepath.name}")

    print(f"\nGenerated {len(generated)} config files in {CONFIGS_DIR}")


if __name__ == "__main__":
    main()
