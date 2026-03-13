"""Merge LoRA adapters into base model and push to HuggingFace Hub.

Standalone script that runs AFTER sft_with_trl.py training completes.
Handles VL-first models (e.g. Qwen3.5) by repackaging the merged checkpoint
into VL format so vLLM can load it identically to the base model.

Usage:
    python examples/data_and_finetuning/merge_and_push.py \
        --base-model Qwen/Qwen3.5-4B \
        --adapter-dir $SCRATCH/qwen35-4b-sft-ascii \
        --push-to-hub rogercc/agentick-qwen35-4b-sft-ascii

    # With explicit dtype
    python examples/data_and_finetuning/merge_and_push.py \
        --base-model Qwen/Qwen3.5-4B \
        --adapter-dir $SCRATCH/qwen35-4b-sft-ascii \
        --push-to-hub rogercc/agentick-qwen35-4b-sft-ascii \
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _fix_adapter_key_mismatch(adapter_dir: Path, base_model) -> None:
    """Detect and fix key prefix mismatch between adapter weights and base model.

    When SFT trains with accelerate/DDP on VL-first models (e.g. Qwen3.5),
    the adapter weight keys may include an extra 'language_model.' segment
    that doesn't exist in the base model loaded via AutoModelForCausalLM.
    This silently prevents PeftModel from loading the trained weights.

    If a mismatch is detected, this rewrites the adapter_model.safetensors
    in-place with corrected keys.
    """
    from safetensors.torch import load_file, save_file

    adapter_weights_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights_path.exists():
        adapter_weights_path = adapter_dir / "adapter_model.bin"
        if not adapter_weights_path.exists():
            return  # nothing to fix

    # Get base model key prefixes (strip trailing .weight/.bias)
    base_keys = set(base_model.state_dict().keys())
    # Build set of module paths from base model (e.g. "model.layers.0.mlp.gate_proj")
    base_modules = {k.rsplit(".", 1)[0] for k in base_keys if "." in k}

    # Load adapter keys and extract module paths
    # Adapter keys look like: base_model.model.<module_path>.lora_A.weight
    peft_prefix = "base_model.model."
    if str(adapter_weights_path).endswith(".safetensors"):
        adapter_state = load_file(str(adapter_weights_path))
    else:
        import torch
        adapter_state = torch.load(str(adapter_weights_path), map_location="cpu")

    # Extract unique module paths from adapter keys
    adapter_modules = set()
    for k in adapter_state:
        if k.startswith(peft_prefix):
            # Strip "base_model.model." and ".lora_A.weight" / ".lora_B.weight"
            rest = k[len(peft_prefix):]
            # Remove .lora_A.weight, .lora_B.weight, .lora_A.default.weight, etc.
            for suffix in (".lora_A.weight", ".lora_B.weight",
                           ".lora_A.default.weight", ".lora_B.default.weight"):
                if rest.endswith(suffix):
                    module_path = rest[: -len(suffix)]
                    adapter_modules.add(module_path)
                    break

    if not adapter_modules:
        return

    # Check if adapter modules match base model modules
    matched = adapter_modules & base_modules
    unmatched = adapter_modules - base_modules
    if not unmatched:
        print("  Adapter keys match base model — no remapping needed.")
        return

    print(f"  Key mismatch detected: {len(unmatched)}/{len(adapter_modules)} "
          f"adapter modules not found in base model.")

    # Try to find a common prefix/segment to strip or add
    # Typical case: adapter has "model.language_model.layers.X..." but
    # base has "model.layers.X..."
    remap = _find_key_remap(unmatched, base_modules)
    if remap is None:
        print("  WARNING: Could not auto-detect key remapping. "
              "Merge may fail.")
        return

    old_segment, new_segment = remap
    print(f"  Remapping adapter keys: '{old_segment}' -> '{new_segment}'")

    # Rewrite adapter weights with corrected keys
    remapped_state = {}
    n_remapped = 0
    for k, v in adapter_state.items():
        if k.startswith(peft_prefix):
            rest = k[len(peft_prefix):]
            if old_segment in rest:
                new_rest = rest.replace(old_segment, new_segment, 1)
                remapped_state[peft_prefix + new_rest] = v
                n_remapped += 1
                continue
        remapped_state[k] = v

    print(f"  Remapped {n_remapped}/{len(adapter_state)} adapter weight keys")

    # Save remapped adapter weights (overwrite in place)
    if str(adapter_weights_path).endswith(".safetensors"):
        save_file(remapped_state, str(adapter_weights_path))
    else:
        import torch
        torch.save(remapped_state, str(adapter_weights_path))
    print(f"  Saved remapped adapter to {adapter_weights_path}")


def _find_key_remap(
    unmatched_adapter: set[str],
    base_modules: set[str],
) -> tuple[str, str] | None:
    """Find a string replacement that maps adapter module paths to base module paths.

    Tries common patterns like stripping 'language_model.' from the path.
    Only validates that keys containing the old_segment map correctly after
    replacement — keys that don't contain the segment (e.g. visual encoder
    modules absent from a CausalLM base model) are ignored, as PeftModel
    will safely skip them.
    """
    # Common remap patterns to try (old_segment -> new_segment)
    candidates = [
        ("model.language_model.", "model."),
        ("model.language_model.model.", "model."),
        ("language_model.", ""),
    ]

    for old_seg, new_seg in candidates:
        # Only validate keys that actually contain the segment
        affected = {m for m in unmatched_adapter if old_seg in m}
        if not affected:
            continue
        remapped = {m.replace(old_seg, new_seg, 1) for m in affected}
        if remapped <= base_modules:
            return (old_seg, new_seg)

    # Try reverse direction: base has the longer path
    for new_seg, old_seg in candidates:
        affected = {m for m in unmatched_adapter if old_seg in m}
        if not affected:
            continue
        remapped = {m.replace(old_seg, new_seg, 1) for m in affected}
        if remapped <= base_modules:
            return (old_seg, new_seg)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters and push to HuggingFace Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-model", required=True,
        help="Base model name on HuggingFace (e.g. Qwen/Qwen3.5-4B)",
    )
    parser.add_argument(
        "--adapter-dir", required=True,
        help="Directory containing adapter_config.json + adapter_model.safetensors",
    )
    parser.add_argument(
        "--push-to-hub", required=True, metavar="REPO_ID",
        help="HuggingFace repo to push merged model to",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16"],
        help="Dtype for loading the base model",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save merged model (default: <adapter-dir>/merged)",
    )
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    merged_dir = Path(args.output_dir) if args.output_dir else adapter_dir / "merged"

    # -------------------------------------------------------------------------
    # Step 1: Verify adapter files exist
    # -------------------------------------------------------------------------
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"No adapter_config.json in {adapter_dir} — "
            "check that sft_with_trl.py training completed successfully."
        )
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.exists():
        adapter_weights = adapter_dir / "adapter_model.bin"
        if not adapter_weights.exists():
            raise FileNotFoundError(
                f"No adapter weights in {adapter_dir} — "
                "expected adapter_model.safetensors or adapter_model.bin"
            )
    print(f"Found adapter files in {adapter_dir}")
    print(f"  config:  {adapter_config}")
    print(f"  weights: {adapter_weights}")

    # -------------------------------------------------------------------------
    # Step 2: Load base model + LoRA adapters, merge
    # -------------------------------------------------------------------------
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"\nLoading base model: {args.base_model} (dtype={args.dtype})")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # ---- Detect and fix adapter key mismatch ----
    # SFT training (via accelerate + TRL) may produce adapter weight keys with
    # a different module prefix than the base model loaded here.  For example,
    # Qwen3.5 VL-first models: training produces keys like
    #   base_model.model.model.language_model.layers.0.*.lora_A.weight
    # but AutoModelForCausalLM gives keys like
    #   model.layers.0.*
    # PeftModel.from_pretrained silently ignores the mismatch, leaving LoRA
    # layers at their init values (lora_B=0) and merge produces zero changes.
    _fix_adapter_key_mismatch(adapter_dir, base_model)

    # Snapshot base weights for later verification
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    print(f"Loading LoRA adapters from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("Merging adapters (safe_merge=True)...")
    model = model.merge_and_unload(safe_merge=True)

    # Verify model type is no longer PeftModel
    model_type = type(model).__name__
    print(f"  Model type after merge: {model_type}")
    if "Peft" in model_type:
        raise RuntimeError(
            f"merge_and_unload() returned {model_type} instead of a base model class — "
            "merge failed."
        )

    # Verify no LoRA keys remain
    lora_keys = [k for k in model.state_dict() if "lora" in k.lower()]
    if lora_keys:
        raise RuntimeError(
            f"Found {len(lora_keys)} leftover LoRA keys in merged model: "
            f"{lora_keys[:5]}..."
        )

    # Verify weights actually changed
    merged_state = model.state_dict()
    n_changed = 0
    n_total = 0
    for key in merged_state:
        if key in base_state:
            n_total += 1
            if not torch.equal(merged_state[key], base_state[key]):
                n_changed += 1
    del base_state

    print(f"  Weight verification: {n_changed}/{n_total} weights differ from base")
    if n_changed == 0:
        raise RuntimeError(
            "LoRA merge produced ZERO weight changes — merged model is identical "
            "to the base model. Training may not have updated adapter weights."
        )

    # -------------------------------------------------------------------------
    # Step 3: Save merged model
    # -------------------------------------------------------------------------
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving merged model to: {merged_dir}")
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    del model

    # Remove leftover adapter files that save_pretrained may have written
    # (known PEFT issue: these cause auto-reattachment on load)
    for leftover in ["adapter_config.json", "adapter_model.safetensors",
                     "adapter_model.bin"]:
        p = merged_dir / leftover
        if p.exists():
            p.unlink()
            print(f"  Removed leftover {leftover}")

    # -------------------------------------------------------------------------
    # Step 4: VL repackaging (for VL-first models like Qwen3.5)
    # -------------------------------------------------------------------------
    if _is_vl_model(args.base_model):
        print("\nBase model is VL-first — repackaging for vLLM compatibility...")
        _repackage_as_vl_checkpoint(args.base_model, merged_dir)
    else:
        print("\nBase model is text-only — no VL repackaging needed.")

    # -------------------------------------------------------------------------
    # Step 5: Upload to HuggingFace Hub
    # -------------------------------------------------------------------------
    print(f"\nUploading to HuggingFace Hub: {args.push_to_hub}")
    _upload_to_hub(args.base_model, args.push_to_hub, merged_dir)

    print(f"\nDone! https://huggingface.co/{args.push_to_hub}")


def _is_vl_model(model_name: str) -> bool:
    """Check if a model is VL-first (has vision_config in its config)."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return hasattr(config, "vision_config") and config.vision_config is not None


def _repackage_as_vl_checkpoint(base_model_id: str, merged_dir: Path) -> None:
    """Convert merged CausalLM checkpoint to VL-format for vLLM.

    VL-first models (e.g. Qwen3.5) are only registered in vLLM under their VL
    architecture (Qwen3_5ForConditionalGeneration). SFT trains the text backbone
    via AutoModelForCausalLM, producing CausalLM-format weights that vLLM can't
    load. This repackages the checkpoint to match the base VL model layout:

    1. Remap weight keys: model.* -> model.language_model.model.*
                          lm_head.* -> model.language_model.lm_head.*
    2. Add vision encoder weights from the base VL model
    3. Replace config.json + processor configs with base model's
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    # Step 1: Load merged CausalLM weights and remap keys
    print("  Step 1/3: Remapping CausalLM weight keys to VL format...")
    safetensors_files = sorted(merged_dir.glob("*.safetensors"))
    causal_state = {}
    for sf in safetensors_files:
        causal_state.update(load_file(str(sf)))

    vl_state = {}
    for key, tensor in causal_state.items():
        # model.layers.X -> model.language_model.model.layers.X
        # lm_head.weight  -> model.language_model.lm_head.weight
        vl_state["model.language_model." + key] = tensor
    text_count = len(vl_state)
    del causal_state
    print(f"    Remapped {text_count} text weight keys")

    # Step 2: Add vision weights from base model
    print("  Step 2/3: Adding vision weights from base model...")
    base_dir = Path(snapshot_download(
        base_model_id,
        allow_patterns=["*.safetensors", "config.json", "preprocessor_config.json",
                        "video_preprocessor_config.json"],
    ))
    vision_count = 0
    for sf in sorted(base_dir.glob("*.safetensors")):
        shard = load_file(str(sf))
        for key, tensor in shard.items():
            if not key.startswith("model.language_model."):
                vl_state[key] = tensor
                vision_count += 1
        del shard
    print(f"    Added {vision_count} vision weight keys")
    print(f"    Total: {len(vl_state)} keys")

    # Step 3: Save combined checkpoint + copy base model configs
    print("  Step 3/3: Saving VL-format checkpoint...")
    for sf in safetensors_files:
        sf.unlink()
    index_file = merged_dir / "model.safetensors.index.json"
    if index_file.exists():
        index_file.unlink()

    save_file(vl_state, str(merged_dir / "model.safetensors"))
    del vl_state

    for cfg_file in ["config.json", "preprocessor_config.json",
                     "video_preprocessor_config.json"]:
        src = base_dir / cfg_file
        if src.exists():
            shutil.copy2(str(src), str(merged_dir / cfg_file))
    print("    Copied config.json and processor configs from base VL model")


def _upload_to_hub(base_model: str, repo_id: str, merged_dir: Path) -> None:
    """Upload merged model folder and copy tokenizer from base model."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)

    api.upload_folder(
        folder_path=str(merged_dir),
        repo_id=repo_id,
        commit_message="Upload SFT merged model",
    )

    # Copy tokenizer and generation config from base model
    for fname in ["tokenizer.json", "tokenizer_config.json",
                  "chat_template.jinja", "generation_config.json"]:
        try:
            src = hf_hub_download(base_model, fname)
            api.upload_file(
                path_or_fileobj=src,
                path_in_repo=fname,
                repo_id=repo_id,
                commit_message=f"Copy {fname} from base model {base_model}",
            )
            print(f"  Copied {fname} from {base_model}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
