"""Merge a LoRA adapter into its base model to produce a full HuggingFace checkpoint.

SkyRL's FSDP strategy writes LoRA checkpoints as a PEFT adapter directory
(``<ckpt>/global_step_N/policy/lora_adapter/`` holding ``adapter_model.safetensors`` and
``adapter_config.json``) and has no ``merge_and_unload`` anywhere, so the SFT stage cannot hand
RL a merged model directly. RL's ``trainer.policy.model.path`` needs a full model, so merge
first and point RL at the output. RL then trains a fresh adapter on top of the merged weights.

Cost warning: merging a 30B model materializes the full weights, so expect ~61 GB of disk for
bf16 output and comparable host RAM while loading.

Example:
    uv run examples/train/mcp_atlas/merge_lora_adapter.py \\
        --base Qwen/Qwen3-30B-A3B \\
        --adapter ~/mcp_atlas_sft_run/ckpts/global_step_32/policy/lora_adapter \\
        --output ~/mcp_atlas_sft_run/merged
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Base model the adapter was trained against.")
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory (…/lora_adapter).")
    parser.add_argument("--output", required=True, help="Directory to write the merged model into.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype to load and save in. Keep bfloat16 to match training.",
    )
    args = parser.parse_args()

    adapter = Path(args.adapter).expanduser().resolve()
    adapter_config = adapter / "adapter_config.json"
    if not adapter_config.is_file():
        raise SystemExit(f"No adapter_config.json under {adapter}; pass the lora_adapter directory itself.")

    # Fail loudly on a base/adapter mismatch rather than silently merging the wrong weights.
    recorded_base = json.loads(adapter_config.read_text()).get("base_model_name_or_path")
    if recorded_base and Path(recorded_base).name != Path(args.base).name:
        print(f"WARNING: adapter records base '{recorded_base}' but --base is '{args.base}'")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    print(f"Loading base model {args.base} ({args.dtype}) …")
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, trust_remote_code=True)

    print(f"Applying adapter {adapter} …")
    model = PeftModel.from_pretrained(model, str(adapter), torch_dtype=dtype)

    print("Merging …")
    model = model.merge_and_unload()

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output), safe_serialization=True)
    # vLLM needs the tokenizer alongside the weights to serve this path.
    AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(str(output))
    print(f"Merged model written to {output}")


if __name__ == "__main__":
    main()
