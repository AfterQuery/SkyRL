#!/bin/bash
set -xeou pipefail

# SFT warm start for Qwen3-8B on GLM-5.2 teacher trajectories from AQ MCP-Atlas.
#
# Distills the teacher's tool-use behaviour (which tool to call, with which arguments, and
# when to stop and answer) into the student before RL. The RL stage then only has to improve
# on a policy that already emits well-formed tool calls, instead of discovering the format
# from a zero-reward start.
#
# Prerequisites:
#   1. Unzip both bundles (trajectories + tasks), then build the SFT dataset. It is emitted
#      pre-tokenized (input_ids + full-sequence loss_mask) with qwen3_acc_thinking.jinja2,
#      because SFTConfig has no chat_template knob and the stock Qwen3 template injects an
#      empty <think></think> into every assistant turn inside the trained span:
#        uv run examples/train/mcp_atlas/prepare_glm_sft_dataset.py \
#          --trajectories-dir ~/AQ-MCP-Atlas-1000-Trajectories-GLM-5.2 \
#          --tasks-dir ~/AQ-MCP-Atlas-1000-Tasks \
#          --output-dir ~/data/mcp_atlas_sft --min-coverage 0.75 --one-per-task
#   2. export WANDB_API_KEY=<key>    (metrics go to W&B; set logger=console to disable)
#   3. HuggingFace auth for the ~61 GB Qwen3-30B-A3B pull: `huggingface-cli login` or
#      export HF_TOKEN=<token>. Unauthenticated pulls of this size get rate limited.
#
# Usage:
#   bash examples/train/mcp_atlas/run_sft_glm_warmstart.sh [extra overrides...]
#
# With LoRA, checkpoints contain a PEFT adapter at
#   $CKPT_PATH/global_step_<N>/policy/lora_adapter/
# and NOT a merged model -- SkyRL has no merge_and_unload anywhere, so hf_save_interval is
# left at 0 rather than producing a misleading "export". Merge before RL:
#   uv run examples/train/mcp_atlas/merge_lora_adapter.py \
#     --base Qwen/Qwen3-30B-A3B \
#     --adapter $CKPT_PATH/global_step_<N>/policy/lora_adapter \
#     --output $HOME/mcp_atlas_sft_run/merged
#   bash examples/train/mcp_atlas/run_mcp_atlas_harbor.sh \
#     trainer.policy.model.path=$HOME/mcp_atlas_sft_run/merged

: "${DATA_DIR:="$HOME/data/mcp_atlas_sft"}"
: "${STORAGE_ROOT:="$HOME/mcp_atlas_sft_run"}"
: "${MODEL_PATH:="Qwen/Qwen3-30B-A3B"}"
RUN_NAME="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')_glm_warmstart_lora"

CKPT_PATH="$STORAGE_ROOT/ckpts"
EXPORT_PATH="$STORAGE_ROOT/hf_exports"

#-----------------------
# Training setup
#-----------------------
# Sequence length: measured over all 531 rows the max is 28558 tokens, so 32768 keeps every
# trajectory whole. Rows above max_length are truncated from the end, which would amputate the
# final answer -- the single most valuable training target -- so do not lower this without
# re-checking the prep script's length report.
MAX_LENGTH=32768

# With --min-coverage 1.0 --one-per-task the dataset is 505 train rows, so at batch_size 32
# that is ~15 steps/epoch (~31 steps for 2 epochs). SAVE_INTERVAL/eval_interval above that
# total means the only checkpoint is the final one -- set it to ~15 to land on epoch
# boundaries, or leave it high to save just once at the end.
NUM_EPOCHS=2
BATCH_SIZE=32
MICRO_BATCH_PER_GPU=1
SAVE_INTERVAL=50

# LoRA. rank/alpha are the capacity knobs; scale = alpha/rank = 2 here.
#
# target_modules is attention-only ON PURPOSE. Qwen3-30B-A3B is an MoE with 128 experts per
# layer across 48 layers, so the default "all-linear" would attach adapters to every expert
# projection -- ~18k LoRA modules -- which is slow and memory-hungry for little benefit.
# NOTE: the bracket form below parses to a real list; a comma-separated *string* would be
# passed to PEFT as one module name and silently train nothing.
LORA_RANK=32
LORA_ALPHA=64
LORA_TARGETS="[q_proj,k_proj,v_proj,o_proj]"

# LoRA wants a markedly higher LR than full fine-tuning (only the adapters move).
# Warmup is short because the whole run is only ~31 steps at batch_size 32.
LR=3e-5
WARMUP_STEPS=5

NUM_GPUS=8

# Build the environment before launching, and fall back to the cache if the network is
# uncooperative. Two of this extra's wheels are GitHub release assets (causal-conv1d among
# them); when several processes resolve at once GitHub answers with HTTP/2 "refused stream"
# and 503, and uv then fails to generate package metadata. Resolving up front means that
# failure happens here, in a second, rather than after the model has started loading -- and
# UV_OFFLINE only ever succeeds if every dependency is already cached, so it cannot mask a
# genuinely missing one.
UV_ARGS=(--isolated --extra fsdp)
if ! uv run "${UV_ARGS[@]}" python -c pass >/dev/null 2>&1; then
  echo "environment build failed (likely a throttled GitHub release asset); retrying from cache"
  if UV_OFFLINE=1 uv run "${UV_ARGS[@]}" python -c pass >/dev/null 2>&1; then
    export UV_OFFLINE=1
    echo "  cache is warm; continuing with UV_OFFLINE=1"
  else
    echo "  could not build the environment offline either -- not a transient fetch failure" >&2
    exit 1
  fi
fi

uv run --isolated --extra fsdp \
    python -m skyrl.train.main_sft \
    strategy=fsdp \
    model.path="$MODEL_PATH" \
    model.lora.rank=$LORA_RANK \
    model.lora.alpha=$LORA_ALPHA \
    model.lora.target_modules="$LORA_TARGETS" \
    pretokenized_dataset_paths="['$DATA_DIR/train.parquet']" \
    eval_pretokenized_dataset_paths="['$DATA_DIR/validation.parquet']" \
    eval_interval=5 \
    max_length=$MAX_LENGTH \
    num_epochs=$NUM_EPOCHS \
    batch_size=$BATCH_SIZE \
    micro_train_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
    remove_microbatch_padding=true \
    dataloader_num_workers=2 \
    seed=42 \
    optimizer_config.lr=$LR \
    optimizer_config.weight_decay=1e-2 \
    optimizer_config.max_grad_norm=1.0 \
    optimizer_config.num_warmup_steps=$WARMUP_STEPS \
    optimizer_config.scheduler=constant_with_warmup \
    placement.num_nodes=1 \
    placement.num_gpus_per_node=$NUM_GPUS \
    fsdp_config.cpu_offload=false \
    fsdp_config.reshard_after_forward=true \
    logger=wandb \
    project_name=mcp_atlas_sft \
    run_name=$RUN_NAME \
    tags="[sft,lora,qwen3-30b-a3b,glm-teacher]" \
    ckpt_path="$CKPT_PATH" \
    ckpt_interval=$SAVE_INTERVAL \
    hf_save_interval=0 \
    export_path="$EXPORT_PATH" \
    resume_from="" \
    "$@"
