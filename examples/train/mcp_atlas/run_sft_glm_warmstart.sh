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
#   2. export WANDB_API_KEY=<key>   (or leave logger=console)
#
# Usage:
#   bash examples/train/mcp_atlas/run_sft_glm_warmstart.sh [extra overrides...]
#
# The exported HF checkpoint feeds straight into RL:
#   bash examples/train/mcp_atlas/run_mcp_atlas.sh \
#     trainer.policy.model.path=$EXPORT_PATH/global_step_<N>

: "${DATA_DIR:="$HOME/data/mcp_atlas_sft"}"
: "${STORAGE_ROOT:="$HOME/mcp_atlas_sft_run"}"
: "${MODEL_PATH:="Qwen/Qwen3-8B"}"

CKPT_PATH="$STORAGE_ROOT/ckpts"
EXPORT_PATH="$STORAGE_ROOT/hf_exports"

#-----------------------
# Training setup
#-----------------------
# Sequence length: measured token lengths over this dataset are median ~2.7k, p99 ~12k,
# max ~28k, so 16384 keeps ~99.5% of trajectories intact. Rows longer than this are
# truncated, which would cut a trajectory's final answer -- raise to 32768 to keep all.
MAX_LENGTH=16384

# With --min-coverage 1.0 --one-per-task the dataset is 505 train rows, so at batch_size 32
# that is ~15 steps/epoch (~31 steps for 2 epochs). SAVE_INTERVAL/eval_interval above that
# total means the only checkpoint is the final one -- set it to ~15 to land on epoch
# boundaries, or leave it high to save just once at the end.
NUM_EPOCHS=2
BATCH_SIZE=32
MICRO_BATCH_PER_GPU=1
SAVE_INTERVAL=100

# LR is the main knob here: this is a small, high-quality distillation set, so too high
# washes out the base model's general ability and too low leaves tool-call formatting unlearned.
LR=1e-5
WARMUP_STEPS=10

NUM_GPUS=8

uv run --isolated --extra fsdp \
    python -m skyrl.train.main_sft \
    strategy=fsdp \
    model.path="$MODEL_PATH" \
    pretokenized_dataset_paths="['$DATA_DIR/train.parquet']" \
    eval_pretokenized_dataset_paths="['$DATA_DIR/validation.parquet']" \
    eval_interval="$SAVE_INTERVAL" \
    max_length=$MAX_LENGTH \
    num_epochs=$NUM_EPOCHS \
    batch_size=$BATCH_SIZE \
    micro_train_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
    remove_microbatch_padding=true \
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
    logger=console \
    project_name=mcp_atlas_sft \
    run_name=qwen3_8b_glm_warmstart \
    ckpt_path="$CKPT_PATH" \
    ckpt_interval=$SAVE_INTERVAL \
    hf_save_interval=$SAVE_INTERVAL \
    export_path="$EXPORT_PATH" \
    resume_from="" \
    "$@"
