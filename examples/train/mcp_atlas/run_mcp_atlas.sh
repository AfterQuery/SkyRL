set -ex

# GRPO training on MCP-Atlas tasks against a local sandbox.
#
# Prerequisites (see README.md):
#   1. The MCP-Atlas sandbox running:
#        docker run -d --name mcp-atlas-sandbox -p 1984:1984 --env-file .env agent-environment:latest
#   2. Dataset prepared:
#        uv run examples/train/mcp_atlas/prepare_mcp_atlas_dataset.py --output-dir $HOME/data/mcp_atlas \
#          --available-tools-only --exclude-mutating
#   3. A judge endpoint (the reward!) — set these to a real LLM API, never the policy:
# export EVAL_LLM_BASE_URL="https://your-litellm-or-openai-endpoint"
# export EVAL_LLM_API_KEY="YOUR_KEY_HERE"
# export EVAL_LLM_MODEL="gemini/gemini-3.1-pro-preview"
#
# wandb api key.
# export WANDB_API_KEY=YOUR_KEY_HERE

#-----------------------
# Dataset setup
#-----------------------
DATA_DIR="$HOME/data/mcp_atlas"
TRAIN_DATA="['$DATA_DIR/train.parquet']"
EVAL_DATA="['$DATA_DIR/val.parquet']"

#-----------------------
# Directory setup
#-----------------------
RUN_NAME="mcp_atlas"
STORAGE_ROOT="$HOME/mcp_atlas_run"
DUMP_ROOT="$STORAGE_ROOT/rollouts"
CKPTS_DIR="$STORAGE_ROOT/ckpts"
EXPORTS_DIR="$STORAGE_ROOT/exports"
LOG_DIR="$STORAGE_ROOT/logs"

#-----------------------
# Training setup
#-----------------------
# NOTE: repetitions of the same task run sequentially against the shared sandbox, so large
# N_SAMPLES_PER_PROMPT increases wall-clock time per step roughly linearly.
N_SAMPLES_PER_PROMPT=4
TRAIN_BATCH_SIZE=8
MAX_MODEL_LEN=32768

#----------------
# Infrastructure setup
#----------------
NUM_POLICY_GPUS=8
NUM_INFERENCE_ENGINES=4
TP_SIZE=2
MAX_CONCURRENT_TASKS=8  # simultaneous rollouts against the sandbox

uv run --isolated --extra fsdp -m examples.train.mcp_atlas.main_mcp_atlas \
  data.train_data="$TRAIN_DATA" \
  data.val_data="$EVAL_DATA" \
  trainer.policy.model.path=Qwen/Qwen3-8B \
  generator.inference_engine.served_model_name=Qwen3-8B \
  mcp_atlas_config.dump_root=$DUMP_ROOT \
  mcp_atlas_config.max_concurrent_tasks=$MAX_CONCURRENT_TASKS \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.log_path=$LOG_DIR \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_POLICY_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=hermes \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=false \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=1 \
  generator.sampling_params.max_generate_length=4096 \
  trainer.epochs=3 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=false \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.ckpt_interval=5 \
  trainer.max_ckpts_to_keep=5 \
  trainer.hf_save_interval=5 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.logger=wandb \
  trainer.project_name=mcp_atlas \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  "$@"
