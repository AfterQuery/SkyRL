#!/usr/bin/env bash
set -euo pipefail

# Two-node (16 GPU), colocated GRPO for Qwen3.8-27B on Harbor-native
# Toolathlon tasks. This expects a two-node Ray cluster to already be running
# and RAY_ADDRESS to point at it (normally: export RAY_ADDRESS=auto).
#
# Required:
#   DAYTONA_API_KEY       Daytona credential visible to Ray workers
#   WANDB_API_KEY         unless TRAINER_LOGGER=console
#   TOOLATHLON_TASKS_DIR  persistent, upload-only Daytona task directory
#
# No local Docker daemon is required. If TOOLATHLON_TASKS_DIR does not exist,
# the launcher stages it from the source tasks using TOOLATHLON_RUNTIME_IMAGE.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/../../.." && pwd -P)

: "${DAYTONA_API_KEY:?Set DAYTONA_API_KEY before launching training}"
: "${RAY_ADDRESS:=auto}"

model_path=${MODEL_PATH:-Qwen/Qwen3.8-27B}
served_model_name=${SERVED_MODEL_NAME:-Qwen3.8-27B}
tasks_dir=${TOOLATHLON_TASKS_DIR:-$HOME/data/toolathlon-harbor/tasks}
source_tasks_dir=${TOOLATHLON_SOURCE_TASKS_DIR:-$repo_root/toolathlon-tasks/tasks}
runtime_image=${TOOLATHLON_RUNTIME_IMAGE:-us-docker.pkg.dev/afterquery-compute/compute-images/toolathlon-json-runtime:v1}
snapshot_template=${DAYTONA_SNAPSHOT_TEMPLATE:-toolathlon-json-runtime-v1}
storage_root=${STORAGE_ROOT:-$HOME/toolathlon_grpo_qwen38_27b}
lora_sync_path=${LORA_SYNC_PATH:-$repo_root/jobs/rl/runs/toolathlon-grpo-qwen38-27b-${SLURM_JOB_ID:-local}/lora_sync}
trial_config=${HARBOR_TRIAL_CONFIG:-$script_dir/harbor_daytona_training_config.yaml}

run_name=${RUN_NAME:-toolathlon-grpo-qwen38-27b-lora-2node-130k-g8}
trainer_logger=${TRAINER_LOGGER:-wandb}

# Four prompts per optimizer step, with eight samples each, yields 32 complete
# agent trajectories per update. Start here because a single trajectory may
# contain many 131k-token model turns; increase only after measuring one update.
group_size=${GROUP_SIZE:-8}
train_batch_size=${TRAIN_BATCH_SIZE:-4}
policy_mini_batch_size=${POLICY_MINI_BATCH_SIZE:-4}
max_model_len=${MAX_MODEL_LEN:-131072}

num_nodes=${NUM_NODES:-2}
gpus_per_node=${GPUS_PER_NODE:-8}
num_inference_engines=${NUM_INFERENCE_ENGINES:-16}
tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-1}
max_concurrency=${MAX_CONCURRENCY:-128}
sequence_parallel_size=${SEQUENCE_PARALLEL_SIZE:-4}
num_logger_train_samples=${NUM_LOGGER_TRAIN_SAMPLES:-2}

lora_rank=${LORA_RANK:-32}
lora_alpha=${LORA_ALPHA:-64}
lora_targets=${LORA_TARGETS:-'[q_proj,k_proj,v_proj,o_proj]'}
learning_rate=${LEARNING_RATE:-1.0e-6}

if [[ ! -d "$tasks_dir" ]]; then
  echo "Staging upload-only Toolathlon tasks at $tasks_dir" >&2
  uv run python "$script_dir/prepare_daytona_tasks.py" \
    --source "$source_tasks_dir" \
    --output "$tasks_dir" \
    --runtime-image "$runtime_image"
fi
if [[ ! -f "$trial_config" ]]; then
  echo "Harbor trial config not found: $trial_config" >&2
  exit 2
fi
if [[ "$trainer_logger" == wandb && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required when TRAINER_LOGGER=wandb." >&2
  exit 2
fi

mkdir -p \
  "$storage_root/trials" \
  "$storage_root/ckpts" \
  "$storage_root/exports" \
  "$storage_root/logs" \
  "$lora_sync_path"

cd "$repo_root"
export RAY_ADDRESS
export TILELANG_CLEANUP_TEMP_FILES=${TILELANG_CLEANUP_TEMP_FILES:-1}

python_bin=${SKYRL_PYTHON_BIN:-$repo_root/.venv/bin/python}
if [[ ! -x "$python_bin" ]]; then
  echo "Shared SkyRL Python is missing: $python_bin" >&2
  exit 2
fi

"$python_bin" -m examples.train_integrations.harbor.entrypoints.main_harbor \
  data.train_data="['$tasks_dir']" \
  trainer.policy.model.path="$model_path" \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.served_model_name="$served_model_name" \
  generator.inference_engine.language_model_only=true \
  harbor_trial_config_path="$trial_config" \
  harbor_trial_config.trials_dir="$storage_root/trials" \
  harbor_trial_config.environment.kwargs.snapshot_template_name="$snapshot_template" \
  trainer.export_path="$storage_root/exports" \
  trainer.ckpt_path="$storage_root/ckpts" \
  trainer.log_path="$storage_root/logs" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.dynamic_sampling.type=filter \
  trainer.algorithm.dynamic_sampling.max_sample_batches=60 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.max_seq_len="$max_model_len" \
  trainer.policy.model.lora.rank="$lora_rank" \
  trainer.policy.model.lora.alpha="$lora_alpha" \
  trainer.policy.model.lora.target_modules="$lora_targets" \
  trainer.policy.model.lora.lora_sync_path="$lora_sync_path" \
  trainer.policy.optimizer_config.lr="$learning_rate" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_nodes="$num_nodes" \
  trainer.placement.ref_num_nodes="$num_nodes" \
  trainer.placement.policy_num_gpus_per_node="$gpus_per_node" \
  trainer.placement.ref_num_gpus_per_node="$gpus_per_node" \
  trainer.policy.sequence_parallel_size="$sequence_parallel_size" \
  generator.inference_engine.num_engines="$num_inference_engines" \
  generator.inference_engine.tensor_parallel_size="$tensor_parallel_size" \
  generator.inference_engine.engine_init_kwargs.max_model_len="$max_model_len" \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=qwen3_xml \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.enforce_eager=false \
  generator.batched=false \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=true \
  generator.n_samples_per_prompt="$group_size" \
  generator.apply_overlong_filtering=true \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second=4 \
  generator.rate_limit.max_concurrency="$max_concurrency" \
  trainer.epochs=3 \
  trainer.train_batch_size="$train_batch_size" \
  trainer.policy_mini_batch_size="$policy_mini_batch_size" \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.update_epochs_per_batch=1 \
  trainer.num_logger_train_samples="$num_logger_train_samples" \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=5 \
  trainer.max_ckpts_to_keep=3 \
  trainer.hf_save_interval=5 \
  trainer.logger="$trainer_logger" \
  trainer.project_name=toolathlon-harbor \
  trainer.run_name="$run_name" \
  trainer.resume_mode=latest \
  "$@"
