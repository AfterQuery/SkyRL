set -ex

# GRPO training on Toolathlon tasks with the decoupled Toolathlon runner.
#
# Prerequisites (see README.md):
#   1. A Toolathlon checkout set up per its README (global_configs.py, docker/podman,
#      task image pulled, app containers deployed via global_preparation/deploy_containers.sh).
#   2. Task lists prepared:
#      uv run examples/train/toolathlon/prepare_toolathlon_dataset.py \
#        --toolathlon-repo $TOOLATHLON_REPO --output-dir $HOME/data/toolathlon
#
# wandb api key.
# export WANDB_API_KEY=YOUR_KEY_HERE

TOOLATHLON_REPO="${TOOLATHLON_REPO:-/home/ubuntu/Toolathlon}"

#-----------------------
# Dataset setup
#-----------------------
DATA_DIR="$HOME/data/toolathlon"
TRAIN_DATA="['$DATA_DIR/train_tasks.txt']"
EVAL_DATA="['$DATA_DIR/val_tasks.txt']"

#-----------------------
# Directory setup
#-----------------------
RUN_NAME="toolathlon"
STORAGE_ROOT="$HOME/toolathlon_run"
DUMP_ROOT="$STORAGE_ROOT/rollouts"
CKPTS_DIR="$STORAGE_ROOT/ckpts"
EXPORTS_DIR="$STORAGE_ROOT/exports"
LOG_DIR="$STORAGE_ROOT/logs"

#-----------------------
# Training setup
#-----------------------
# NOTE: repetitions of the same task run sequentially (shared external app state), so large
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
MAX_CONCURRENT_TASKS=8  # simultaneous Toolathlon task containers

uv run --isolated --extra fsdp -m examples.train.toolathlon.main_toolathlon \
  data.train_data="$TRAIN_DATA" \
  data.val_data="$EVAL_DATA" \
  trainer.policy.model.path=Qwen/Qwen3-8B \
  generator.inference_engine.served_model_name=Qwen3-8B \
  toolathlon_config.repo_path=$TOOLATHLON_REPO \
  toolathlon_config.dump_root=$DUMP_ROOT \
  toolathlon_config.max_concurrent_tasks=$MAX_CONCURRENT_TASKS \
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
  trainer.logger=console \
  trainer.project_name=toolathlon \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  "$@"
