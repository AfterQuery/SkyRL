set -x

# Colocated GRPO training+generation for Qwen3-8B on Harbor-format tasks.
# Uses 1 node with 8 GPUs.
#
#   uvx harbor datasets download terminal-bench@2.0 -o ~/data/harbor_tasks/terminal-bench-2.0
#   # ...or any Harbor task tree; open-thoughts/CodeContests unpacks to the same layout
#   uv run --isolated examples/train/mini_swe_agent_harbor/preprocess_harbor.py \
#       --tasks_dir ~/data/harbor_tasks/CodeContests --output_dir ~/data/harbor_codecontests
#   bash examples/train/mini_swe_agent_harbor/run_mini_swe_harbor_8B.sh
#
# Podman must be usable on every Ray worker: one container per trajectory.

DATA_DIR="$HOME/data/harbor_codecontests"
CKPT_PATH="$HOME/ckpts/llm_mini_swe_harbor"

# Save trajectories here for debugging
# NOTE: For a multi-node cluster, ensure that this is on NFS so that you can save all trajectories in the same path
MINISWE_TRAJ_DIR="$HOME/mini_swe_harbor_trajs"

NUM_GPUS=8
NNODES=1
NUM_INFERENCE_ENGINES=4
TP_SIZE=2
LOGGER=wandb

# NOTE: `generator.max_turns` is unused; the effective limit is `step_limit` in harbor.yaml.
# It simply has to be > 1.
uv run --isolated --extra fsdp --extra miniswe --env-file examples/train/mini_swe_agent_harbor/.env.harbor -m examples.train.mini_swe_agent_harbor.main_mini_swe_harbor \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-8B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.policy.sequence_parallel_size=2 \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.dump_data_batch=true \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=4096 \
  generator.max_input_length=30720 \
  generator.max_turns=20 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=True \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="mini_swe_harbor" \
  trainer.run_name="mini_swe_8B_harbor_codecontests" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.miniswe_config_path="examples/train/mini_swe_agent_harbor/harbor.yaml" \
  generator.miniswe_traj_dir=$MINISWE_TRAJ_DIR \
  generator.harbor_verifier_timeout=600 \
  $@
