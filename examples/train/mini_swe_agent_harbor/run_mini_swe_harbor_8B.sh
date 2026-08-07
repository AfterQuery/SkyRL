set -x

# Colocated GRPO training+generation for Qwen3-8B on Harbor-format tasks.
# Uses 1 node with 6 GPUs (policy and vLLM colocated on the same 6).
#
#   uv run --isolated examples/train/mini_swe_agent_harbor/preprocess_harbor.py \
#       --tasks_dir ~/swebench-pro-tinker/ez_500_verified \
#       --output_dir ~/data/harbor_ez500 --skip_build
#   bash examples/train/mini_swe_agent_harbor/run_mini_swe_harbor_8B.sh
#
# Podman must be usable on every Ray worker: one container per trajectory, and the images
# named in the parquet's instance.image_name must be pullable from that worker.
#
# Registry auth. Task images live in a private Artifact Registry, so podman needs a
# credential. Use the service-account JSON key, NOT `gcloud auth print-access-token`:
# access tokens expire after ~1 hour and images are pulled lazily throughout the run, so a
# token-based login dies partway through with an opaque pull failure. The `_json_key`
# username takes the whole key file as the password and does not expire.
#
#   podman login -u _json_key --password-stdin us-docker.pkg.dev < ~/liheng-image-pull-key.json
#
# By default podman writes that credential to $XDG_RUNTIME_DIR/containers/auth.json, which
# is tmpfs and is lost on reboot. REGISTRY_AUTH_FILE below points at a persistent copy and
# is inherited by the Ray workers that shell out to podman. Log in once with:
#
#   REGISTRY_AUTH_FILE=$HOME/.config/containers/auth.json \
#     podman login -u _json_key --password-stdin us-docker.pkg.dev < ~/liheng-image-pull-key.json
#
# On a multi-node cluster this file must exist on every worker (shared storage, or copy it).
export REGISTRY_AUTH_FILE="$HOME/.config/containers/auth.json"
#
# Currently sized as a VERIFICATION run: max_training_steps=2, eval_before_train=false,
# 24 trajectories/step. See the batch-size note below before scaling up.

# Output of preprocess_harbor.py. Must contain train.parquet AND validation.parquet --
# preprocess_harbor.py writes `validation.parquet`, not `test.parquet`, so a dataset dir
# built by some other tool will fail at data-loading time.
DATA_DIR="$HOME/data/harbor_ez500"
CKPT_PATH="$HOME/ckpts/llm_mini_swe_harbor"

# Save trajectories here for debugging
# NOTE: For a multi-node cluster, ensure that this is on NFS so that you can save all trajectories in the same path
MINISWE_TRAJ_DIR="$HOME/mini_swe_harbor_trajs"

# Use GPUs 0-5 only. This host has 8, but GPU 6 is in a fault state
# ("GPU requires reset" in nvidia-smi -q -i 6, NV_ERR_GPU_IN_FULLCHIP_RESET in dmesg).
# NVML still enumerates it, so torch.cuda.device_count() returns 8 while the CUDA runtime
# sees 7 -- and torch's lazy init walks every index, so the one bad device makes
# get_device_properties FAIL ON ALL EIGHT with
#   RuntimeError: device >= 0 && device < num_gpus ... device=7, num_gpus=7
# Masking it here is what makes the run possible; NUM_GPUS alone does not help, because the
# failure happens during CUDA init, before any placement decision. Drop this line once the
# GPU has been reset (sudo nvidia-smi -r -i 6) or the host rebooted.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

NUM_GPUS=6
NNODES=1
NUM_INFERENCE_ENGINES=6
TP_SIZE=1
LOGGER=console
SP_SIZE=2

# Batch sizes must satisfy skyrl/train/utils/utils.py::validate_batch_sizes:
#
#   dp_size       = NUM_GPUS * NNODES / SP_SIZE   = 6 / 2 = 3
#   mini_per_gpu  = policy_mini_batch_size * n_samples_per_prompt // dp_size
#   train_per_gpu = train_batch_size       * n_samples_per_prompt // dp_size
#   train_per_gpu % mini_per_gpu == 0  and  mini_per_gpu % micro_train_batch == 0
#
# That `//` is integer division, so a batch that does not divide evenly across dp_size is
# silently TRUNCATED, not rejected. The previous 16 x 4 = 64 over dp_size=3 gave 21 per GPU
# and quietly dropped a sample every step. Keep train_batch_size * n_samples_per_prompt a
# multiple of dp_size.
#
# Sized here for a verification run: 6 x 4 = 24 trajectories/step, 8 per GPU, 8 micro-steps.
# For a real run use train_batch_size=12 / policy_mini_batch_size=12 (48/step), or set
# NUM_GPUS=8 (this host has 8) which makes dp_size=4 and 16 x 4 = 64 divide evenly.

# NOTE: `generator.max_turns` is unused; the effective limit is `step_limit` in harbor.yaml.
# It simply has to be > 1.
# --with-requirements pins mini-swe-agent to the v1 fork this example is written against,
# overriding uv.lock (which floats to 2.x) for this invocation only. See
# requirements-miniswe.txt.
uv run --isolated --extra fsdp --extra miniswe \
  --with-requirements examples/train/mini_swe_agent_harbor/requirements-miniswe.txt \
  --env-file examples/train/mini_swe_agent_harbor/.env.harbor \
  -m examples.train.mini_swe_agent_harbor.main_mini_swe_harbor \
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
  trainer.policy.sequence_parallel_size=$SP_SIZE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  trainer.epochs=20 \
  trainer.max_training_steps=2 \
  trainer.eval_batch_size=6 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=6 \
  trainer.policy_mini_batch_size=6 \
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
