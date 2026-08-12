set -ex

# GRPO on MCP-Atlas through Harbor. Mirrors train_integrations/harbor/run_codecontest.sh;
# the differences are called out inline.
#
# Synchronous training (colocate_all=true): policy and inference share all 8 GPUs and take
# turns. The fully-async variant asserts `not colocate_all`, so it would split the node into
# 4 policy + 4 inference GPUs instead.

#-----------------------
# Prerequisites
#-----------------------
# 1. The harbor dependency must carry the mcp-atlas agent (AfterQuery/harbor-aq @ aq).
#    Bump the rev in pyproject.toml and re-run `uv lock` after every harbor push.
#
# 2. The runtime image, locally tagged (task Dockerfiles use an unqualified FROM):
#      docker pull us-east1-docker.pkg.dev/afterqueryai/mcp-atlas-redelivery-staging/runtime:redeliv-final3-20260622
#      docker tag  <that> mcp-atlas-runtime:delivery7-20260625
#
# 3. Task bundles with the LLM judge swapped in. All 1000 AQ tasks are the training set.
#    Patch a COPY -- editing the delivered set in place is how it ends up with stale graders:
#      HARBOR=/path/to/harbor/adapters/mcp_atlas
#      cp -r ~/AQ-MCP-Atlas-1000-Tasks/tasks ~/data/mcp_atlas_train
#      for d in ~/data/mcp_atlas_train/*/; do cp $HARBOR/template/tests/grade.py "$d/tests/grade.py"; done
#
# 4. API keys. Real secrets belong in .env.mcp_atlas.local (gitignored); the committed
#    .env.mcp_atlas holds placeholders only. Override with ENV_FILE=... to point elsewhere.
#    MCP_ATLAS_JUDGE_* is the reward signal -- nothing trains without it.
: "${ENV_FILE:=$(dirname "$0")/.env.mcp_atlas}"
#
# 5. export WANDB_API_KEY=YOUR_KEY_HERE

#-----------------------
# Dataset setup
#-----------------------
# Harbor tasks are directories of bundles, not parquet: main_harbor takes the paths directly.
# All 1000 AQ tasks train. No eval: get_eval_dataset() returns None when val_data is unset or
# eval_interval <= 0, so both are omitted below and no held-out set is reserved.
TRAIN_DATA="['$HOME/data/mcp_atlas_train']"

#-----------------------
# Directory setup
#-----------------------
RUN_NAME="mcp_atlas_harbor"
STORAGE_ROOT="$HOME/mcp_atlas_harbor_run"
TRIALS_DIR="$STORAGE_ROOT/trials_run"
CKPTS_DIR="$STORAGE_ROOT/ckpts"
EXPORTS_DIR="$STORAGE_ROOT/exports"
LOG_DIR="$STORAGE_ROOT/logs"

TRIAL_CONFIG="$(dirname "$0")/harbor_trial_config.yaml"

#-----------------------
# Training setup
#-----------------------
# The SFT warm start, not the base model: an untrained Qwen3-30B-A3B scores 0.160 on the
# 20-task eval and makes no tool calls at all on 14 of 20, so GRPO would see almost no
# reward variance to learn from. Point at the base model to measure a cold start.
: "${MODEL_PATH:=$HOME/mcp_atlas_sft_run/merged}"
SERVED_MODEL_NAME="Qwen3-30B-A3B"

N_SAMPLES_PER_PROMPT=8
MINI_BATCH_SIZE=8
MAX_MODEL_LEN=32768

# Algorithmic parameters
# token_mean, as in run_codecontest.sh: the loss sums token losses over all valid tokens in
# the batch, so it is identical whether or not step-wise turns got prefix-merged.
#
# sequence_mean is tempting -- with merging it would weight every task equally regardless of
# turn count, instead of letting a rambling 20-turn rollout outweigh a crisp 2-turn one by
# token count. But it is not merge-invariant, and merging demonstrably does not always
# succeed here. Merging needs prompt[i]+response[i] to be a prefix of prompt[i+1], and the
# tool-call round trip breaks that: the model emits `<tool_call>\n {"name"...` with a stray
# space, vLLM's parser lifts the call into structured tool_calls, and the chat template
# re-renders it canonically without the space. Measured on real rollouts, 1 trajectory in 3
# split into two merge groups -- and under sequence_mean would have taken double weight in
# the loss for that reason alone, uncorrelated with its quality.
LOSS_REDUCTION="token_mean"
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false
APPLY_OVERLONG_FILTERING=true

# Essentially achieves interleaved thinking (does not strip thinking tokens). Allows our step-wise
# training to be able to merge more step-wise outputs and hence speed up training.
CHAT_TEMPLATE_PATH="$(dirname "$0")/../../../skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"

# TIS corrections
TIS_TYPE=token
TIS_IMP_RATIO_CAP=2.0

# LoRA, matching the SFT warm start so RL continues from the same adapter shape. A 30B MoE
# full-finetune would also need optimizer state for 128 experts x 48 layers; attention-only
# LoRA is ~27M trainable parameters instead of 30B.
LORA_RANK=32
LORA_ALPHA=64
LORA_TARGETS="[q_proj,k_proj,v_proj,o_proj]"

#----------------
# Infrastructure setup
#----------------
NUM_POLICY_GPUS=8
NUM_INFERENCE_ENGINES=4
TP_SIZE=2
ENABLE_RATE_LIMITING=true
TRAJECTORIES_PER_SECOND=5
# Lower than run_codecontest.sh's 512 on purpose. That runs on Daytona, where trials are
# cloud sandboxes; these are local Docker containers, each running Postgres plus the
# simulated services, so the Docker daemon and image-layer contention bind long before RAM
# does (a live task container measures ~305 MiB). MINI_BATCH_SIZE * N_SAMPLES = 256 trials
# are wanted per step; the rest queue.
MAX_CONCURRENCY=64

# Harbor trial config, with trials_dir pointed at this run's storage.
#
# ${VAR} is Harbor's env-template syntax, resolved inside the trial so secrets stay out of
# persisted configs. But this dict passes through Hydra first, and OmegaConf reads ${...} as
# its own interpolation -- ${MCP_ATLAS_JUDGE_KEY} looks like a custom resolver call and dies
# with "Unsupported interpolation type". Escaping to \${...} makes OmegaConf yield the
# literal string, which Harbor then resolves as intended. run_codecontest.sh never hits this
# because its trial config contains no templates at all.
HARBOR_TRIAL_CONFIG_JSON="$(
  TRIALS_DIR="$TRIALS_DIR" python3 -c '
import json, os, sys, yaml


def escape(o):
    if isinstance(o, str):
        return o.replace("${", "\\${")
    if isinstance(o, dict):
        return {k: escape(v) for k, v in o.items()}
    if isinstance(o, list):
        return [escape(v) for v in o]
    return o


cfg = yaml.safe_load(open(sys.argv[1]))
cfg["trials_dir"] = os.environ["TRIALS_DIR"]
print(json.dumps(escape(cfg)))
' "$TRIAL_CONFIG"
)"

# NOTE: enable_auto_tool_choice / tool_call_parser are REQUIRED here, unlike the previous
# host-side agent which forbade them. The agent's loop runs inside the task container and
# talks to this endpoint directly, so it reads native `tool_calls`; vLLM also rejects a
# request carrying `tools=` under the default tool_choice="auto" unless both flags are set.
uv run --isolated --extra fsdp --extra harbor --env-file "$ENV_FILE" \
  -m examples.train_integrations.harbor.entrypoints.main_harbor \
  data.train_data="$TRAIN_DATA" \
  trainer.policy.model.path="$MODEL_PATH" \
  generator.inference_engine.served_model_name=$SERVED_MODEL_NAME \
  harbor_trial_config="$HARBOR_TRIAL_CONFIG_JSON" \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.log_path=$LOG_DIR \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_ALPHA \
  trainer.policy.model.lora.target_modules="$LORA_TARGETS" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_POLICY_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=hermes \
  trainer.epochs=3 \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=5 \
  trainer.max_ckpts_to_keep=5 \
  trainer.hf_save_interval=5 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger=wandb \
  trainer.project_name=mcp_atlas_harbor \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=false \
  generator.inference_engine.enforce_eager=false \
  generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  "$@"
