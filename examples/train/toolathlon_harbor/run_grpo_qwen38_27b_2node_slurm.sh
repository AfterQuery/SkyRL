#!/usr/bin/env bash
#SBATCH --job-name=toolathlon-grpo-qwen38-27b
#SBATCH --partition=gpu
#SBATCH --wckey=afterquery_research
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=768G
#SBATCH --time=48:00:00
#SBATCH --exclude=node-2

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
  repo_root=$(cd -- "$script_dir/../../.." && pwd -P)
else
  repo_root=$(pwd -P)
  script_dir="$repo_root/examples/train/toolathlon_harbor"
fi

# Make the same script convenient to invoke from a login shell. Any arguments
# are forwarded to the inner SkyRL launcher as Hydra overrides.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  log_dir=${SLURM_LOG_DIR:-$repo_root/jobs/rl/slurm}
  mkdir -p "$log_dir"
  exec sbatch --parsable \
    --output="$log_dir/toolathlon-grpo-qwen38-27b-%j.out" \
    --chdir="$repo_root" \
    "$0" "$@"
fi

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#nodes[@]} -ne 2 ]]; then
  echo "Expected exactly two allocated nodes; got ${#nodes[@]}: ${nodes[*]}" >&2
  exit 2
fi

head_node=${nodes[0]}
worker_node=${nodes[1]}
ray_port=${RAY_PORT:-6379}
ray_dashboard_port=${RAY_DASHBOARD_PORT:-8265}
ray_start_timeout=${RAY_START_TIMEOUT:-300}
ray_num_cpus=${RAY_NUM_CPUS_PER_NODE:-${SLURM_CPUS_PER_TASK:-128}}
ray_num_gpus=${RAY_NUM_GPUS_PER_NODE:-8}
ray_tmp_root=${RAY_TMP_ROOT:-/tmp/skyrl-ray-$SLURM_JOB_ID}
cuda_home=${SKYRL_CUDA_HOME:-/usr/local/cuda-12.8}
shared_tasks_dir=${TOOLATHLON_SHARED_TASKS_DIR:-$repo_root/data/toolathlon-harbor/tasks}
local_tasks_dir=${TOOLATHLON_LOCAL_TASKS_DIR:-$ray_tmp_root/toolathlon-harbor/tasks}

python_bin=${SKYRL_PYTHON_BIN:-$repo_root/.venv/bin/python}
ray_bin=${SKYRL_RAY_BIN:-$repo_root/.venv/bin/ray}
if [[ ! -x "$python_bin" || ! -x "$ray_bin" ]]; then
  echo "Shared SkyRL environment is incomplete under $repo_root/.venv" >&2
  exit 2
fi
if [[ ! -x "$cuda_home/bin/nvcc" ]]; then
  echo "CUDA toolkit is incomplete: $cuda_home/bin/nvcc is not executable" >&2
  exit 2
fi
export CUDA_HOME="$cuda_home"
export TILELANG_CLEANUP_TEMP_FILES=${TILELANG_CLEANUP_TEMP_FILES:-1}
export PATH="$CUDA_HOME/bin:$PATH"
ray_cmd=("$ray_bin")
head_ip=$(srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --gres=gpu:0 --nodelist="$head_node" \
  bash -lc "hostname -I | awk '{for (i=1; i<=NF; i++) if (\$i ~ /^10\.65\.0\./) {print \$i; found=1; break} if (!found) print \$1}'")
head_ip=${head_ip//$'\n'/}
if [[ -z "$head_ip" ]]; then
  echo "Could not determine the Ray head IP for $head_node." >&2
  exit 2
fi
ray_address="$head_ip:$ray_port"

head_step_pid=""
worker_step_pid=""
cleanup() {
  status=$?
  trap - EXIT INT TERM
  for pid in "$worker_step_pid" "$head_step_pid"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Staging Toolathlon tasks from shared storage onto both nodes"
srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=4 \
  --gres=gpu:0 \
  "$script_dir/stage_tasks_local.sh" "$shared_tasks_dir" "$local_tasks_dir"
export TOOLATHLON_TASKS_DIR="$local_tasks_dir"
# Ray rendezvous stays on the routable 10.65.0.x network. Use the eight
# matching 400 Gb/s rails for both Gloo's CPU model-state distribution and
# NCCL's GPU collectives; pinning either backend to ens1 bottlenecks it at
# 10 Gb/s.
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ens2,ens3,ens4,ens5,ens6,ens7,ens8,ens9}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens2,ens3,ens4,ens5,ens6,ens7,ens8,ens9}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-=rocep9s0,rocep23s0,rocep64s0,rocep73s0,rocep134s0,rocep143s0,rocep189s0,rocep198s0}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}

echo "Distributed networking: GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
echo "Distributed networking: NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME NCCL_IB_HCA=$NCCL_IB_HCA"

echo "Starting Ray head on $head_node ($head_ip)"
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-128}" \
  --gres="gpu:$ray_num_gpus" --nodelist="$head_node" \
  bash -lc "
    set -euo pipefail
    cd '$repo_root'
    mkdir -p '$ray_tmp_root/head'
    '$repo_root/.venv/bin/ray' stop --force >/dev/null 2>&1 || true
    exec '$repo_root/.venv/bin/ray' start \
      --head --block --disable-usage-stats \
      --node-ip-address='$head_ip' --port='$ray_port' \
      --dashboard-host=0.0.0.0 --dashboard-port='$ray_dashboard_port' \
      --num-cpus='$ray_num_cpus' --num-gpus='$ray_num_gpus' \
      --temp-dir='$ray_tmp_root/head'
  " &
head_step_pid=$!

deadline=$((SECONDS + ray_start_timeout))
until "${ray_cmd[@]}" status --address="$ray_address" >/dev/null 2>&1; do
  if ! kill -0 "$head_step_pid" 2>/dev/null; then
    echo "Ray head step exited during startup." >&2
    wait "$head_step_pid"
  fi
  if (( SECONDS >= deadline )); then
    echo "Ray head did not become ready within $ray_start_timeout seconds." >&2
    exit 1
  fi
  sleep 2
done

echo "Starting Ray worker on $worker_node"
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-128}" \
  --gres="gpu:$ray_num_gpus" --nodelist="$worker_node" \
  bash -lc "
    set -euo pipefail
    cd '$repo_root'
    mkdir -p '$ray_tmp_root/worker'
    worker_ip=\$(hostname -I | awk '{for (i=1; i<=NF; i++) if (\$i ~ /^10\.65\.0\./) {print \$i; found=1; break} if (!found) print \$1}')
    '$repo_root/.venv/bin/ray' stop --force >/dev/null 2>&1 || true
    exec '$repo_root/.venv/bin/ray' start \
      --block --disable-usage-stats --address='$ray_address' \
      --node-ip-address=\"\$worker_ip\" \
      --num-cpus='$ray_num_cpus' --num-gpus='$ray_num_gpus' \
      --temp-dir='$ray_tmp_root/worker'
  " &
worker_step_pid=$!

echo "Waiting for Ray to report 2 nodes and 16 GPUs"
deadline=$((SECONDS + ray_start_timeout))
until RAY_ADDRESS="$ray_address" "$python_bin" - <<'PY'
import ray

ray.init(address="auto", logging_level="ERROR")
alive = [node for node in ray.nodes() if node["Alive"]]
resources = ray.cluster_resources()
raise SystemExit(0 if len(alive) == 2 and resources.get("GPU", 0) >= 16 else 1)
PY
do
  if ! kill -0 "$worker_step_pid" 2>/dev/null; then
    echo "Ray worker step exited during startup." >&2
    wait "$worker_step_pid"
  fi
  if (( SECONDS >= deadline )); then
    echo "The two-node Ray cluster did not become ready within $ray_start_timeout seconds." >&2
    "${ray_cmd[@]}" status --address="$ray_address" || true
    exit 1
  fi
  sleep 3
done

echo "Ray cluster ready at $ray_address; launching Toolathlon GRPO"
export RAY_ADDRESS="$ray_address"
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:0 \
  --nodelist="$head_node" --chdir="$repo_root" \
  "$script_dir/run_grpo_qwen38_27b_2node.sh" "$@"
