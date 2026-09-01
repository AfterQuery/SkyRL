#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
SHARED_REPO_ROOT="/workspace/users/${USER}/$(basename "$LOCAL_REPO_ROOT")"
if [[ -d "$SHARED_REPO_ROOT" ]]; then
  REPO_ROOT=${REPO_ROOT:-$SHARED_REPO_ROOT}
else
  REPO_ROOT=${REPO_ROOT:-$LOCAL_REPO_ROOT}
fi

MODEL=${MODEL:-Qwen/Qwen3.8-27B}
SERVED_MODEL=${SERVED_MODEL:-Qwen3.8-27B}
LORA_PATH=${LORA_PATH:-}
LORA_NAME=${LORA_NAME:-}
MAX_LORA_RANK=${MAX_LORA_RANK:-32}
SERVER_COUNT=${SERVER_COUNT:-8}
GPUS=${GPUS:-8}
BASE_PORT=${BASE_PORT:-18000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
WORKER_START_TIMEOUT=${WORKER_START_TIMEOUT:-900}

PARTITION=${PARTITION:-gpu}
WCKEY=${WCKEY:-afterquery_research}
EXCLUDE_NODES=${EXCLUDE_NODES:-node-2}
NODELIST=${NODELIST:-}
CPUS_PER_TASK=${CPUS_PER_TASK:-64}
MEMORY=${MEMORY:-768G}
TIME_LIMIT=${TIME_LIMIT:-12:00:00}
JOB_NAME=${JOB_NAME:-qwen38-27b-8x-262k}
LOG_ROOT=${LOG_ROOT:-}
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $0 [options]

Launch one vLLM server per GPU in a single-node Slurm allocation. Workers are
started in parallel and listen on consecutive ports beginning at --base-port.

Options:
  --model PATH                 Hugging Face model path (default: $MODEL)
  --served-model-name NAME     OpenAI API model name (default: $SERVED_MODEL)
  --lora-path PATH             Optional PEFT LoRA adapter directory
  --lora-name NAME             OpenAI API name for the LoRA adapter
  --max-lora-rank COUNT        Maximum supported LoRA rank (default: $MAX_LORA_RANK)
  --servers COUNT              Number of vLLM workers (default: $SERVER_COUNT)
  --gpus COUNT                 GPUs requested from Slurm (default: $GPUS)
  --base-port PORT             First worker port (default: $BASE_PORT)
  --max-model-len TOKENS       Maximum context length (default: $MAX_MODEL_LEN)
  --gpu-memory-utilization N   vLLM GPU memory fraction (default: $GPU_MEMORY_UTILIZATION)
  --worker-start-timeout SEC   Readiness timeout per worker (default: $WORKER_START_TIMEOUT)
  --time D-HH:MM:SS            Slurm time limit (default: $TIME_LIMIT)
  --partition NAME             Slurm partition (default: $PARTITION)
  --wckey NAME                 Slurm workload key (default: $WCKEY)
  --exclude NODE[,NODE...]     Nodes to exclude (default: $EXCLUDE_NODES)
  --nodelist NODE[,NODE...]    Explicit eligible nodes (default: let Slurm choose)
  --cpus COUNT                 CPUs per task (default: $CPUS_PER_TASK)
  --mem SIZE                   Node memory request (default: $MEMORY)
  --job-name NAME              Slurm job name (default: $JOB_NAME)
  --repo-root PATH             Repo path visible on compute nodes (default: $REPO_ROOT)
  --log-root PATH              Log directory (default: REPO_ROOT/jobs/model_server)
  --dry-run                    Print the sbatch command without submitting
  -h, --help                   Show this help

Environment variables with the uppercase names above may also set defaults.
EOF
}

while (($# > 0)); do
  case "$1" in
    --model) MODEL=$2; shift 2 ;;
    --served-model-name) SERVED_MODEL=$2; shift 2 ;;
    --lora-path) LORA_PATH=$2; shift 2 ;;
    --lora-name) LORA_NAME=$2; shift 2 ;;
    --max-lora-rank) MAX_LORA_RANK=$2; shift 2 ;;
    --servers) SERVER_COUNT=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --base-port) BASE_PORT=$2; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN=$2; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION=$2; shift 2 ;;
    --worker-start-timeout) WORKER_START_TIMEOUT=$2; shift 2 ;;
    --time) TIME_LIMIT=$2; shift 2 ;;
    --partition) PARTITION=$2; shift 2 ;;
    --wckey) WCKEY=$2; shift 2 ;;
    --exclude) EXCLUDE_NODES=$2; shift 2 ;;
    --nodelist) NODELIST=$2; shift 2 ;;
    --cpus) CPUS_PER_TASK=$2; shift 2 ;;
    --mem) MEMORY=$2; shift 2 ;;
    --job-name) JOB_NAME=$2; shift 2 ;;
    --repo-root) REPO_ROOT=$2; shift 2 ;;
    --log-root) LOG_ROOT=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

LOG_ROOT=${LOG_ROOT:-$REPO_ROOT/jobs/model_server}

for value in "$SERVER_COUNT" "$GPUS" "$BASE_PORT" "$MAX_MODEL_LEN" "$MAX_LORA_RANK" "$WORKER_START_TIMEOUT"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Expected a non-negative integer, got: $value" >&2; exit 2; }
done
((SERVER_COUNT > 0)) || { echo "--servers must be greater than zero" >&2; exit 2; }
((GPUS > 0)) || { echo "--gpus must be greater than zero" >&2; exit 2; }
((SERVER_COUNT <= GPUS)) || { echo "--servers cannot exceed --gpus" >&2; exit 2; }
if [[ -n "$LORA_PATH" ]]; then
  [[ -d "$LORA_PATH" ]] || { echo "LoRA adapter directory not found: $LORA_PATH" >&2; exit 2; }
  [[ -n "$LORA_NAME" ]] || { echo "--lora-name is required with --lora-path" >&2; exit 2; }
elif [[ -n "$LORA_NAME" ]]; then
  echo "--lora-path is required with --lora-name" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT/slurm" "$LOG_ROOT/servers"

# Do not use sbatch --export here. On this cluster, explicit --export settings
# have caused allocations to fail with NODE_FAIL. Slurm's default environment
# propagation carries these exported variables into the job.
export SERVER_REPO_ROOT="$REPO_ROOT"
export SERVER_LOG_ROOT="$LOG_ROOT/servers"
export MODEL SERVED_MODEL LORA_PATH LORA_NAME MAX_LORA_RANK
export SERVER_COUNT BASE_PORT MAX_MODEL_LEN
export GPU_MEMORY_UTILIZATION WORKER_START_TIMEOUT

SBATCH_ARGS=(
  --parsable
  --partition="$PARTITION"
  --wckey="$WCKEY"
  --job-name="$JOB_NAME"
  --nodes=1
  --ntasks=1
  --gres="gpu:$GPUS"
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="$MEMORY"
  --time="$TIME_LIMIT"
  --output="$LOG_ROOT/slurm/$JOB_NAME-%j.out"
  --chdir="$REPO_ROOT"
)
[[ -n "$EXCLUDE_NODES" ]] && SBATCH_ARGS+=(--exclude="$EXCLUDE_NODES")
[[ -n "$NODELIST" ]] && SBATCH_ARGS+=(--nodelist="$NODELIST")

JOB_BODY='set -euo pipefail
cd "$SERVER_REPO_ROOT"
log_dir="$SERVER_LOG_ROOT/job-$SLURM_JOB_ID"
mkdir -p "$log_dir"

pids=()
ports=()
cleanup() {
  if ((${#pids[@]} > 0)); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for ((gpu = 0; gpu < SERVER_COUNT; gpu++)); do
  port=$((BASE_PORT + gpu))
  lora_args=()
  if [[ -n "$LORA_PATH" ]]; then
    lora_args=(--enable-lora --max-lora-rank "$MAX_LORA_RANK" --lora-modules "$LORA_NAME=$LORA_PATH")
  fi
  CUDA_VISIBLE_DEVICES=$gpu uv run --isolated --extra fsdp \
    python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    "${lora_args[@]}" \
    --host 0.0.0.0 \
    --port "$port" \
    >"$log_dir/server-$port.log" 2>&1 &
  pids+=("$!")
  ports+=("$port")
done

echo "Started $SERVER_COUNT vLLM processes in parallel on $(hostname)."
for ((i = 0; i < SERVER_COUNT; i++)); do
  elapsed=0
  until curl --fail --silent --max-time 2 "http://127.0.0.1:${ports[$i]}/health" >/dev/null 2>&1; do
    if ! kill -0 "${pids[$i]}" 2>/dev/null; then
      echo "Worker on port ${ports[$i]} exited during startup; see $log_dir/server-${ports[$i]}.log" >&2
      exit 1
    fi
    if ((elapsed >= WORKER_START_TIMEOUT)); then
      echo "Worker on port ${ports[$i]} did not become healthy within $WORKER_START_TIMEOUT seconds" >&2
      exit 1
    fi
    sleep 5
    ((elapsed += 5))
  done
  echo "Worker on port ${ports[$i]} is healthy."
done

last_port=$((BASE_PORT + SERVER_COUNT - 1))
echo "All $SERVER_COUNT vLLM servers are healthy on $(hostname), ports $BASE_PORT-$last_port."
wait
'
export JOB_BODY

if ((DRY_RUN)); then
  printf 'Environment:\n'
  printf '  MODEL=%q SERVED_MODEL=%q SERVER_COUNT=%q BASE_PORT=%q MAX_MODEL_LEN=%q GPU_MEMORY_UTILIZATION=%q\n' \
    "$MODEL" "$SERVED_MODEL" "$SERVER_COUNT" "$BASE_PORT" "$MAX_MODEL_LEN" "$GPU_MEMORY_UTILIZATION"
  if [[ -n "$LORA_PATH" ]]; then
    printf '  LORA_PATH=%q LORA_NAME=%q MAX_LORA_RANK=%q\n' "$LORA_PATH" "$LORA_NAME" "$MAX_LORA_RANK"
  fi
  printf 'Command:\n  sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}"
  printf ' --wrap=%q\n' 'exec bash -c "$JOB_BODY"'
  exit 0
fi

job_id=$(sbatch "${SBATCH_ARGS[@]}" --wrap='exec bash -c "$JOB_BODY"')
echo "Submitted vLLM cluster as Slurm job $job_id."
echo "Status: squeue -j $job_id -o '%.10i %.28j %.8T %.12M %.20R'"
echo "Slurm log: $LOG_ROOT/slurm/$JOB_NAME-$job_id.out"
echo "Worker logs: $LOG_ROOT/servers/job-$job_id/server-<port>.log"
