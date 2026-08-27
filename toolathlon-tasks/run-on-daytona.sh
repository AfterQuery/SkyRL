#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./toolathlon-tasks/run-on-daytona.sh TASK_ID [HARBOR_RUN_OPTIONS...]

Stages a self-contained copy of one Toolathlon task, then runs it on Daytona.
If no Harbor options are supplied, the Oracle agent is used as a smoke test.

Examples:
  ./toolathlon-tasks/run-on-daytona.sh 401k-watchlist-recency-window-refresh
  ./toolathlon-tasks/run-on-daytona.sh 401k-watchlist-recency-window-refresh \
    --agent claude-code --model MODEL

Environment variables:
  HARBOR_BIN   Harbor executable (default: harbor found on PATH)
  KEEP_STAGE   Set to 1 to retain and print the temporary staged task directory
EOF
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  [[ $# -ge 1 ]] && exit 0
  exit 2
fi

if [[ -z "${DAYTONA_API_KEY:-}" ]]; then
  echo "DAYTONA_API_KEY is not set." >&2
  exit 2
fi

task_id=$1
shift

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_task="$script_dir/tasks/$task_id"
runtime_archive="$script_dir/runtime/toolathlon-json-runtime-src.tar.gz"
harbor_bin=${HARBOR_BIN:-harbor}

if [[ ! -d "$source_task" ]]; then
  echo "Task not found: $source_task" >&2
  exit 2
fi
if [[ ! -f "$runtime_archive" ]]; then
  echo "Runtime archive not found: $runtime_archive" >&2
  exit 2
fi
if ! command -v "$harbor_bin" >/dev/null 2>&1; then
  echo "Harbor executable not found: $harbor_bin" >&2
  echo "Set HARBOR_BIN to its path, for example .venv/bin/harbor." >&2
  exit 2
fi

stage_root=$(mktemp -d "${TMPDIR:-/tmp}/toolathlon-daytona.XXXXXX")
cleanup() {
  if [[ "${KEEP_STAGE:-0}" == "1" ]]; then
    echo "Staged task retained at: $stage_root" >&2
  else
    rm -rf -- "$stage_root"
  fi
}
trap cleanup EXIT

staged_task="$stage_root/tasks/$task_id"
staged_environment="$staged_task/environment"
runtime_source="$staged_environment/runtime-src"

mkdir -p "$stage_root/tasks"
cp -a -- "$source_task" "$staged_task"
mkdir -p "$runtime_source"
tar --warning=no-unknown-keyword -xzf "$runtime_archive" \
  --strip-components=1 -C "$runtime_source"

original_dockerfile="$staged_environment/Dockerfile.task"
mv -- "$staged_environment/Dockerfile" "$original_dockerfile"
{
  cat <<'EOF'
FROM python:3.13-slim

WORKDIR /opt/runtime
COPY runtime-src/pyproject.toml ./
COPY runtime-src/t3 ./t3
RUN pip install --no-cache-dir . && mkdir -p /logs /agent_workspace

RUN mkdir -p /etc/claude-code
COPY runtime-src/managed-settings.json /etc/claude-code/managed-settings.json
EOF
  sed '/^[[:space:]]*FROM[[:space:]]/d' "$original_dockerfile"
} > "$staged_environment/Dockerfile"
rm -- "$original_dockerfile"

task_toml_tmp="$staged_task/task.toml.daytona"
awk '!/^[[:space:]]*docker_image[[:space:]]*=/' "$staged_task/task.toml" \
  > "$task_toml_tmp"
mv -- "$task_toml_tmp" "$staged_task/task.toml"

if [[ $# -eq 0 ]]; then
  set -- --agent oracle
fi

echo "Running $task_id on Daytona..." >&2
"$harbor_bin" run \
  --path "$stage_root/tasks" \
  -i "$task_id" \
  --env daytona \
  --no-force-build \
  --ek auto_snapshot=true \
  "$@"
