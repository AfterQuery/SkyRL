#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

TASKS_DIR="${TOOLATHLON_TASKS_DIR:-$REPO_ROOT/toolathlon-tasks/tasks/tasks}"
RUNTIME_ARCHIVE="${TOOLATHLON_RUNTIME_ARCHIVE:-$REPO_ROOT/toolathlon-tasks/tasks/runtime/toolathlon-json-runtime-src.tar.gz}"
RUNTIME_IMAGE="${TOOLATHLON_RUNTIME_IMAGE:-toolathlon-json-runtime:v1}"
API_BASE="${TOOLATHLON_API_BASE:?Set TOOLATHLON_API_BASE to an OpenAI-compatible /v1 endpoint}"
API_KEY="${TOOLATHLON_API_KEY:-dummy}"
MODEL="${TOOLATHLON_MODEL:?Set TOOLATHLON_MODEL to the model name served by the endpoint}"

if [[ ! -d "$TASKS_DIR" ]]; then
  echo "Toolathlon tasks directory not found: $TASKS_DIR" >&2
  exit 2
fi

if ! docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1; then
  if [[ ! -f "$RUNTIME_ARCHIVE" ]]; then
    echo "Toolathlon runtime archive not found: $RUNTIME_ARCHIVE" >&2
    exit 2
  fi
  BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/toolathlon-runtime.XXXXXX")"
  trap 'rm -rf -- "$BUILD_DIR"' EXIT
  tar xzf "$RUNTIME_ARCHIVE" -C "$BUILD_DIR" --strip-components=1
  docker build -t "$RUNTIME_IMAGE" "$BUILD_DIR"
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec harbor run \
  --path "$TASKS_DIR" \
  --agent-import-path examples.train_integrations.harbor.mcp_agent:HarborMCPAgent \
  --model "$MODEL" \
  --mcp-config "$HERE/mcp.json" \
  --agent-env "OPENAI_BASE_URL=$API_BASE" \
  --agent-env "OPENAI_API_KEY=$API_KEY" \
  "$@"
