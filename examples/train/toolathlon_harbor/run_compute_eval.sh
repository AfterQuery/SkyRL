#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${COMPUTE_API_KEY:?Set COMPUTE_API_KEY to an AfterQuery Compute API key}"
export COMPUTE_API_URL="${COMPUTE_API_URL:-https://compute-api.afterquery.com}"
export COMPUTE_IMAGE_REGISTRY="${COMPUTE_IMAGE_REGISTRY:-us-docker.pkg.dev/afterquery-compute/compute-images}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker with Buildx is required to build and push Harbor task images." >&2
  exit 2
fi

exec "$HERE/run_eval.sh" --env compute "$@"
