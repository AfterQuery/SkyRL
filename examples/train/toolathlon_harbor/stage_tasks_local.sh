#!/usr/bin/env bash
set -euo pipefail

# Stage Toolathlon tasks from shared storage onto node-local storage before a
# Slurm training run. Local staging avoids repeatedly reading thousands of small
# task files from the shared filesystem. The source and copied destination must
# each contain TOOLATHLON_EXPECTED_TASKS task.toml files (1,900 by default).
#
# rsync intentionally updates the destination in place without --delete. Files
# left over from an earlier staging run are therefore retained; extra task.toml
# files are detected by the destination count, but other stale files are not.

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 SHARED_TASKS_DIR LOCAL_TASKS_DIR" >&2
  exit 2
fi

source_dir=$1
destination_dir=$2
expected_tasks=${TOOLATHLON_EXPECTED_TASKS:-1900}

if [[ ! -d "$source_dir" ]]; then
  echo "Shared Toolathlon task directory is missing: $source_dir" >&2
  exit 2
fi

source_count=$(find "$source_dir" -mindepth 2 -maxdepth 2 -name task.toml -type f | wc -l)
if [[ "$source_count" -ne "$expected_tasks" ]]; then
  echo "Expected $expected_tasks shared tasks, found $source_count in $source_dir" >&2
  exit 2
fi

mkdir -p "$destination_dir"
rsync -a "$source_dir/" "$destination_dir/"

destination_count=$(find "$destination_dir" -mindepth 2 -maxdepth 2 -name task.toml -type f | wc -l)
if [[ "$destination_count" -ne "$expected_tasks" ]]; then
  echo "Expected $expected_tasks staged tasks, found $destination_count in $destination_dir" >&2
  exit 1
fi

echo "node=$(hostname) staged_tasks=$destination_count destination=$destination_dir"
