"""Turn a directory of Harbor-format tasks into SkyRL parquet, pre-building task images.

Mirrors ``examples/train/mini_swe_agent/preprocess_swegym.py``'s output schema
(``data_source`` / ``prompt`` / ``env_class`` / ``instance``) so the generator's contract is
unchanged. The ``instance`` dict carries the four fields the pipeline reads:

    instance_id        -- task dir name; used for image naming and trajectory filenames
    problem_statement  -- instruction.md; rendered by mini-swe-agent's instance_template
    image_name         -- pre-built tag. get_docker_image_name() short-circuits on this, so
                          the SWE-Bench registry-naming branch never runs
    tests_dir          -- absolute path to tests/, uploaded into the container at grade time

Images are built **here**, ahead of training, rather than in the rollout path: a group of
``n_samples_per_prompt`` Ray tasks would otherwise race N identical ``podman build`` calls.

Usage::

    uv run --isolated examples/train/mini_swe_agent_harbor/preprocess_harbor.py \\
        --tasks_dir ~/data/harbor_tasks/CodeContests \\
        --output_dir ~/data/harbor_codecontests

For a multi-node Ray cluster, ``--tasks_dir`` must be on shared storage (the grader reads
``tests_dir`` from whichever worker runs the trajectory), and the images must exist on every
node -- either build on each node or push to a registry and pass ``--image_prefix``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.train.mini_swe_agent_harbor.harbor_tasks import (  # noqa: E402
    DEFAULT_IMAGE_PREFIX,
    HarborTask,
    build_task_image,
    discover_tasks,
    image_tag,
)


def build_row(task: HarborTask, data_source: str, image_name: str) -> dict:
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": task.instruction}],
        # mini-swe-agent owns the loop, so there is no skyrl_gym environment.
        "env_class": "null",
        "instance": {
            "instance_id": task.name,
            "problem_statement": task.instruction,
            "image_name": image_name,
            "tests_dir": str(task.tests_dir),
            "task_dir": str(task.task_dir),
            # Surfaced for convenience; the generator reads timeouts from generator_cfg.
            "agent_timeout_sec": task.agent_timeout(0.0),
            "verifier_timeout_sec": task.verifier_timeout(0.0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_dir", default="~/data/harbor_tasks/CodeContests")
    parser.add_argument("--output_dir", default="~/data/harbor_codecontests")
    parser.add_argument("--data_source", default=None, help="defaults to harbor/<tasks dir name>")
    parser.add_argument("--val_size", type=int, default=32, help="tasks held out for validation")
    parser.add_argument("--limit", type=int, default=None, help="keep only the first N tasks")
    parser.add_argument("--executable", default="podman", help="container tool for builds")
    parser.add_argument("--image_prefix", default=DEFAULT_IMAGE_PREFIX)
    parser.add_argument("--build_timeout", type=float, default=1800.0)
    parser.add_argument(
        "--skip_build",
        action="store_true",
        help="only write parquet; assume images already exist (e.g. pulled from a registry)",
    )
    args = parser.parse_args()

    tasks_dir = Path(os.path.expanduser(args.tasks_dir))
    tasks = discover_tasks(tasks_dir, limit=args.limit)
    if not tasks:
        raise SystemExit(f"no Harbor tasks found under {tasks_dir}")
    print(f"found {len(tasks)} tasks under {tasks_dir}")

    if args.val_size >= len(tasks):
        raise SystemExit(f"--val_size {args.val_size} leaves no training tasks ({len(tasks)} total)")

    data_source = args.data_source or f"harbor/{tasks_dir.name.lower()}"
    rows, failed = [], []
    for index, task in enumerate(tasks, 1):
        tag = image_tag(task, args.image_prefix)
        if not args.skip_build:
            try:
                tag = build_task_image(
                    task,
                    executable=args.executable,
                    prefix=args.image_prefix,
                    build_timeout=args.build_timeout,
                )
            except Exception as e:  # noqa: BLE001 - one bad Dockerfile shouldn't stop the set
                print(f"[{index}/{len(tasks)}] BUILD FAILED {task.name}: {str(e)[:200]}")
                failed.append(task.name)
                continue
        rows.append(build_row(task, data_source, tag))
        if index % 50 == 0 or index == len(tasks):
            print(f"[{index}/{len(tasks)}] prepared {len(rows)} tasks, {len(failed)} failed")

    if not rows:
        raise SystemExit("every task failed to build; nothing to write")
    if failed:
        # Loud, because a silently smaller dataset is easy to mistake for a working run.
        print(f"\nWARNING: dropped {len(failed)} tasks whose image failed to build: {failed[:10]}")

    val_rows, train_rows = rows[: args.val_size], rows[args.val_size :]
    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, split_rows in (("train", train_rows), ("validation", val_rows)):
        path = output_dir / f"{name}.parquet"
        datasets.Dataset.from_list(split_rows).to_parquet(str(path))
        print(f"wrote {len(split_rows)} rows to {path}")


if __name__ == "__main__":
    main()
