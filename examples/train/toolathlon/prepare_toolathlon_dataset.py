"""Prepare train/val task lists for Toolathlon training.

Scans the Toolathlon tasks directory, optionally filters tasks by the MCP servers they
require, and writes newline-separated task-list files consumed by ToolathlonTaskDataset.

Examples:
    # All 109 tasks, 90/10 split:
    uv run examples/train/toolathlon/prepare_toolathlon_dataset.py \
        --toolathlon-repo /home/ubuntu/Toolathlon --output-dir ~/data/toolathlon

    # Only tasks whose MCP servers are all in the allowlist (no external accounts needed):
    uv run examples/train/toolathlon/prepare_toolathlon_dataset.py \
        --toolathlon-repo /home/ubuntu/Toolathlon --output-dir ~/data/toolathlon \
        --include-only-servers filesystem,arxiv_local,scholarly,fetch,pdf_reader
"""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--toolathlon-repo", required=True, help="Path to the Toolathlon repository checkout.")
    parser.add_argument("--tasks-domain", default="finalpool", help="Subdirectory of <repo>/tasks holding tasks.")
    parser.add_argument("--output-dir", required=True, help="Directory to write train_tasks.txt / val_tasks.txt.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of tasks held out for eval.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for the split.")
    parser.add_argument(
        "--include-only-servers",
        default=None,
        help="Comma-separated MCP server allowlist; keep only tasks whose needed_mcp_servers "
        "are all in this list. Useful to restrict to tasks that need no external accounts.",
    )
    parser.add_argument(
        "--exclude-servers",
        default=None,
        help="Comma-separated MCP server denylist; drop tasks needing any of these servers.",
    )
    args = parser.parse_args()

    tasks_dir = Path(args.toolathlon_repo).expanduser() / "tasks" / args.tasks_domain
    if not tasks_dir.is_dir():
        raise SystemExit(f"Tasks directory not found: {tasks_dir}")

    include_only = set(args.include_only_servers.split(",")) if args.include_only_servers else None
    exclude = set(args.exclude_servers.split(",")) if args.exclude_servers else set()

    tasks = []
    skipped = []
    for task_dir in sorted(tasks_dir.iterdir()):
        config_file = task_dir / "task_config.json"
        if not config_file.is_file():
            continue
        servers = set(json.loads(config_file.read_text()).get("needed_mcp_servers", []))
        if include_only is not None and not servers <= include_only:
            skipped.append((task_dir.name, sorted(servers - include_only)))
            continue
        if servers & exclude:
            skipped.append((task_dir.name, sorted(servers & exclude)))
            continue
        tasks.append(task_dir.name)

    if not tasks:
        raise SystemExit("No tasks left after filtering.")
    for name, reason in skipped:
        print(f"Skipped {name}: uses {reason}")

    rng = random.Random(args.seed)
    rng.shuffle(tasks)
    num_val = max(1, int(len(tasks) * args.val_fraction)) if args.val_fraction > 0 else 0
    val_tasks = sorted(tasks[:num_val])
    train_tasks = sorted(tasks[num_val:])

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_tasks.txt").write_text("\n".join(train_tasks) + "\n")
    (output_dir / "val_tasks.txt").write_text(("\n".join(val_tasks) + "\n") if val_tasks else "")

    print(f"Wrote {len(train_tasks)} train tasks and {len(val_tasks)} val tasks to {output_dir}")


if __name__ == "__main__":
    main()
