"""Prepare train/val parquet files for MCP-Atlas training.

Downloads the 500-task ScaleAI/MCP-Atlas dataset from HuggingFace, optionally filters tasks,
and writes parquet files in SkyRL's standard PromptDataset format:
``prompt`` (chat messages), ``env_class`` (null), plus ``task_id`` / ``enabled_tools_json`` /
``gtfa_claims_json`` columns that reach the generator as env_extras.

Examples:
    # All 500 tasks, 90/10 split:
    uv run examples/train/mcp_atlas/prepare_mcp_atlas_dataset.py --output-dir ~/data/mcp_atlas

    # Only tasks whose tools are all served by the running sandbox (drops tasks needing
    # API-key-gated servers that are offline), and that use no state-mutating tools:
    uv run examples/train/mcp_atlas/prepare_mcp_atlas_dataset.py --output-dir ~/data/mcp_atlas \
        --available-tools-only --exclude-mutating
"""

import argparse
import ast
import json
import random
import urllib.request
from pathlib import Path

# Tool-name prefixes that mutate shared sandbox state (filesystem, memory graph, git trees,
# external DBs/SaaS). Rollouts of such tasks can interfere across a shared sandbox.
MUTATING_TOOL_PREFIXES = (
    "filesystem_write",
    "filesystem_edit",
    "filesystem_move",
    "filesystem_create",
    "memory_create",
    "memory_add",
    "memory_delete",
    "git_add",
    "git_commit",
    "git_checkout",
    "git_create",
    "git_reset",
    "desktop-commander_write",
    "desktop-commander_edit",
    "desktop-commander_start",
    "desktop-commander_interact",
    "mcp-code-executor_",
    "mcp-server-code-runner_",
    "mongodb_insert",
    "mongodb_update",
    "mongodb_delete",
    "notion_API-post",
    "notion_API-patch",
    "slack_conversations_add",
    "slack_chat_post",
)


def parse_enabled_tools(raw: str) -> list:
    """Normalize ENABLED_TOOLS: JSON list of names, list of {"name": ...} dicts, or comma string."""
    try:
        tools = json.loads(raw)
    except json.JSONDecodeError:
        return [t.strip() for t in raw.split(",") if t.strip()]
    return [t["name"] if isinstance(t, dict) else str(t) for t in tools]


def parse_claims(raw: str) -> list:
    """GTFA_CLAIMS is a Python-repr list; fall back through literal_eval and JSON."""
    for parser in (ast.literal_eval, json.loads):
        try:
            claims = parser(raw)
            if isinstance(claims, list):
                return [str(c) for c in claims if len(str(c).strip()) > 3]
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Directory to write train.parquet / val.parquet.")
    parser.add_argument("--dataset", default="ScaleAI/MCP-Atlas", help="HuggingFace dataset name.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of tasks held out for eval.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for the split.")
    parser.add_argument(
        "--available-tools-only",
        action="store_true",
        help="Keep only tasks whose enabled tools are all served by the running sandbox "
        "(drops tasks that need API-key-gated servers that are offline).",
    )
    parser.add_argument("--sandbox-url", default="http://localhost:1984", help="Sandbox URL for the tool check.")
    parser.add_argument(
        "--exclude-mutating",
        action="store_true",
        help="Drop tasks whose enabled tools can mutate shared sandbox state (filesystem writes, "
        "memory graph, git, DB/SaaS writes). Recommended for parallel rollouts on one sandbox.",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train")

    available_tools = None
    if args.available_tools_only:
        req = urllib.request.Request(f"{args.sandbox_url.rstrip('/')}/list-tools", method="POST")
        with urllib.request.urlopen(req, timeout=180) as resp:
            available_tools = {t["name"] for t in json.loads(resp.read())}
        print(f"Sandbox serves {len(available_tools)} tools")

    rows, skipped = [], []
    for r in ds:
        task_id = r["TASK"]
        enabled_tools = parse_enabled_tools(r["ENABLED_TOOLS"])
        claims = parse_claims(r["GTFA_CLAIMS"])
        if not claims:
            skipped.append((task_id, "no parseable claims"))
            continue
        if available_tools is not None:
            missing = [t for t in enabled_tools if t not in available_tools]
            if missing:
                skipped.append((task_id, f"tools unavailable: {missing[:3]}"))
                continue
        if args.exclude_mutating:
            mutating = [t for t in enabled_tools if t.startswith(MUTATING_TOOL_PREFIXES)]
            if mutating:
                skipped.append((task_id, f"mutating tools: {mutating[:3]}"))
                continue
        rows.append(
            {
                "prompt": [{"role": "user", "content": r["PROMPT"]}],
                "env_class": None,
                "task_id": task_id,
                "enabled_tools_json": json.dumps(enabled_tools),
                "gtfa_claims_json": json.dumps(claims),
            }
        )

    if not rows:
        raise SystemExit("No tasks left after filtering.")
    print(f"Kept {len(rows)} tasks, skipped {len(skipped)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    num_val = max(1, int(len(rows) * args.val_fraction)) if args.val_fraction > 0 else 0
    val_rows, train_rows = rows[:num_val], rows[num_val:]

    import pandas as pd

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(output_dir / "train.parquet", index=False)
    if val_rows:
        pd.DataFrame(val_rows).to_parquet(output_dir / "val.parquet", index=False)
    print(f"Wrote {len(train_rows)} train / {len(val_rows)} val tasks to {output_dir}")


if __name__ == "__main__":
    main()
