"""Convert GLM-5.2 teacher trajectories into an SFT dataset for warm-starting a smaller policy.

Reads the ``AQ-MCP-Atlas-1000-Trajectories-GLM-5.2`` bundle (3 rollouts x 1000 Harbor tasks,
each graded by claim-coverage LLM judge) plus the matching ``AQ-MCP-Atlas-1000-Tasks`` bundle,
keeps only high-coverage rollouts, and writes parquet in the shape ``SFTTrainer`` expects:
``messages`` (OpenAI format), ``tools`` (JSON-encoded function schemas), and ``task_id`` /
``coverage`` metadata columns.

Two conversions matter for correctness:

1. **Anthropic -> OpenAI messages.** The bundle stores assistant ``content`` as a block list
   (``text`` / ``tool_use`` with a JSON-string ``input``) and tool results under
   ``tool_use_id``. ``SFTTrainer`` tokenizes with ``apply_chat_template``, which wants
   ``content`` as text plus a separate ``tool_calls`` list, and ``tool_call_id`` on tool
   messages.
2. **Tool observations are flattened the same way the RL generator flattens them** — MCP
   content blocks joined on their ``text`` fields and capped at ``--max-tool-output-chars``
   (the generator's ``tool_output_cap`` default). Warm-starting on a different observation
   format than RL will produce is the main avoidable train/rollout mismatch.

Tool JSON schemas are **not** shipped in either bundle (they live in the
``mcp-atlas-runtime`` image), so parameter schemas are reconstructed from the argument keys
the teacher actually used, unioned per tool name across the selected rollouts. Every tool in
the task's ``enabled_tools`` is included -- including distractors the teacher never called,
which get an empty parameter object -- because the teacher chose among the full list and the
student should face the same choice.

Usage::

    uv run examples/train/mcp_atlas/prepare_glm_sft_dataset.py \\
        --trajectories-dir ~/AQ-MCP-Atlas-1000-Trajectories-GLM-5.2 \\
        --tasks-dir ~/AQ-MCP-Atlas-1000-Tasks \\
        --output-dir ~/data/mcp_atlas_sft \\
        --min-coverage 0.75 --one-per-task
"""

import argparse
import collections
import csv
import gzip
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_JSON_TYPE = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
    list: "array",
    dict: "object",
}


def _parse_toml_string_list(text: str, key: str) -> List[str]:
    """Extract a TOML array-of-strings value without a TOML dependency.

    task.toml is machine-generated with one quoted entry per line, so a bracket scan is
    sufficient and avoids adding tomli/tomllib version handling.
    """
    start = text.find(f"{key} = [")
    if start == -1:
        return []
    end = text.find("]", start)
    if end == -1:
        return []
    body = text[start + len(f"{key} = [") : end]
    return [part.strip().strip('",').strip('"') for part in body.split("\n") if part.strip().strip(",").strip()]


def _flatten_tool_result(raw: Any, cap: Optional[int]) -> str:
    """Flatten an MCP tool result into observation text, mirroring the RL generator."""
    text = raw if isinstance(raw, str) else json.dumps(raw)
    try:
        blocks = json.loads(text)
        if isinstance(blocks, list):
            joined = "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
            if joined:
                text = joined
    except (json.JSONDecodeError, TypeError):
        pass
    if cap and len(text) > cap:
        text = text[:cap] + f"\n[... truncated to {cap} characters]"
    return text


def convert_messages(raw_messages: List[Dict[str, Any]], cap: Optional[int]) -> Tuple[List[Dict[str, Any]], int]:
    """Convert one bundle transcript to OpenAI-format messages.

    Returns the messages and the number of tool calls whose result was missing from the
    transcript (dangling calls are kept -- predicting the call is valid signal -- but their
    absent observations are simply not emitted).
    """
    # Index tool results by the call id they answer.
    results: Dict[str, Dict[str, Any]] = {}
    for msg in raw_messages:
        if msg.get("role") == "tool":
            results[msg.get("tool_use_id")] = msg

    out: List[Dict[str, Any]] = []
    missing = 0
    for msg in raw_messages:
        role = msg.get("role")
        if role == "user":
            out.append({"role": "user", "content": msg.get("content") or ""})
        elif role == "tool":
            continue  # emitted alongside the assistant turn that requested it
        elif role == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                out.append({"role": "assistant", "content": content})
                continue
            texts, tool_calls = [], []
            for block in content or []:
                btype = block.get("type")
                if btype == "text":
                    texts.append(block.get("text") or "")
                elif btype == "tool_use":
                    raw_input = block.get("input")
                    arguments = raw_input if isinstance(raw_input, str) else json.dumps(raw_input or {})
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {"name": block.get("name"), "arguments": arguments},
                        }
                    )
            assistant: Dict[str, Any] = {"role": "assistant", "content": "\n".join(t for t in texts if t)}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            out.append(assistant)
            # Append this turn's observations in call order.
            for call in tool_calls:
                result = results.get(call["id"])
                if result is None:
                    missing += 1
                    continue
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": _flatten_tool_result(result.get("content"), cap),
                    }
                )
        else:
            raise ValueError(f"Unexpected role {role!r} in trajectory")
    return out, missing


def build_tool_schemas(enabled_tools: List[str], observed_args: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """Build OpenAI function schemas for a task's enabled tools from observed argument usage."""
    schemas = []
    for name in enabled_tools:
        props = {k: {"type": t} for k, t in sorted(observed_args.get(name, {}).items())}
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"MCP tool {name}.",
                    "parameters": {"type": "object", "properties": props},
                },
            }
        )
    return schemas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories-dir", required=True, help="Extracted GLM trajectories bundle.")
    parser.add_argument("--tasks-dir", required=True, help="Extracted AQ-MCP-Atlas-1000-Tasks bundle.")
    parser.add_argument("--output-dir", default="~/data/mcp_atlas_sft", help="Where to write train/val parquet.")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.75,
        help="Keep only rollouts whose judge coverage is >= this (the bundle's pass threshold).",
    )
    parser.add_argument(
        "--one-per-task",
        action="store_true",
        help="Keep only the highest-coverage rollout per task, so easy tasks with 3 good runs "
        "do not dominate the mixture.",
    )
    parser.add_argument(
        "--prefer",
        choices=["shortest", "longest", "first"],
        default="shortest",
        help="Tie-break for --one-per-task when several rollouts share the top coverage. "
        "'shortest' keeps the fewest-tool-call run (same reward for less work, and it trims the "
        "token weight easy tasks get under per-token loss normalization).",
    )
    parser.add_argument(
        "--max-tool-output-chars",
        type=int,
        default=10000,
        help="Cap per tool observation; match generator tool_output_cap so SFT and RL agree. 0 = uncapped.",
    )
    parser.add_argument("--max-tool-calls", type=int, default=60, help="Drop rollouts with more calls than this.")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-8B",
        help="Tokenizer used only to report sequence-length stats for choosing max_length.",
    )
    args = parser.parse_args()

    traj_dir = Path(args.trajectories_dir).expanduser().resolve()
    tasks_dir = Path(args.tasks_dir).expanduser().resolve()
    index_path = traj_dir / "index.csv"
    if not index_path.is_file():
        raise SystemExit(f"index.csv not found under {traj_dir}")
    cap = args.max_tool_output_chars or None

    # 1. Select rollouts from the index (no decompression needed).
    rows = list(csv.DictReader(index_path.open()))
    selected = []
    for row in rows:
        if row["status"] != "graded":
            continue
        coverage = row.get("coverage")
        if not coverage:
            continue
        coverage = float(coverage)
        if coverage < args.min_coverage:
            continue
        if args.max_tool_calls and int(row["tool_calls"] or 0) > args.max_tool_calls:
            continue
        selected.append((row["task_id"], int(row["run"]), coverage, row["file"], int(row["tool_calls"] or 0)))

    if args.one_per_task:
        # Rank by coverage first, then break ties with --prefer. Ties are the common case at
        # coverage 1.0: 256 of the 534 perfect tasks have all three rollouts perfect.
        def sort_key(item):
            _, run, coverage, _, n_calls = item
            if args.prefer == "shortest":
                return (-coverage, n_calls, run)
            if args.prefer == "longest":
                return (-coverage, -n_calls, run)
            return (-coverage, run)

        best: Dict[str, tuple] = {}
        for item in sorted(selected, key=sort_key):
            best.setdefault(item[0], item)
        selected = list(best.values())

    if not selected:
        raise SystemExit("No rollouts passed the filters.")
    print(
        f"Selected {len(selected)} rollouts from {len(rows)} "
        f"(status=graded, coverage>={args.min_coverage}"
        f"{', one per task' if args.one_per_task else ''}) "
        f"covering {len({t for t, _, _, _, _ in selected})} tasks"
    )

    # 2. Load each task's enabled_tools (includes distractors the teacher had to reject).
    enabled_by_task: Dict[str, List[str]] = {}
    for task_id in {t for t, _, _, _, _ in selected}:
        toml_path = tasks_dir / "tasks" / task_id / "task.toml"
        if not toml_path.is_file():
            continue
        enabled_by_task[task_id] = _parse_toml_string_list(toml_path.read_text(), "enabled_tools")
    missing_tasks = [t for t, _, _, _, _ in selected if t not in enabled_by_task]
    if missing_tasks:
        print(f"WARNING: {len(set(missing_tasks))} tasks missing task.toml; their rows get tools=[]")

    # 3. Pass one: convert transcripts and collect observed argument types per tool name.
    observed_args: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
    converted: List[tuple] = []
    total_missing_results = 0
    skipped = 0
    for task_id, run, coverage, relpath, _ in selected:
        path = traj_dir / relpath
        if not path.is_file():
            skipped += 1
            continue
        traj = json.load(gzip.open(path, "rt"))
        try:
            messages, missing = convert_messages(traj["messages"], cap)
        except ValueError as exc:
            print(f"WARNING: skipping {task_id} run{run}: {exc}")
            skipped += 1
            continue
        total_missing_results += missing
        if not any(m["role"] == "assistant" for m in messages):
            skipped += 1
            continue
        for call in traj.get("tool_calls", []):
            name, call_args = call.get("name"), call.get("arguments")
            if not name or not isinstance(call_args, dict):
                continue
            for key, value in call_args.items():
                observed_args[name].setdefault(key, _JSON_TYPE.get(type(value), "string"))
        converted.append((task_id, run, coverage, messages))

    print(f"Converted {len(converted)} rollouts (skipped {skipped}); {total_missing_results} tool calls had no result")
    print(f"Reconstructed parameter schemas for {len(observed_args)} distinct tools from teacher usage")

    # 4. Pass two: attach per-task tool schemas and emit rows.
    out_rows = []
    for task_id, run, coverage, messages in converted:
        enabled = enabled_by_task.get(task_id, [])
        out_rows.append(
            {
                "messages": messages,
                "tools": json.dumps(build_tool_schemas(enabled, observed_args)),
                "task_id": task_id,
                "run_index": run,
                "coverage": coverage,
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(out_rows)
    num_val = int(len(out_rows) * args.val_fraction) if args.val_fraction > 0 else 0
    val_rows, train_rows = out_rows[:num_val], out_rows[num_val:]

    import pandas as pd

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(output_dir / "train.parquet", index=False)
    if val_rows:
        # Named "validation" so `load_dataset(<dir>)` exposes it under that split name.
        pd.DataFrame(val_rows).to_parquet(output_dir / "validation.parquet", index=False)
    print(f"Wrote {len(train_rows)} train / {len(val_rows)} val rows to {output_dir}")

    # 5. Report token lengths so max_length can be set without guessing.
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        sample = out_rows if len(out_rows) <= 400 else rng.sample(out_rows, 400)
        lengths = []
        for row in sample:
            # return_dict=False keeps this a plain id list; with return_dict the call yields a
            # BatchEncoding whose len() is its key count, not the sequence length.
            ids = tokenizer.apply_chat_template(
                row["messages"],
                tools=json.loads(row["tools"]) or None,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=False,
            )
            lengths.append(len(ids))
        lengths.sort()
        pct = lambda p: lengths[min(len(lengths) - 1, int(len(lengths) * p))]  # noqa: E731
        print(
            f"Token lengths over {len(lengths)} sampled rows ({args.tokenizer}): "
            f"median={pct(0.5)} p90={pct(0.9)} p99={pct(0.99)} max={lengths[-1]}"
        )
        for limit in (8192, 16384, 32768):
            print(f"  fit within max_length={limit}: {sum(1 for n in lengths if n <= limit)}/{len(lengths)}")
    except Exception as exc:  # noqa: BLE001
        print(f"(skipped token-length report: {exc})")


if __name__ == "__main__":
    main()
