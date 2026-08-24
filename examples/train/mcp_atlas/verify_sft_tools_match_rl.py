"""Assert that every SFT row offers exactly the tools RL serves for that task.

The menu is part of the prompt, so a divergence here is a silent train/serve mismatch. It has
already happened once: the prep script built menus from ``task.toml``'s curated ``enabled_tools``
while the runner offers whatever the container gateway's ``/list-tools`` returns, which left the
student trained on ~16 tools and served ~32, and 45% of rows demonstrating a call to a tool the
row never offered.

Ground truth is ``trajectory.rollout.prompt_token_ids`` from a real Harbor run -- the literal
token array vLLM was fed, not a reconstruction of it. This decodes those tokens, extracts the
``<tools>`` block, and compares names **and order** against the emitted parquet.

Only tasks present in both the dataset and the run can be checked, so the report states its own
coverage rather than implying the whole set was verified.

Usage::

    uv run --isolated --with pandas --with pyarrow --with transformers \\
        examples/train/mcp_atlas/verify_sft_tools_match_rl.py \\
        --dataset ~/data/mcp_atlas_sft/train.parquet \\
        --harbor-run ~/mcp_atlas_harbor_run/trials_run \\
        --tokenizer Qwen/Qwen3-30B-A3B
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

TOOLS_BLOCK = re.compile(r"<tools>\n(.*?)\n</tools>", re.S)


def tool_names(rendered: str) -> List[str]:
    """Tool names in prompt order, from a rendered ``<tools>`` block."""
    match = TOOLS_BLOCK.search(rendered)
    if not match:
        return []
    names = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            names.append(json.loads(line)["function"]["name"])
        except (json.JSONDecodeError, KeyError, TypeError):
            # A malformed entry is itself a finding; surface it rather than skipping silently.
            names.append(f"<unparsable: {line[:60]}>")
    return names


def rl_menus(harbor_run: Path, tokenizer) -> Dict[str, List[str]]:
    """task_id -> tool names RL actually served, decoded from the tokens vLLM was fed."""
    out: Dict[str, List[str]] = {}
    for path in glob.glob(str(harbor_run / "*" / "agent" / "trajectory.json")):
        # Trial dirs are '<task_id>__<suffix>'; several rollouts share one task.
        task_id = os.path.basename(os.path.dirname(os.path.dirname(path))).split("__")[0]
        if task_id in out:
            continue
        try:
            rollout = (json.load(open(path)).get("rollout") or {}).get("prompt_token_ids")
        except (json.JSONDecodeError, OSError):
            continue
        if not rollout:
            continue
        first = rollout[0] if isinstance(rollout[0], list) else rollout
        names = tool_names(tokenizer.decode(first))
        if names:
            out[task_id] = names
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=Path("~/data/mcp_atlas_sft/train.parquet"))
    parser.add_argument(
        "--harbor-run",
        type=Path,
        default=Path("~/mcp_atlas_harbor_run/trials_run"),
        help="Harbor trials directory holding agent/trajectory.json with rollout_details.",
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument(
        "--min-tasks",
        type=int,
        default=20,
        help="Fail if fewer than this many tasks could be compared. A run that overlaps the "
        "dataset in only a handful of tasks proves little, and silently passing on 2 tasks "
        "would be worse than reporting that the check did not run.",
    )
    args = parser.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = pd.read_parquet(str(args.dataset).replace("~", os.path.expanduser("~")))
    served = rl_menus(Path(os.path.expanduser(str(args.harbor_run))), tokenizer)
    print(f"RL menus recovered for {len(served)} distinct tasks")

    # The dataset may be pretokenized (input_ids) or online (tools JSON); handle both.
    sft: Dict[str, List[str]] = {}
    for _, row in dataset.iterrows():
        task_id = row["task_id"]
        if "tools" in dataset.columns:
            sft[task_id] = [t["function"]["name"] for t in json.loads(row["tools"])]
        else:
            sft[task_id] = tool_names(tokenizer.decode(list(row["input_ids"])))
    print(f"SFT rows: {len(dataset)} covering {len(sft)} tasks")

    shared = sorted(set(sft) & set(served))
    print(f"comparable (in both): {len(shared)} tasks")
    if len(shared) < args.min_tasks:
        print(
            f"FAIL: only {len(shared)} tasks overlap, below --min-tasks {args.min_tasks}. "
            "Point --harbor-run at a run that covers the training tasks.",
            file=sys.stderr,
        )
        sys.exit(2)

    set_mismatch: List[str] = []
    order_mismatch: List[str] = []
    example: Optional[tuple] = None
    for task_id in shared:
        want, got = served[task_id], sft[task_id]
        if set(want) != set(got):
            set_mismatch.append(task_id)
            if example is None:
                example = (
                    task_id,
                    sorted(set(want) - set(got))[:5],
                    sorted(set(got) - set(want))[:5],
                )
        elif want != got:
            order_mismatch.append(task_id)

    print(f"\n  exact name-set match : {len(shared) - len(set_mismatch)}/{len(shared)}")
    print(f"  exact order match    : {len(shared) - len(set_mismatch) - len(order_mismatch)}/{len(shared)}")
    if example:
        print(f"\n  e.g. {example[0]}\n    RL serves but SFT omits : {example[1]}\n"
              f"    SFT offers but RL does not: {example[2]}")
    if order_mismatch:
        print(f"  order differs on: {order_mismatch[:5]}")

    if set_mismatch or order_mismatch:
        print(
            f"\nFAIL: {len(set_mismatch)} set and {len(order_mismatch)} order mismatches. The SFT "
            "prompts do not match what RL serves.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"\nPASS: all {len(shared)} comparable tasks offer exactly the tools RL serves, in order.")


if __name__ == "__main__":
    main()
