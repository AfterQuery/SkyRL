"""Convert GLM-5.2 teacher trajectories into an SFT dataset for warm-starting a smaller policy.

Reads the ``AQ-MCP-Atlas-1000-Full-Delivery`` bundle (3 rollouts x 1000 Harbor tasks, each
graded by the claim-coverage LLM judge) plus the matching task packages, keeps high-coverage
rollouts, and writes parquet in the shape ``SFTTrainer`` expects: ``messages`` (OpenAI
format), ``tools`` (JSON-encoded function schemas), and ``task_id`` / ``coverage`` metadata.

The full-delivery bundle was produced by this repo's own ``mcp-atlas`` agent, which changes
three things versus the earlier trajectories-only bundle:

1. **No message conversion is needed.** ``trajectory.messages`` is already OpenAI format --
   string ``content``, ``tool_calls`` carrying ``function.name`` plus a JSON-string
   ``function.arguments``, and ``tool`` messages keyed by ``tool_call_id``.

2. **Tool observations are kept verbatim, not flattened.** The runner stores each result as
   the raw MCP block-list JSON and feeds exactly that back to the model, so flattening it
   here would train the student on an observation format RL will never produce. Only
   ``--max-tool-output-chars`` is applied, mirroring the runner's ``tool_output_cap``.

3. **Reasoning is available.** ``trajectory.reasoning`` carries the teacher's chain of
   thought per turn, keyed by ``step``. The earlier bundle had none, which is why a model
   trained on it emitted a ``<think>`` block on only 3% of turns. It is re-injected into the
   owning assistant turn as ``<think>...</think>`` so the student learns to reason before
   acting; use ``--no-inject-reasoning`` to train on answers alone.

   Reasoning is matched by ``step``, not by position: a turn whose response carried no
   reasoning simply has no entry, so zipping the two lists would silently attach one turn's
   thinking to another (measured: only 105 of 400 rollouts have equal counts).

Tool JSON schemas are **not** shipped in the bundle (they live in the ``mcp-atlas-runtime``
image), so ``--tool-schemas`` supplies them from a dump of the gateway's ``/list-tools``. That
dump does double duty: it carries the schemas, and its key order -- harvested with every service
enabled -- is the gateway's own tool order, which is what makes each task's menu reproducible
offline. It is required; there is no reconstruction fallback.

The **menu** -- which tools a row offers -- is built the way RL builds it: every tool of every
service in the task's ``AQ_SIM_ENABLED_SERVERS``, in the gateway's own order. It is emphatically
not ``task.toml``'s ``enabled_tools``, which is a curated subset. The teacher ran through this
repo's ``mcp-atlas`` agent against the same gateway, so it chose from the served menu, and
56% of teacher rollouts call a tool ``enabled_tools`` omits. Building rows against the curated
list therefore produced targets calling tools the row never offered -- 45% of rows, 12% of
calls -- training the student that the menu does not constrain it. ``--max-off-menu-row-fraction``
guards the repair.

Usage::

    uv run --extra skyrl-train examples/train/mcp_atlas/prepare_glm_sft_dataset.py \\
        --trajectories-dir ~/aq_full_delivery/AQ-MCP-Atlas-1000-Full-Delivery-20260812/trajectories \\
        --tasks-dir ~/AQ-MCP-Atlas-1000-Tasks \\
        --output-dir ~/data/mcp_atlas_sft \\
        --min-coverage 0.85

`--extra skyrl-train` rather than `--extra fsdp`: the only SkyRL import here is the loss-mask
helper, which transitively needs pydantic/torch/numpy but none of fsdp's CUDA-kernel wheels
(causal-conv1d, flashinfer, flash-linear-attention). Those are fetched from GitHub releases
and throttle when several processes resolve at once. Verified byte-identical output either way.
"""

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _parse_enabled_services(dockerfile_text: str) -> List[str]:
    """The task's ``AQ_SIM_ENABLED_SERVERS``, which is what the gateway actually serves."""
    match = re.search(r'AQ_SIM_ENABLED_SERVERS="([^"]+)"', dockerfile_text)
    if not match:
        return []
    return [s.strip() for s in match.group(1).split(",") if s.strip()]


def _service_of(tool: str, services: List[str]) -> Optional[str]:
    """Owning service for a tool name, longest prefix first.

    Longest-prefix matters: 'google_calendar' and 'google_forms' would both lose to a bare
    'google' prefix, and 'ddg-search'/'open-library' carry hyphens inside the service name.
    """
    candidates = [s for s in services if tool.startswith(s + "_")]
    return max(candidates, key=len) if candidates else None


def served_tool_schemas(
    dockerfile_text: str, real_schemas: Dict[str, Dict[str, Any]], services: List[str]
) -> List[Dict[str, Any]]:
    """The tools RL serves for a task: every tool of its enabled services, gateway order.

    The runner does not filter -- it offers whatever ``/list-tools`` returns (see
    ``mcp_atlas_runner.py``), and the gateway serves the full surface of every service in
    ``AQ_SIM_ENABLED_SERVERS``. ``real_schemas`` was dumped from that same endpoint with all
    services enabled, so its key order *is* the gateway's ordering and filtering it preserves
    both the set and the order. That is also why this returns schemas rather than names: the
    menu is a subset of the dump, so there is nothing to look up afterwards and no tool can
    be missing a schema.

    Verified against ``rollout.prompt_token_ids`` from 512 live RL rollouts: exact name-set
    match 512/512 and exact order match 512/512.
    """
    enabled = set(_parse_enabled_services(dockerfile_text))
    if not enabled:
        return []
    return [s for name, s in real_schemas.items() if _service_of(name, services) in enabled]


def _tool_content(raw: Any) -> str:
    """The tool observation exactly as the runner stores and re-sends it.

    Neither flattened nor truncated. The runner sets ``content`` to ``json.dumps`` of the
    gateway's reply -- the raw MCP block list -- and feeds that straight back as history, so
    that is the observation format RL produces; reshaping it here would make SFT and RL
    disagree on every tool result. Trajectory length is governed by one thing only, the
    context budget applied after tokenization.
    """
    return raw if isinstance(raw, str) else json.dumps(raw)


def reasoning_by_turn(reasoning: List[Dict[str, Any]]) -> Dict[int, str]:
    """Map assistant-turn index -> chain of thought, keyed by the recorded ``step``.

    The runner appends one entry per turn that produced reasoning, stamped with the loop
    step, which equals the assistant-turn index. Turns whose response carried none are
    absent, so this must be a lookup by step rather than a zip: only 105 of 400 rollouts
    have as many reasoning entries as assistant turns, and pairing by position would attach
    one turn's thinking to a later turn's answer.
    """
    out: Dict[int, str] = {}
    for entry in reasoning or []:
        if not isinstance(entry, dict):
            continue
        step, text = entry.get("step"), entry.get("text")
        if isinstance(step, int) and text:
            out[step] = text
    return out


def convert_messages(
    raw_messages: List[Dict[str, Any]],
    reasoning: Optional[Dict[int, str]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Normalise one full-delivery transcript for ``apply_chat_template``.

    The bundle is already OpenAI-shaped, so this only copies it through and, when
    ``reasoning`` is given, prefixes each assistant turn with its ``<think>`` block.

    Returns the messages and the number of tool messages that answer no known call id --
    kept rather than dropped, since predicting the call is valid signal, but counted so a
    malformed bundle is visible instead of silently shrinking the dataset.
    """
    known_ids = {
        call.get("id")
        for msg in raw_messages
        if isinstance(msg, dict)
        for call in (msg.get("tool_calls") or [])
    }
    out: List[Dict[str, Any]] = []
    dangling = 0
    assistant_index = 0
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            content = msg.get("content") or ""
            if reasoning is not None:
                think = reasoning.get(assistant_index)
                if think:
                    content = f"<think>\n{think.strip()}\n</think>\n\n{content}"
            turn: Dict[str, Any] = {"role": "assistant", "content": content}
            if msg.get("tool_calls"):
                turn["tool_calls"] = msg["tool_calls"]
            out.append(turn)
            assistant_index += 1
        elif role == "tool":
            if msg.get("tool_call_id") not in known_ids:
                dangling += 1
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id"),
                    "content": _tool_content(msg.get("content")),
                }
            )
        elif role in ("user", "system"):
            out.append({"role": role, "content": msg.get("content") or ""})
        else:
            raise ValueError(f"Unexpected role {role!r} in trajectory")
    return out, dangling


def pretokenize_row(
    messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], tokenizer, chat_template: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Tokenize one conversation into ``input_ids`` + a full-sequence ``loss_mask``.

    Mirrors ``sft_trainer._tokenize_chat_all_assistants`` (the path
    ``train_on_what=all_assistant_messages`` would take) so the only difference from online
    tokenization is the chat template: leading non-assistant messages are rendered with the
    tool schemas and masked out, then every later message is encoded individually by
    ``get_response_ids_and_loss_mask_from_messages``, which masks user/tool observations.
    Returns None when the conversation has no assistant turn.
    """
    from skyrl.train.generators.utils import get_response_ids_and_loss_mask_from_messages

    tokenizer_kwargs: Dict[str, Any] = {}
    if tools:
        tokenizer_kwargs["tools"] = tools
    if chat_template:
        tokenizer_kwargs["chat_template"] = chat_template

    first_assistant = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), None)
    if first_assistant is None:
        return None

    prompt_ids = tokenizer.apply_chat_template(
        messages[:first_assistant],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        **tokenizer_kwargs,
    )
    response_ids, response_mask, _ = get_response_ids_and_loss_mask_from_messages(
        messages[first_assistant:], tokenizer, tokenizer_kwargs=tokenizer_kwargs
    )
    # The pretokenized loader wants loss_mask aligned 1:1 with input_ids and infers
    # num_actions from the first nonzero entry, so zero out the prompt span here.
    return {
        "input_ids": prompt_ids + response_ids,
        "loss_mask": [0] * len(prompt_ids) + response_mask,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trajectories-dir",
        required=True,
        help="The full-delivery bundle's `trajectories/` directory (contains rollouts/).",
    )
    parser.add_argument("--tasks-dir", required=True, help="Extracted AQ-MCP-Atlas-1000-Tasks bundle.")
    parser.add_argument("--output-dir", default="~/data/mcp_atlas_sft", help="Where to write train/val parquet.")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.85,
        help="Keep only rollouts whose judge coverage is >= this. The full-delivery bundle "
        "sets pass_threshold=0.85 (the earlier bundle used 0.75).",
    )
    parser.add_argument(
        "--max-off-menu-row-fraction",
        type=float,
        default=0.05,
        help="Fail if more than this fraction of emitted rows call a tool absent from their own "
        "tools block. With the served menu the residue is teacher typos only (~1%% of rollouts); "
        "the curated task.toml menu scored 45%%, which is the regression this guards against.",
    )
    parser.add_argument(
        "--one-per-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep exactly one rollout per task -- the highest-coverage run that clears "
        "--min-coverage (default). Without it a task whose three runs all scored well "
        "contributes three near-duplicate conversations, so easy tasks dominate the mixture: "
        "measured across pairs of perfect runs the extra rollouts are largely redundant (mean "
        "tool-set Jaccard 0.855). Use --no-one-per-task to keep every qualifying rollout.",
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
        "--inject-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefix each assistant turn with the teacher's <think> block (default). The "
        "earlier bundle shipped no reasoning, and a model trained on it produced a think "
        "block on only 3%% of turns; this bundle has it, matched by recorded step.",
    )
    parser.add_argument(
        "--tool-schemas",
        type=Path,
        default=Path("~/data/mcp_atlas_tool_schemas.json"),
        help="JSON of real tool schemas from dump_tool_schemas.py, dumped from the runtime "
        "image's /list-tools. Required: it supplies the schemas RL serves, and its key order "
        "supplies each task's menu and that menu's order.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="The only length control: drop rollouts whose tokenized conversation exceeds "
        "this many tokens. Set it to the policy's context window (and to SFTConfig.max_length, "
        "which would otherwise silently truncate a row mid-answer). Nothing else is capped -- "
        "tool observations are kept byte-for-byte as the runner re-sends them, and turn count "
        "is unbounded -- so trajectory length is governed here and nowhere else. 0 disables it.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Fraction held out as a validation split. Defaults to 0: every qualifying "
        "rollout trains. At this size (a few hundred rows after one-per-task selection) a 5%% "
        "split is ~26 rows, too few to read a loss curve from and a measurable slice of the "
        "data to give up. Pass e.g. 0.05 to reinstate one.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-30B-A3B",
        help="Tokenizer for pre-tokenization and sequence-length stats.",
    )
    parser.add_argument(
        "--pretokenize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit `input_ids` + full-sequence `loss_mask` for SFTConfig.pretokenized_dataset_paths "
        "(default). This is the only way to control the chat template, since SFTConfig has no "
        "chat_template field: the stock Qwen3 template injects an empty `<think></think>` into every "
        "assistant turn *inside the trained span*, teaching the model to skip reasoning. "
        "Use --no-pretokenize to emit `messages`/`tools` for online tokenization instead.",
    )
    parser.add_argument(
        "--chat-template",
        default=str(
            Path(__file__).resolve().parents[3]
            / "skyrl"
            / "train"
            / "utils"
            / "templates"
            / "qwen3_acc_thinking.jinja2"
        ),
        help="Jinja chat template used for pre-tokenization. The default never injects a think block "
        "and never strips reasoning from earlier turns, so it matches what RL should serve via "
        "generator.inference_engine.engine_init_kwargs.chat_template. Pass '' for the tokenizer default.",
    )
    args = parser.parse_args()

    traj_dir = Path(args.trajectories_dir).expanduser().resolve()
    tasks_dir = Path(args.tasks_dir).expanduser().resolve()
    rollout_dir = traj_dir / "rollouts"
    if not rollout_dir.is_dir():
        raise SystemExit(
            f"{rollout_dir} not found. Point --trajectories-dir at the full-delivery bundle's "
            "`trajectories/` directory. (The earlier bundle's index.csv layout is not supported "
            "by this script; its transcripts were Anthropic-shaped and carried no reasoning.)"
        )
    manifest_path = traj_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        print(
            f"Batch {manifest.get('batch_id')}: model={manifest.get('model')} "
            f"judge={manifest.get('judge_model')} pass_threshold={manifest.get('pass_threshold')} "
            f"mean_coverage={manifest.get('mean_coverage')}"
        )

    # 1. Read every rollout and select on the judge's coverage.
    rollouts = sorted(rollout_dir.glob("*.json"))
    if not rollouts:
        raise SystemExit(f"No rollout files under {rollout_dir}")
    selected, unusable = [], collections.Counter()
    for path in rollouts:
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            unusable["unparseable"] += 1
            continue
        traj = payload.get("trajectory") or {}
        if not traj.get("messages"):
            unusable["no messages"] += 1
            continue
        # A rollout that errored mid-flight still has the turns it completed, but its final
        # answer is missing or partial and the judge scored that, not the trajectory.
        if traj.get("error"):
            unusable["errored"] += 1
            continue
        coverage = (payload.get("reward") or {}).get("coverage")
        if coverage is None:
            unusable["ungraded"] += 1
            continue
        if coverage < args.min_coverage:
            unusable[f"coverage < {args.min_coverage}"] += 1
            continue
        n_calls = len(traj.get("tool_calls") or [])
        selected.append(
            (payload.get("task_id"), int(payload.get("run_index") or 0), float(coverage), path, n_calls)
        )

    if args.one_per_task:
        # Rank by coverage, then break ties with --prefer. Ties at the top are common: the
        # bundle reports mean coverage near 0.7 with a large mass at 1.0.
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
        raise SystemExit(f"No rollouts passed the filters. Rejected: {dict(unusable)}")
    if args.one_per_task:
        # Asserted rather than assumed: this is the guarantee the mixture depends on, and a
        # duplicate would quietly double one task's weight.
        per_task = collections.Counter(task for task, _, _, _, _ in selected)
        duplicated = {t: n for t, n in per_task.items() if n > 1}
        assert not duplicated, f"one-per-task violated for {duplicated}"
    print(
        f"Selected {len(selected)} of {len(rollouts)} rollouts "
        f"(coverage>={args.min_coverage}{', one per task' if args.one_per_task else ''}) "
        f"covering {len({t for t, _, _, _, _ in selected})} tasks"
    )
    if unusable:
        print(f"  rejected: {dict(unusable.most_common())}")

    # The dump is required, not optional: it supplies both the schemas and -- through its key
    # order -- the menu and the menu's order. There is nothing to fall back to.
    schema_path = Path(str(args.tool_schemas)).expanduser()
    if not schema_path.is_file():
        raise SystemExit(
            f"{schema_path} not found. Generate it with:\n"
            f"  uv run examples/train/mcp_atlas/dump_tool_schemas.py "
            f"--tasks-dir <tasks> --output {schema_path}"
        )
    payload = json.loads(schema_path.read_text())
    real_schemas: Dict[str, Dict[str, Any]] = payload.get("schemas") or {}
    dump_services: List[str] = payload.get("services") or []
    dump_image = payload.get("image")
    if not real_schemas or not dump_services:
        raise SystemExit(f"{schema_path} has no schemas/services; regenerate it")
    print(f"Loaded {len(real_schemas)} real tool schemas from {schema_path} (image {dump_image})")

    # 2. Build each task's tool menu the way RL builds it: every tool of every service in the
    #    task's AQ_SIM_ENABLED_SERVERS, in the gateway's own order.
    #
    #    NOT task.toml's enabled_tools. That is a curated subset, and using it made the dataset
    #    internally inconsistent, because the teacher ran against the same gateway and so chose
    #    from the wider menu: 1667/2982 teacher rollouts (56%) call a tool absent from
    #    enabled_tools, which left 45% of emitted rows demonstrating a call to a tool their own
    #    <tools> block never offered. Measured menu sizes: enabled_tools mean 15.6, served mean
    #    32.4 (max 150), with the served menu strictly larger for 959/1000 tasks.
    #    The served path yields schemas directly. Only the degraded path below needs a list of
    #    names, because there a name may have no served schema and must be reconstructed.
    served_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for task_id in {t for t, _, _, _, _ in selected}:
        dockerfile = tasks_dir / "tasks" / task_id / "environment" / "Dockerfile"
        # No fallback. A row whose menu is not the served menu is the defect this replaced, and
        # falling back would reintroduce it silently for the affected tasks. Verified reachable
        # for the whole set: across all 1000 tasks there are no missing Dockerfiles, no
        # unparsable AQ_SIM_ENABLED_SERVERS, no services absent from the dump, and no empty
        # menus -- so this raising is a real invariant, not an unhandled case.
        if not dockerfile.is_file():
            raise SystemExit(f"{task_id}: no environment/Dockerfile, cannot determine served tools")
        schemas = served_tool_schemas(dockerfile.read_text(), real_schemas, dump_services)
        if not schemas:
            raise SystemExit(
                f"{task_id}: empty served menu. Either AQ_SIM_ENABLED_SERVERS is missing from its "
                f"Dockerfile, or its services are absent from {args.tool_schemas} (dumped from "
                f"image {dump_image})."
            )
        served_by_task[task_id] = schemas
    sizes = sorted(len(v) for v in served_by_task.values())
    print(
        f"Tool menu per task: mean {sum(sizes)/len(sizes):.1f}, median {sizes[len(sizes)//2]}, "
        f"max {sizes[-1]} (served menu, matching RL)"
    )

    # 3. Pass one: normalise transcripts.
    converted: List[tuple] = []
    total_dangling = 0
    skipped = 0
    with_reasoning = 0
    for task_id, run, coverage, path, _ in selected:
        payload = json.loads(path.read_text())
        traj = payload["trajectory"]
        reasoning = reasoning_by_turn(traj.get("reasoning") or []) if args.inject_reasoning else None
        try:
            messages, dangling = convert_messages(traj["messages"], reasoning)
        except ValueError as exc:
            print(f"WARNING: skipping {task_id} run{run}: {exc}")
            skipped += 1
            continue
        total_dangling += dangling
        if not any(m["role"] == "assistant" for m in messages):
            skipped += 1
            continue
        if reasoning:
            with_reasoning += 1
        converted.append((task_id, run, coverage, messages))

    print(f"Converted {len(converted)} rollouts (skipped {skipped}); {total_dangling} tool messages answered no known call")
    if args.inject_reasoning:
        print(f"Injected teacher reasoning into {with_reasoning}/{len(converted)} rollouts")

    # 4. Pass two: attach per-task tool schemas and emit rows.
    out_rows = []
    off_menu_rows = 0
    off_menu_calls = 0
    total_calls = 0
    off_menu_names: collections.Counter = collections.Counter()
    for task_id, run, coverage, messages in converted:
        tools = served_by_task[task_id]
        # A target that calls a tool the row never offered teaches the student that the menu
        # does not constrain it. This is the defect the served menu fixes, so it is measured
        # on the emitted rows rather than assumed away.
        menu = {t["function"]["name"] for t in tools}
        called = [
            (call.get("function") or {}).get("name")
            for message in messages
            if message.get("role") == "assistant"
            for call in (message.get("tool_calls") or [])
        ]
        called = [c for c in called if c]
        off = [c for c in called if c not in menu]
        total_calls += len(called)
        off_menu_calls += len(off)
        if off:
            off_menu_rows += 1
            off_menu_names.update(off)
        out_rows.append(
            {
                "messages": messages,
                "tools": json.dumps(tools),
                "task_id": task_id,
                "run_index": run,
                "coverage": coverage,
            }
        )

    pct_rows = off_menu_rows / len(out_rows) if out_rows else 0.0
    print(
        f"Off-menu calls (target calls a tool absent from its own tools block): "
        f"{off_menu_rows}/{len(out_rows)} rows ({pct_rows:.0%}), "
        f"{off_menu_calls}/{total_calls} calls"
    )
    if off_menu_names:
        # With the served menu the only residue should be teacher typos -- names no service
        # ever exposed (e.g. 'calculator_calcalculate', 'google-calendar_list_events' for the
        # real 'google_calendar_list_events'). Anything beyond a trickle means the menu is
        # being built wrong again, so it is a hard failure rather than a warning.
        print(f"  off-menu names: {off_menu_names.most_common(8)}")
        assert pct_rows <= args.max_off_menu_row_fraction, (
            f"{pct_rows:.0%} of rows call an off-menu tool, above "
            f"--max-off-menu-row-fraction {args.max_off_menu_row_fraction:.0%}. The tool menu "
            f"almost certainly does not match what RL serves. Worst offenders: "
            f"{off_menu_names.most_common(5)}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(out_rows)

    tokenizer = None
    if args.pretokenize:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        chat_template = None
        if args.chat_template:
            chat_template = Path(args.chat_template).expanduser().read_text()
            print(f"Pre-tokenizing with chat template {args.chat_template}")
        else:
            print(f"Pre-tokenizing with the {args.tokenizer} default chat template")

        tokenized_rows, no_trainable, over_length = [], 0, []
        for row in out_rows:
            tokenized = pretokenize_row(row["messages"], json.loads(row["tools"]) or None, tokenizer, chat_template)
            if tokenized is None or sum(tokenized["loss_mask"]) == 0:
                no_trainable += 1
                continue
            n = len(tokenized["input_ids"])
            # The single length control. Dropped rather than truncated: cutting a conversation
            # mid-answer trains the model to stop early, and cutting the prompt end detaches
            # the observations an answer was derived from.
            if args.max_length and n > args.max_length:
                over_length.append(n)
                continue
            tokenized_rows.append({**tokenized, "task_id": row["task_id"], "coverage": row["coverage"]})
        if no_trainable:
            print(f"Dropped {no_trainable} rows with no trainable assistant tokens")
        if over_length:
            over_length.sort()
            print(
                f"Dropped {len(over_length)} rows over --max-length={args.max_length} "
                f"(their lengths: median={over_length[len(over_length)//2]}, max={over_length[-1]})"
            )
        out_rows = tokenized_rows

        think_ids = {tid for tag in ("<think>", "</think>") for tid in tokenizer.encode(tag, add_special_tokens=False)}
        with_think = sum(1 for r in out_rows if think_ids & set(r["input_ids"]))
        expectation = (
            "expected to be high: teacher reasoning is being trained on"
            if args.inject_reasoning
            else "expected 0: the template must not inject empty blocks"
        )
        print(f"Rows containing a <think> tag: {with_think}/{len(out_rows)} ({expectation})")

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

    # Report sequence lengths so max_length can be chosen rather than guessed.
    if args.pretokenize:
        lengths = sorted(len(r["input_ids"]) for r in out_rows)
        trained = [sum(r["loss_mask"]) for r in out_rows]
    else:
        try:
            from transformers import AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as exc:  # noqa: BLE001
            print(f"(skipped token-length report: {exc})")
            return
        sample = out_rows if len(out_rows) <= 400 else rng.sample(out_rows, 400)
        lengths, trained = [], []
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

    def pct(p):
        return lengths[min(len(lengths) - 1, int(len(lengths) * p))]

    print(
        f"Sequence lengths over {len(lengths)} rows ({args.tokenizer}): "
        f"mean={sum(lengths) // len(lengths)} median={pct(0.5)} p90={pct(0.9)} p99={pct(0.99)} max={lengths[-1]}"
    )
    if trained:
        print(
            f"Trained (loss_mask=1) tokens per row: mean={sum(trained) // len(trained)} "
            f"max={max(trained)}  -> {100 * sum(trained) / sum(lengths):.1f}% of all tokens"
        )
    for limit in (8192, 16384, 32768):
        print(f"  fit within max_length={limit}: {sum(1 for n in lengths if n <= limit)}/{len(lengths)}")


if __name__ == "__main__":
    main()
