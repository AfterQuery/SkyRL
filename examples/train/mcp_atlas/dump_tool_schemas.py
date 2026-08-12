"""Dump the runtime image's real tool schemas so SFT can present the prompt RL will serve.

The task bundles ship no tool schemas -- they live in the ``mcp-atlas-runtime`` image, and the
agent discovers them at run time from the container-local gateway's ``/list-tools``. That left
the SFT prep script reconstructing them from teacher usage and fabricating descriptions
(``"MCP tool wikipedia_search_wikipedia."``), which is a large, silent train/serve mismatch:
the schema block is roughly a third of the prompt, and the GLM teacher itself saw the real
descriptions, so a dataset built on stubs misrepresents the very context the demonstrations
were produced under.

Schemas are per **service**, not per task, which makes this cheap. The 1000 tasks use 803
distinct ``AQ_SIM_ENABLED_SERVERS`` combinations but only 33 distinct services, and a single
container with all of them enabled serves every tool -- one ~20s boot instead of 803.

Usage::

    uv run examples/train/mcp_atlas/dump_tool_schemas.py \\
        --tasks-dir ~/aq_tasks_v2/AQ-MCP-Atlas-1000-Tasks \\
        --output ~/data/mcp_atlas_tool_schemas.json

Then pass ``--tool-schemas ~/data/mcp_atlas_tool_schemas.json`` to
``prepare_glm_sft_dataset.py``.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Set

GATEWAY_BOOT = "/workspace/runtime_env/docker/run_agent_environment.sh"
CONTAINER = "mcp-atlas-schema-dump"


def enabled_services(tasks_dir: Path) -> Dict[str, Set[str]]:
    """Map task id -> its enabled services, read from each task's Dockerfile."""
    out: Dict[str, Set[str]] = {}
    for path in sorted(glob.glob(str(tasks_dir / "tasks" / "*" / "environment" / "Dockerfile"))):
        task = Path(path).parents[1].name
        match = re.search(r'AQ_SIM_ENABLED_SERVERS="([^"]+)"', Path(path).read_text())
        if match:
            out[task] = {s.strip() for s in match.group(1).split(",") if s.strip()}
    return out


def _docker(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def harvest(image: str, services: Set[str], boot_timeout: int) -> List[dict]:
    """Boot one container with every service enabled and return the gateway's tool list."""
    _docker("rm", "-f", CONTAINER)
    run = _docker(
        "run", "-d", "--name", CONTAINER,
        "-e", f"AQ_SIM_ENABLED_SERVERS={','.join(sorted(services))}",
        image, "tail", "-f", "/dev/null",
    )
    if run.returncode != 0:
        raise SystemExit(f"could not start {image}: {run.stderr.strip()[:300]}")
    try:
        # Harbor replaces the image CMD with a keepalive, so nothing starts the gateway but us.
        _docker("exec", "-u", "root", CONTAINER, "bash", "-c",
                f"nohup bash {GATEWAY_BOOT} > /tmp/gw.log 2>&1 &")
        deadline = time.monotonic() + boot_timeout
        while True:
            probe = _docker("exec", CONTAINER, "curl", "-sf", "http://127.0.0.1:1984/health")
            if probe.returncode == 0:
                break
            if time.monotonic() >= deadline:
                log = _docker("exec", CONTAINER, "tail", "-5", "/tmp/gw.log").stdout
                raise SystemExit(f"gateway did not come up within {boot_timeout}s. Log:\n{log}")
            time.sleep(2)
        health = json.loads(_docker("exec", CONTAINER, "curl", "-s",
                                    "http://127.0.0.1:1984/health").stdout or "{}")
        print(f"  gateway up: {len(health.get('configured_servers') or [])} services configured")
        listed = _docker("exec", CONTAINER, "curl", "-s", "-XPOST",
                         "http://127.0.0.1:1984/list-tools",
                         "-H", "Content-Type: application/json", "-d", "{}")
        tools = json.loads(listed.stdout)
        # /list-tools returns a bare list on this runtime and {"tools": [...]} elsewhere.
        return tools.get("tools", tools) if isinstance(tools, dict) else tools
    finally:
        _docker("rm", "-f", CONTAINER)


def to_openai(tool: dict) -> dict:
    """One gateway tool as an OpenAI function schema, exactly as the runner builds it."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            # Same key precedence the runner uses: inputSchema on this runtime,
            # input_schema upstream.
            "parameters": tool.get("input_schema")
            or tool.get("inputSchema")
            or {"type": "object", "properties": {}},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", default="mcp-atlas-runtime:delivery7-20260625")
    parser.add_argument("--boot-timeout", type=int, default=300)
    parser.add_argument(
        "--emit-target-tools",
        type=Path,
        default=None,
        help="Also write tests/target_tools.json into each task under this directory, copied "
        "from its task.toml. The verifier needs it to score tool recall, and only tests/ is "
        "uploaded into the container -- task.toml is not. Point this at the patched copy of "
        "the task set, alongside the grade.py swap.",
    )
    args = parser.parse_args()

    per_task = enabled_services(args.tasks_dir.expanduser())
    if not per_task:
        raise SystemExit(f"no task Dockerfiles under {args.tasks_dir}")
    services = set().union(*per_task.values())
    combos = {frozenset(v) for v in per_task.values()}
    print(
        f"{len(per_task)} tasks, {len(combos)} distinct service combinations, "
        f"{len(services)} distinct services -> 1 container boot"
    )

    tools = harvest(args.image, services, args.boot_timeout)
    by_name = {t["name"]: to_openai(t) for t in tools if t.get("name")}
    print(f"  harvested {len(by_name)} tool schemas")

    # Report tools a task expects but the gateway never serves. A missing schema would send
    # the student a stub for a tool RL presents in full, which is the mismatch this fixes.
    declared: Set[str] = set()
    for path in sorted(glob.glob(str(args.tasks_dir.expanduser() / "tasks" / "*" / "task.toml"))):
        text = Path(path).read_text()
        start = text.find("enabled_tools = [")
        if start == -1:
            continue
        body = text[start + len("enabled_tools = [") : text.find("]", start)]
        declared |= {p.strip().strip('",').strip('"') for p in body.split("\n") if p.strip(" ,")}
    missing = sorted(t for t in declared if t and t not in by_name)
    print(f"  tools declared by tasks: {len(declared)}; without a served schema: {len(missing)}")
    if missing:
        print(f"    e.g. {missing[:5]}")

    if args.emit_target_tools:
        root = args.emit_target_tools.expanduser()
        written = skipped = 0
        for toml_path in sorted(glob.glob(str(root / "*" / "task.toml"))):
            text = Path(toml_path).read_text()
            start = text.find("target_tools = [")
            if start == -1:
                skipped += 1
                continue
            body = text[start + len("target_tools = [") : text.find("]", start)]
            tools = [p.strip().strip('",').strip('"') for p in body.split("\n") if p.strip(" ,")]
            tests = Path(toml_path).parent / "tests"
            if not tests.is_dir():
                skipped += 1
                continue
            (tests / "target_tools.json").write_text(
                json.dumps({"target_tools": [t for t in tools if t]}, indent=1) + "\n"
            )
            written += 1
        print(f"  wrote tests/target_tools.json for {written} tasks ({skipped} skipped)")

    payload = {
        "image": args.image,
        "n_services": len(services),
        "services": sorted(services),
        "schemas": by_name,
        "declared_without_schema": missing,
    }
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(payload, indent=1) + "\n")
    print(f"Wrote {len(by_name)} schemas to {args.output}")


if __name__ == "__main__":
    main()
