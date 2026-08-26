# Toolathlon Tasks

2,000 harbor tasks on a shared JSON-world runtime. Each task ships a world file,
an instruction, a reference solution, and a deterministic verifier.

    tasks/      2,000 harbor task directories
    runtime/    source and reference Dockerfile for the shared base image

Every task is pinned to **harbor schema 1.4**.

## Build

Build the shared base image once; every task image layers on it.

    cd runtime
    tar xzf toolathlon-json-runtime-src.tar.gz
    docker build -t toolathlon-json-runtime:v1 toolathlon-json-runtime

The bundled `Dockerfile` already applies the tool policy below; `Dockerfile.reference`
is an identical copy kept for reference.

## Run

    harbor run --path tasks -i <task-id> --agent claude-code --model <model> \
      --mcp-config tasks/<task-id>/environment/mcp.json

On a machine without Docker, stage the shared runtime and let Daytona build and
cache the image remotely:

    DAYTONA_API_KEY=... HARBOR_BIN=../.venv/bin/harbor \
      ./run-on-daytona.sh <task-id> --agent oracle

The helper changes only a temporary task copy. Extra arguments after the task ID
are forwarded to `harbor run`; with no extra arguments it defaults to the Oracle
agent for an end-to-end smoke test.

No additional flags are required. To confirm a task end to end without a model:

    harbor run --path tasks -i <task-id> --agent oracle   # expect reward 1.0
    harbor run --path tasks -i <task-id> --agent nop      # expect reward 0.0

## Verification

Every task in `tasks/` clears both gates:

| Gate | Result |
|---|---|
| Reference solution replays to reward | **1.00 on all 2,000** |
| No-op run scores | **0.00 on all 2,000** |
| Harbor + docker E2E (58-task stratified sample) | oracle 1.0 / nop 0.0 on all 58 |
| Tool surface in-container | only the task's namespaced MCP tools; no built-in leak |

Grading runs against the world flushed to `T3_WORLD_DUMP` (`/logs/world_after.json`),
not the container filesystem. Worlds are in-process and file-backed with the clock
frozen at `meta.frozen_now`, so runs are deterministic.

Every task carries at least 4 *effective* checks — assertions a no-op actually
fails. Nominal check counts run higher (median 13) because most tasks add
`preserve_*` guards that only bite on collateral damage; those are excluded from
the floor. A stratified 303-task LLM audit additionally verified that a correct
agent can earn reward 1: every blocking defect it alleged was handed to an
independent skeptic, and 40% did not survive that refutation.

Tasks that failed a gate are not here — they are bucketed, with per-task
reasons, in `../Rejected Tasks - August 24`.

## Tool policy

The tasks are difficulty-calibrated against the task's MCP tools only. The base
image enforces this through Claude Code managed settings at
`/etc/claude-code/managed-settings.json`, which deny the agent's built-in tools.
A default run therefore presents the MCP tools and nothing else.

Harbor's task schema cannot express a tool constraint — the `[agent]` block
carries only `timeout_sec`, `user`, and network policy — so the policy is baked
into the image rather than supplied on the command line.

Two operational notes:

- The deny list must remain exhaustive. Revisit it when upgrading Claude Code, as
  a newly added built-in would widen the tool surface.
- Agents that do not read Claude Code managed settings are not covered by this
  mechanism and require an equivalent control.

Results produced with additional tools available are not comparable to the
published difficulty band.

## Network policy

All tasks run with `network_mode = "public"`. In-container agents install their
own tooling and call their model API during setup; under `no-network` that setup
fails before the agent starts. Docker cannot change network policy after
container start, so the policy cannot be narrowed to the agent phase alone.

The reachable surface remains limited: no simulator uses the network, and the
managed settings deny every network-capable tool. If the model is driven from
outside the container, egress is unnecessary and `network_mode` may be set to
`"no-network"`.

## Instructions

Each `instruction.md` opens with a fixed preamble identifying the workspace and
directing the agent to the MCP tools, followed by `---` and the task text. It is
delivered as the user message; harbor's task format has no system-prompt slot.
A system-role message can be added at run time with
`--agent-kwarg append_system_prompt=…`.

`/workspace` refers to the world's workspace root, which the MCP tools address
and the verifier grades. The container shell's working directory,
`/agent_workspace`, is not read by the checks.

## Provenance

`MANIFEST.json` carries, per task: the MCP servers it uses, gold call count,
scored check count, verified gold/nop rewards, which source delivery it came
from, and any pass@5 difficulty-band evidence (nemotron-3-nano and, where
measured, a five-model panel).

## Residual audit signal

Of the 2,000 shipped: **1,442 audited `good`, 549 `borderline`, 9 `bad`.**

The 9 `bad` verdicts are all tasks with **measured in-band difficulty** (a real
model scored pass@5 in [1,4] on them). Every defect alleged against them is a
*solvability* claim — unreachable information, unstated format, over-strict
literal, or ambiguity. Measured in-band evidence directly refutes that class: a
model demonstrably produced a passing answer, so a correct agent can earn reward
1. Empirical evidence outranks an audit opinion, and they are retained.

Tasks flagged for `ungraded-core-work` or `grades-only-shape` were NOT retained
on that basis. Those defects say the task is solvable but the checks grade the
wrong thing, which pass@5 cannot refute — all four such tasks were removed and
replaced with the highest-ranked previously-trimmed tasks.

`borderline` means a real but minor issue that leaves the task usable; every
blocking defect confirmed by both model families was cut.
