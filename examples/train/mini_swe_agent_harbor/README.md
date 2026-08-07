## Guide: Mini-SWE-Agent + Harbor tasks + SkyRL

Trains a coding agent with [Mini-SWE-Agent](https://github.com/SWE-agent/mini-swe-agent) on
[Harbor](https://github.com/harbor-framework/harbor)-format tasks. Sibling of
`examples/train/mini_swe_agent`, which does the same for SWE-Bench/SWE-Gym.

```bash
# 1. Authenticate to the registry holding the task images (see "Registry auth" below)
REGISTRY_AUTH_FILE=$HOME/.config/containers/auth.json \
  podman login -u _json_key --password-stdin us-docker.pkg.dev < ~/<sa-key>.json

# 2. Write parquet (no image builds needed when tasks declare a prebuilt image)
uv run --isolated examples/train/mini_swe_agent_harbor/preprocess_harbor.py \
    --tasks_dir ~/swebench-pro-tinker/ez_500_verified \
    --output_dir ~/data/harbor_ez500

# 3. Train
bash examples/train/mini_swe_agent_harbor/run_mini_swe_harbor_8B.sh
```

## Version pin (read this first)

This example targets **mini-swe-agent v1**, pinned in `requirements-miniswe.txt` to the
scaleapi fork at `d74716a` (which reports `1.15.0`) and applied by the run script:

```bash
uv run --isolated --extra fsdp --extra miniswe \
  --with-requirements examples/train/mini_swe_agent_harbor/requirements-miniswe.txt ...
```

`--with-requirements` overrides `uv.lock` for that invocation only; nothing repo-wide
changes. The pin is **not optional**. `pyproject.toml` declares `mini-swe-agent>=1.12.0`
with no upper bound, and `uv.lock` currently resolves **2.4.2** — across a breaking 1.x→2.x
rewrite of every API this example touches. All four failures are silent:

| | v1 (pinned) | v2 (`uv.lock`) | if you run v2 anyway |
|---|---|---|---|
| `Environment.execute` | `(command: str)` | `(action: dict)` | `AttributeError: 'str' object has no attribute 'get'` from `env_startup_command`, before the agent exists → every trajectory empty → `generate()` fails the step with "Found no valid responses" |
| `Agent.run()` | `-> tuple[str, str]` | `-> dict` | unpacks the dict's *keys*; submission becomes the literal `"submission"` |
| turn hook | `get_observation` | `execute_actions` | override never called; turn reminder vanishes |
| model default | text-based, parses ```` ```bash ```` | tool-calling `tools=[BASH_TOOL]` | `FormatError` every turn |

Note this cuts the other way too: `examples/train/mini_swe_agent` is written against v1 and
has no pin, so **it is broken in-tree** — the lock crossed the boundary on 2026-02-18
(`1.11.1` at the original commit → `2.2.0` after the monorepo reorg → `2.4.2` today).

One upside of v1: `AgentConfig` and `LitellmModelConfig` are plain dataclasses, so an
unknown key in `agent:` or `model:` raises `TypeError` instead of being silently dropped.

## Registry auth

Harbor task images typically live in a private registry. Use a **service-account JSON key**,
not `gcloud auth print-access-token`: access tokens expire after ~1 hour, images are pulled
lazily throughout a run (one per distinct task), and a token-based login dies partway through
with an opaque pull failure. The `_json_key` username takes the whole key file as the
password and does not expire.

podman writes credentials to `$XDG_RUNTIME_DIR/containers/auth.json` by default, which is
**tmpfs** and lost on reboot. The run script exports
`REGISTRY_AUTH_FILE=$HOME/.config/containers/auth.json`, which the Ray workers inherit, so
log in against that path. On multi-node, that file must exist on every worker.

## What is reused

`harbor_generator.py` subclasses `MiniSweAgentGenerator`. Inherited unchanged: `generate()`
batching, `__init__`, and `get_sb_environment` from `mini_swe_agent/mini_swe_utils.py`.
`DefaultAgentWithReminder` is imported and subclassed as `FormatErrorCountingAgent` (which
adds only a `format_errors` counter).

`minisweagent_agent_loop` is overridden. The copy exists because `init_and_run` is resolved
in the *defining module's* globals: an inherited method would call the SWE-Bench Ray task,
not the Harbor one. It deviates from the original in two places — a `last_idx >= 0` guard
that stops an all-`user` response from `IndexError`ing, and the grading-failure drop below.

`get_sb_environment` works as-is because `get_docker_image_name` short-circuits on
`instance["image_name"]`, which `preprocess_harbor.py` always fills in, so the SWE-Bench
registry-naming branch never runs and `data_source` is inert.

## What differs, and why

| | `mini_swe_agent` (SWE-Bench) | `mini_swe_agent_harbor` |
|---|---|---|
| **Image** | pulled: registry tag derived from `instance_id` + `data_source` | `task.toml`'s `[environment].docker_image` when present, else built from the task's `environment/Dockerfile` |
| **Working directory** | fixed `/testbed` | per task, from the image's own `WORKDIR` |
| **Grading target** | a **fresh** container; `git apply` the agent's `git diff --cached`, then run `instance["eval_script"]` | the **live** container the agent mutated; upload `tests/`, run `tests/test.sh` |
| **Reward** | `int(resolved)` from the eval script's exit code | `float` from `/logs/verifier/reward.txt` (or `reward.json`) |
| **Submission** | `echo MARKER && git add -A && git diff --cached` | `echo MARKER` — nothing to transport |
| **Unverdicted episode** | scored `0` | trajectory dropped |
| **Container cleanup** | left to GC | explicit, per trajectory |

The grading difference is the substantive one. SWE-Bench transports state as a patch, so it
can grade in a clean container. Harbor has no patch — the mutated filesystem *is* the
submission — so the environment must reach the verifier. That is why `init_and_run` is
reimplemented rather than reused: it is the only place holding the live `env`, and Ray
re-imports the defining module in the worker, so monkeypatching the original from the driver
would not take effect.

## Reward

`evaluate_harbor_task` returns `grading_failed` alongside the reward, and the generator
**drops** the trajectory when it is set rather than training on a `0`. A dead container, a
failed test upload, a timed-out verifier, or a `test.sh` that exits without writing a reward
file are all indistinguishable from "tests failed" in the reward value alone; training on
them teaches the policy that a possibly-correct solution was wrong. The flag starts `True`
and is cleared only once a reward file has actually been read, so a new error path cannot
silently become a trainable zero.

Malformed turns (not exactly one ```` ```bash ```` block) cost
`generator.harbor_format_error_penalty` each, capped in total at
`harbor_format_error_penalty_cap`, so a format spiral costs reward rather than only turns.
The cap keeps shaping from outweighing the task signal. The penalty is applied only to
trajectories that will actually be trained on.

## Timeouts

`[verifier] timeout_sec` from each `task.toml` wins over
`generator.harbor_verifier_timeout` (which is only the fallback for tasks declaring none).
A single global value badly under-serves a typical task set — in `ez_500_verified`, 301 of
437 tasks ask for more than 600s — and a verifier killed early scores a slow-but-correct
solution `0`.

`environment.container_timeout` is a hard ceiling on agent time **plus** grading time (PID 1
in the container is `sleep <container_timeout>`). Raise it if you raise `step_limit`.

## Config notes

- `environment.cwd` is only a **fallback**, kept at `/` because podman refuses to start a
  container whose workdir does not exist. The real value comes per task from
  `instance["workdir"]`, recorded by `preprocess_harbor.py` from the Dockerfile's `WORKDIR`.
- `environment.executable` is `podman`, as in the SWE-Bench example.
- `run.env_startup_command` creates `/logs/verifier`; some `test.sh` scripts assume it
  exists. This works only on v1 — see the version-pin table.
- `forward_env: []` — do not leak host env (API keys) into a container running
  model-authored code.
- The `agent:` prompts are mini-swe-agent's own (`config/mini.yaml`), used verbatim.
  `instance_template` interpolates `{{system}} {{release}} {{version}} {{machine}}`, which
  `LocalEnvironment` supplies from `platform.uname()` but `DockerEnvironment` does not — and
  Jinja runs with `StrictUndefined`. `init_and_run` reads `uname` from inside the container
  and passes it to `agent.run()`; without that, every episode dies on
  `UndefinedError: 'system' is undefined`.

## Known limitation

`response_ids` are reconstructed by re-encoding `agent.messages`, and `rollout_logprobs` is
`None` — inherited from the SWE-Bench generator. Those are **not** the exact tokens the
policy emitted, and there is no importance-sampling correction. If that matters, see
`mini-swe-tinker/` in this workspace, which plugs a model directly into mini-swe-agent and
records ids and logprobs at the sampler.

## Multi-node

`instance["tests_dir"]` is read on whichever worker runs the trajectory, so `--tasks_dir`
must be on shared storage. Task images must be pullable from every node, and
`REGISTRY_AUTH_FILE` must resolve on each. For task sets without a declared
`docker_image`, either run `preprocess_harbor.py` on each node or push to a registry and
pass `--image_prefix` plus `--skip_build`.
