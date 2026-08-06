## Guide: Mini-SWE-Agent + Harbor tasks + SkyRL

Trains a coding agent with [Mini-SWE-Agent](https://github.com/SWE-agent/mini-swe-agent) on
[Harbor](https://github.com/harbor-framework/harbor)-format tasks. Sibling of
`examples/train/mini_swe_agent`, which does the same for SWE-Bench/SWE-Gym.

```bash
# 1. Get Harbor tasks (any tree with environment/Dockerfile + tests/test.sh + instruction.md)
uvx harbor datasets download terminal-bench@2.0 -o ~/data/harbor_tasks/terminal-bench-2.0

# 2. Build task images and write parquet
uv run --isolated examples/train/mini_swe_agent_harbor/preprocess_harbor.py \
    --tasks_dir ~/data/harbor_tasks/CodeContests \
    --output_dir ~/data/harbor_codecontests

# 3. Train
bash examples/train/mini_swe_agent_harbor/run_mini_swe_harbor_8B.sh
```

## What is reused

`harbor_generator.py` subclasses `MiniSweAgentGenerator`. Inherited unchanged: `generate()`
batching, `__init__`, and `get_sb_environment` from `mini_swe_agent/mini_swe_utils.py`.

`minisweagent_agent_loop` is overridden, but its body is a **verbatim copy** of the
original apart from one guarded line (see below). The copy exists only because
`init_and_run` is resolved in the *defining module's* globals: an inherited method would
call the SWE-Bench Ray task, not the Harbor one. Nothing else about the method differs —
`init_and_run` even keeps the unused `data_source` parameter so the call site matches
byte-for-byte and the two stay easy to diff.

The single deviation:

```python
# theirs — walks negative and IndexErrors if every response message is `user`
while response_messages[last_idx]["role"] == "user":
# ours
while last_idx >= 0 and response_messages[last_idx]["role"] == "user":
```

`get_sb_environment` works as-is because `get_docker_image_name` short-circuits on
`instance["image_name"]`. `preprocess_harbor.py` pre-builds each Harbor image and writes its
tag there, so the SWE-Bench registry-naming branch never runs and `data_source` is inert.

## What differs, and why

| | `mini_swe_agent` (SWE-Bench) | `mini_swe_agent_harbor` |
|---|---|---|
| **Image** | pulled: registry tag derived from `instance_id` + `data_source` | **built** from the task's `environment/Dockerfile`, content-addressed, ahead of training |
| **Grading target** | a **fresh** container; `git apply` the agent's `git diff --cached`, then run `instance["eval_script"]` | the **live** container the agent mutated; upload `tests/`, run `tests/test.sh` |
| **Reward** | `int(resolved)` from the eval script's exit code | `float` from `/logs/verifier/reward.txt` (or `reward.json`) |
| **Submission** | `echo MARKER && git add -A && git diff --cached` | `echo MARKER` — nothing to transport |
| **Container cleanup** | left to GC | explicit, per trajectory |

The grading difference is the substantive one. SWE-Bench transports state as a patch, so it
can grade in a clean container. Harbor has no patch — the mutated filesystem *is* the
submission — so the environment must reach the verifier. That is why `init_and_run` is
reimplemented rather than reused: it is the only place holding the live `env`, and Ray
re-imports the defining module in the worker, so monkeypatching the original from the driver
would not take effect.

## Three v2 incompatibilities fixed here

`uv.lock` resolves `mini-swe-agent==2.4.2`, which is the **v2** API, but the SWE-Bench
example is written against v1. All three failures are silent rather than loud, so they are
easy to miss. They apply to `mini_swe_agent` too:

1. **`agent.run()` returns a dict in v2.** The original does
   `exit_status, result = agent.run(...)`, which unpacks the dict into its *keys* — so the
   "patch" it feeds to `git apply` is the literal string `"submission"`. We read the dict.
2. **`model_class` defaults to the tool-calling `LitellmModel`.** `swebench.yaml` sets none,
   so v2 sends `tools=[BASH_TOOL]` and parses `tool_calls` — while the prompts instruct
   ```` ```bash ```` text blocks, producing a `FormatError` every turn. `harbor.yaml` pins
   `model_class: litellm_textbased` and sets `action_regex` for the ```` ```bash ```` fence
   (v2's default fence is `mswea_bash_command`).
3. **The turn reminder is dead code.** `DefaultAgentWithReminder` overrides
   `get_observation` and uses `self.model.n_calls` / `add_message` / 
   `action_observation_template` — none of which exist in v2. The override is simply never
   called, so the reminder silently vanishes. Ours hooks `execute_actions` and reads
   `self.n_calls` off the agent.

Also: `action_observation_template` and `format_error_template` moved from `agent:` to the
**model** config in v2 (as `observation_template` / `format_error_template`). `AgentConfig`
ignores unknown keys, so leaving them under `agent:` drops them without an error.

## Config notes

- `environment.cwd` is `/app` (Harbor CodeContests' `WORKDIR`), not `/testbed`. Set it to
  match your task images.
- `environment.executable` is `podman`, as in the SWE-Bench example.
- `run.env_startup_command` creates `/logs/verifier`; some `test.sh` scripts assume it exists.
- `forward_env: []` — do not leak host env (API keys) into a container running
  model-authored code.

## Known limitation

`response_ids` are reconstructed by re-encoding `agent.messages`, and `rollout_logprobs` is
`None` — inherited from the SWE-Bench generator. Those are **not** the exact tokens the
policy emitted, and there is no importance-sampling correction. If that matters, see
`mini-swe-tinker/` in this workspace, which plugs a model directly into mini-swe-agent and
records ids and logprobs at the sampler.

## Multi-node

`instance["tests_dir"]` is read on whichever worker runs the trajectory, so `--tasks_dir`
must be on shared storage. Task images must also exist on every node: either run
`preprocess_harbor.py` on each, or push to a registry and pass `--image_prefix` plus
`--skip_build`.
