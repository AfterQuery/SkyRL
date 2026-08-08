# Training on Toolathlon

RL training on [Toolathlon](https://github.com/hkust-nlp/Toolathlon) — a benchmark of 100+
long-horizon, real-software tool-use tasks (600+ MCP tools) with per-task programmatic
evaluators.

## How it works

The integration uses Toolathlon's **decoupled agent loop** so SkyRL owns generation while
Toolathlon owns the environment and scoring:

```
SkyRL trainer
  └─ ToolathlonGenerator (one subprocess per trajectory)
       └─ bash scripts/run_single_decoupled.sh <task> ...
            ├─ task container: preprocess (seed apps, reset state)
            ├─ task container: MCP gateway (all task tools over SSE)
            ├─ host: Toolathlon agent loop ──► SkyRL OpenAI-compatible endpoint
            └─ task container: eval  ──► eval_res.json {"pass": true|false|null}
```

Per trajectory, the generator:

1. Points the harness at SkyRL's inference endpoint (`TOOLATHLON_OPENAI_BASE_URL`) with the
   trainer's sampling params (via `TOOLATHLON_MODEL_PARAMS_FILE`).
2. Reads the conversation from `traj_log.json` and the reward from `eval_res.json`
   (`pass: true → 1.0`, `false → 0.0`, `null` from max-turns → 0.0; `null` from harness
   crashes → trajectory loss-masked and retried up to 2 times).
3. Re-tokenizes the conversation with the policy chat template
   (`get_response_ids_and_loss_mask_from_messages`), loss-masking tool/user observation
   tokens.

Since training data is re-tokenized from chat messages (not exact rollout token IDs),
`rollout_logprobs` are not produced: TIS off-policy correction and fully-async training are
not supported. Tasks in the same conflict group (`tasks/finalpool/task_conflict.json`) and
repetitions of the same task are serialized with locks because they mutate shared external
app state (each run's preprocess resets it).

## Prerequisites

1. A Toolathlon checkout set up per its README:

   ```bash
   git clone https://github.com/hkust-nlp/Toolathlon
   cd Toolathlon
   bash global_preparation/install_env_minimal.sh true
   cp configs/global_configs_example.py configs/global_configs.py   # set podman_or_docker
   cp configs/token_key_session_example.py configs/token_key_session.py
   docker pull lockon0927/toolathlon-task-image:1016beta
   bash global_preparation/deploy_containers.sh true
   bash global_preparation/check_installation_containerized.sh
   ```

   Tasks touching external services (Notion, GitHub, HuggingFace, ...) need real accounts
   configured in `configs/token_key_session.py`; use the dataset filter below to start with
   self-contained tasks.

2. The policy model's tool-calling must be parseable by vLLM: set
   `generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true` and the
   matching `tool_call_parser` (e.g. `hermes` for Qwen), as done in `run_toolathlon.sh`.

## Dataset preparation

```bash
# All tasks, 90/10 train/val split:
uv run examples/train/toolathlon/prepare_toolathlon_dataset.py \
  --toolathlon-repo /path/to/Toolathlon --output-dir ~/data/toolathlon

# Restrict to tasks that only need local MCP servers (no external accounts):
uv run examples/train/toolathlon/prepare_toolathlon_dataset.py \
  --toolathlon-repo /path/to/Toolathlon --output-dir ~/data/toolathlon \
  --include-only-servers filesystem,arxiv_local,scholarly,fetch,pdf_reader
```

The output files are newline-separated task-name lists; edit them freely.

## Debug generation (recommended first step)

Runs rollouts for a few tasks without training:

```bash
uv run --isolated --extra fsdp -m examples.train.toolathlon.main_toolathlon_generate \
  data.train_data="['$HOME/data/toolathlon/train_tasks.txt']" \
  trainer.policy.model.path=Qwen/Qwen3-8B \
  generator.inference_engine.served_model_name=Qwen3-8B \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=hermes \
  trainer.algorithm.max_seq_len=32768 \
  toolathlon_config.repo_path=/path/to/Toolathlon
```

Inspect the dump directory (`toolathlon_config.dump_root`, default `~/toolathlon_rollouts`)
for `traj_log.json`, `eval_res.json`, and per-run `runner.log`.

## Training

```bash
export TOOLATHLON_REPO=/path/to/Toolathlon
bash examples/train/toolathlon/run_toolathlon.sh
```

## Configuration

Toolathlon harness settings live in `toolathlon_config.yaml` and are overridable
from the CLI as `toolathlon_config.<key>=<value>`:

| Key | Default | Meaning |
|---|---|---|
| `repo_path` | `/home/ubuntu/Toolathlon` | Toolathlon checkout root |
| `tasks_domain` | `finalpool` | Task pool under `<repo>/tasks` |
| `max_steps` | `50` | Max agent steps per task |
| `max_concurrent_tasks` | `8` | Simultaneous task containers |
| `task_timeout_seconds` | `3600` | Wall-clock limit per task run |
| `dump_root` | `~/toolathlon_rollouts` | Rollout dumps, grouped by training step |
| `image_name` | `lockon0927/toolathlon-task-image:1016beta` | Task container image |
| `eval_config` | `scripts/formal_run_v0.json` | Toolathlon run config template |

## Caveats

- Rewards are binary per task; there is no partial credit. With
  `trainer.algorithm.advantage_estimator=grpo`, prompts where all repetitions pass or all
  fail contribute no gradient signal.
- Repetitions of a task run sequentially (shared app state), so step wall-clock scales with
  `generator.n_samples_per_prompt`; each task run also pays container preprocess/eval
  overhead (typically minutes).
- Some task evaluators reach the live internet (arXiv, HuggingFace, GitHub) — rewards can be
  flaky under high parallelism.
- Toolathlon retries model calls up to 10 times on API errors; repeated malformed tool calls
  are resampled silently rather than penalized.
