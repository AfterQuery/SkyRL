# Training on MCP-Atlas

RL training on [MCP-Atlas](https://github.com/scaleapi/mcp-atlas) — Scale AI's benchmark of
500 tool-use tasks over 36 real MCP servers (~307 tools) in a reproducible Docker sandbox,
scored by claim-coverage LLM-as-judge.

## How it works

Unlike the official eval (which routes through a TypeScript harness), the generator owns the
agent loop directly — the harness pins its LLM endpoint at boot and strips token metadata, so
bypassing it is both simpler and more faithful for training:

```
SkyRL trainer
  └─ MCPAtlasGenerator (agent loop per trajectory)
       ├─ POST {sandbox}/list-tools        (once; tool schemas → OpenAI function format)
       ├─ loop: SkyRL /v1/chat/completions (policy, with task's enabled tools)
       │        POST {sandbox}/call-tool   (execute tool_calls, observations → tool messages)
       │        until no tool_calls | max_turns | max_tool_calls
       └─ reward: ClaimCoverageJudge — per-claim LLM judge over the final response,
                  fulfilled=1 / partial=0.5 / not=0, averaged → coverage ∈ [0,1]
```

The judge prompt and scoring are copied verbatim from the official scorer
(`services/scoring/score_claims.py`), so rewards equal the benchmark's leaderboard metric
(`coverage_score`; the leaderboard's pass rates are thresholds over it). Conversations are
re-tokenized with the policy chat template, tool observations loss-masked. As with the
Toolathlon example, re-tokenization means no `rollout_logprobs` (no TIS / fully-async).

**State warning:** all rollouts share ONE sandbox container — filesystem, memory graph, git
trees, and a 48h tool-response cache are global and never reset. The generator serializes
repetitions of the same task, but for mutating tasks you should prepare the dataset with
`--exclude-mutating` (recommended) and/or recycle the sandbox container between epochs.

## Prerequisites

1. **The sandbox** (one long-running container, ~8-10 GB RAM):

   ```bash
   cd /path/to/mcp-atlas
   cp env.template .env       # add MCP server API keys if you have them (optional)
   docker pull ghcr.io/scaleapi/mcp-atlas:1.2.7
   docker tag ghcr.io/scaleapi/mcp-atlas:1.2.7 agent-environment:latest
   docker run -d --name mcp-atlas-sandbox -p 1984:1984 --env-file .env agent-environment:latest
   # wait 1-3 min, then verify:
   curl -s http://localhost:1984/enabled-servers | jq -c
   ```

   With no API keys, 20 of 36 servers run (arxiv, calculator, fetch, filesystem, git,
   wikipedia, ddg-search, ...). Keys in `.env` enable more (github, brave-search, notion, ...).

2. **A judge endpoint** — the reward source. Set `EVAL_LLM_BASE_URL` / `EVAL_LLM_API_KEY` /
   `EVAL_LLM_MODEL` (or `mcp_atlas_config.judge.*`) to a real LLM API (the official default
   judge is `gemini/gemini-3.1-pro-preview` via LiteLLM). It must never point at the policy
   being trained.

3. vLLM tool-call parsing for the policy: `engine_init_kwargs.enable_auto_tool_choice=true`
   and the matching `tool_call_parser` (e.g. `hermes` for Qwen), as in `run_mcp_atlas.sh`.

## Dataset preparation

```bash
# Recommended: only tasks whose tools the sandbox actually serves, and no state-mutating tools
uv run examples/train/mcp_atlas/prepare_mcp_atlas_dataset.py --output-dir ~/data/mcp_atlas \
  --available-tools-only --exclude-mutating
```

Writes `train.parquet` / `val.parquet` in SkyRL's standard PromptDataset format; the
`task_id`, `enabled_tools_json`, and `gtfa_claims_json` columns reach the generator as
env_extras. Ground truth is `GTFA_CLAIMS`: 2-6 natural-language claims per task that the
final answer must support.

## Debug generation (recommended first step)

```bash
uv run --isolated --extra fsdp -m examples.train.mcp_atlas.main_mcp_atlas_generate \
  data.train_data="['$HOME/data/mcp_atlas/train.parquet']" \
  trainer.policy.model.path=Qwen/Qwen3-8B \
  generator.inference_engine.served_model_name=Qwen3-8B \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=hermes \
  trainer.algorithm.max_seq_len=32768 \
  mcp_atlas_config.judge.enabled=false   # rollouts only, no judge needed
```

Inspect per-trajectory conversation dumps under `mcp_atlas_config.dump_root`
(default `~/mcp_atlas_rollouts`).

## Warm start: SFT on GLM-5.2 teacher trajectories

RL from a cold start is expensive here: an 8B model that emits malformed tool calls earns
coverage 0 on every rollout, so GRPO sees no reward variance and no gradient. Distilling a
stronger teacher's tool-use behaviour first gives RL a policy that already calls tools
correctly.

Inputs are the two AfterQuery bundles (1000 Harbor-schema MCP-Atlas tasks, and 3 GLM-5.2
rollouts per task graded with the same claim-coverage judge). Unzip both, then:

```bash
uv run examples/train/mcp_atlas/prepare_glm_sft_dataset.py \
  --trajectories-dir ~/AQ-MCP-Atlas-1000-Trajectories-GLM-5.2 \
  --tasks-dir ~/AQ-MCP-Atlas-1000-Tasks \
  --output-dir ~/data/mcp_atlas_sft \
  --min-coverage 0.75 --one-per-task

bash examples/train/mcp_atlas/run_sft_glm_warmstart.sh          # Qwen3-8B, FSDP, 8 GPUs
```

Then start RL from the exported checkpoint:

```bash
bash examples/train/mcp_atlas/run_mcp_atlas.sh \
  trainer.policy.model.path=$HOME/mcp_atlas_sft_run/hf_exports/global_step_86
```

The converter handles three things that matter:

- **Anthropic → OpenAI messages.** The bundle stores assistant `content` as a block list
  (`text` / `tool_use` with a JSON-string `input`) and results under `tool_use_id`;
  `SFTTrainer` needs text content plus a `tool_calls` list and `tool_call_id`.
- **Observations flattened exactly as the RL generator flattens them** (MCP blocks joined on
  `text`, capped at `--max-tool-output-chars`, default 10000 = the generator's
  `tool_output_cap`). A different observation format in SFT than RL produces is the main
  avoidable train/rollout mismatch.
- **Tool schemas reconstructed from teacher usage**, because neither bundle ships them (they
  live in the `mcp-atlas-runtime` image). Argument keys are unioned per tool name; every tool
  in the task's `enabled_tools` is included — *including distractors the teacher never
  called* — so the student faces the same choice the teacher did. Tools never called anywhere
  get an empty parameter object; load the runtime image if you need exact schemas.

Selection knobs: `--min-coverage 0.75 --one-per-task` yields 718 rollouts over 718 tasks
(683 train / 35 validation). Dropping `--one-per-task` gives ~1862 rows but lets easy tasks
with three good runs dominate; `--min-coverage 0.99` keeps only near-perfect demonstrations.
Rollouts with `status != graded` are always excluded — those are GLM never terminating
(one hit 1053 tool calls), which carry no answer to learn from.

Measured token lengths on this dataset (Qwen3-8B tokenizer): median 2733, p90 4771,
p99 12063, max 28475 — hence `max_length=16384` in the run script, which keeps ~99.5% of
trajectories whole. The prep script prints these stats so the value can be re-derived if you
change the filters.

**Note on reasoning:** GLM's reasoning traces were not saved in the bundle, so Qwen3's chat
template renders an empty `<think></think>` before each assistant turn. The student therefore
learns to act without visible reasoning. If you would rather preserve thinking, use
`skyrl/train/utils/templates/qwen3_acc_thinking.jinja2` — decide before a real run, since it
shapes the policy RL starts from.

## Training

```bash
export EVAL_LLM_BASE_URL=... EVAL_LLM_API_KEY=... EVAL_LLM_MODEL=...
bash examples/train/mcp_atlas/run_mcp_atlas.sh
```

## Configuration

Settings live in `mcp_atlas_config.yaml`, overridable as `mcp_atlas_config.<key>=<value>`:

| Key | Default | Meaning |
|---|---|---|
| `sandbox_url` | `http://localhost:1984` | The running sandbox container |
| `max_turns` / `max_tool_calls` | `32` / `50` | Agent loop limits (official eval: 256/100) |
| `tool_output_cap` | `10000` | Char cap per tool observation (null = uncapped) |
| `max_concurrent_tasks` | `8` | Simultaneous rollouts against the sandbox |
| `dump_root` | `~/mcp_atlas_rollouts` | Per-trajectory conversation dumps |
| `judge.*` | env `EVAL_LLM_*` | Judge endpoint, retries, concurrency |

## Caveats

- Reward is continuous coverage in [0,1] — with GRPO, groups still need reward variance to
  produce gradient; the judge's 0.5 granularity per claim helps.
- Judge cost: one LLM call per claim per rollout (~2-6 calls). Judge failures after retries
  loss-mask the trajectory rather than injecting a fake 0 reward.
- Many of the 500 tasks need API-key-gated servers (github, notion, airtable, mongodb,
  slack, ...); without keys those tool calls fail. Use `--available-tools-only` to keep only
  runnable tasks.
- Known image issue (`1.2.7`, observed 2026-08): the 7 Python/uvx-based servers (arxiv,
  calculator, cli-mcp-server, ddg-search, fetch, git, pubmed) fail at startup because `uvx`
  resolves a current `mcp` library that is API-incompatible with the pinned server versions
  (`UV_OFFLINE=1` does not help). Until fixed upstream, only ~13 no-key servers come online,
  which shrinks the no-key task pool drastically — configure API keys for more servers, or
  rebuild the image with `mcp` pinned in `mcp_server_template.json` (`--with mcp==<version>`).
- The sandbox's 48h tool-response cache makes repeated identical tool calls deterministic and
  cheap (usually good for RL); `POST {sandbox}/cache-clear` resets it.
