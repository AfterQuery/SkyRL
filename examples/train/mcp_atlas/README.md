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

2. **API keys**, declared in `.env.mcp_atlas` and loaded automatically by both run scripts
   (override the path with `ENV_FILE=...`):

   - **Judge — required.** `EVAL_LLM_BASE_URL` / `EVAL_LLM_API_KEY` / `EVAL_LLM_MODEL`. This
     *is* the reward signal, so nothing trains without it, and it must never point at the
     policy being trained. The upstream default judge is `gemini/gemini-3.1-pro-preview`.
   - **MCP server keys — optional**, one group per key-gated server (`GITHUB_TOKEN`,
     `NOTION_TOKEN`, `BRAVE_API_KEY`, `AIRTABLE_API_KEY`, …). Only 35 of the 500 upstream
     tasks need no keys at all, so these decide how much of the benchmark is reachable. A
     server whose keys are absent is simply not enabled; only tasks needing it fail.

   `.env.mcp_atlas` is **committed and holds placeholders only**. Put real secrets in
   `.env.mcp_atlas.local` (gitignored) and run with
   `ENV_FILE=examples/train/mcp_atlas/.env.mcp_atlas.local`, or export them in your shell.

   On the Harbor path the keys reach the task container through
   `harbor_trial_config.yaml`, which references them as `${VAR}` templates. Harbor resolves
   those from the host at trial start and re-serializes sensitive values back to `${VAR}`
   when persisting the config, so keys stay out of trial logs. Optional keys use
   `${VAR:-}` so a missing one leaves that server disabled instead of failing the trial.

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
  --min-coverage 1.0 --one-per-task

bash examples/train/mcp_atlas/run_sft_glm_warmstart.sh          # Qwen3-8B, FSDP, 8 GPUs
```

Then start RL from the exported checkpoint:

```bash
bash examples/train/mcp_atlas/run_mcp_atlas.sh \
  trainer.policy.model.path=$HOME/mcp_atlas_sft_run/hf_exports/global_step_32
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

Selection knobs: `--min-coverage 1.0 --one-per-task` yields 531 rollouts over 531 tasks
(505 train / 26 validation). Dropping `--one-per-task` gives 1207 rows, but 256 of those 534 tasks have all three runs
perfect, so easy tasks would supply 64% of the data; measured across pairs of perfect runs the
extra rollouts are largely redundant (mean tool-set Jaccard 0.855). `--prefer shortest`
(default) breaks the frequent coverage-1.0 ties toward the fewest-tool-call demonstration.
Rollouts with `status != graded` are always excluded — those are GLM never terminating
(one hit 1053 tool calls), which carry no answer to learn from.

**Reasoning and the chat template.** GLM's reasoning traces were not saved in the bundle, so
the stock Qwen3 template injects an empty `<think>\n\n</think>` into *every* assistant turn —
and those tokens land inside the trained span (loss_mask=1), which actively teaches the model
to skip reasoning. `enable_thinking=False` does not suppress it for completed messages.

The fix is `skyrl/train/utils/templates/qwen3_acc_thinking.jinja2`, which never injects a think
block and never strips reasoning from earlier turns. Since `SFTConfig` has no `chat_template`
field (the online-tokenization worker passes a fixed argument tuple), the prep script emits the
dataset **pre-tokenized** — `input_ids` plus a full-sequence `loss_mask`, consumed via
`pretokenized_dataset_paths`. `run_mcp_atlas.sh` serves the same template to vLLM via
`engine_init_kwargs.chat_template`, so SFT and rollout serialization match.

Matching matters beyond the empty block: the stock template strips reasoning from all but the
last assistant turn, so a thinking policy's rollout context and its re-tokenized training
sequence would disagree — RL would compute logprobs on a context the policy never had.

Measured over all 531 rows (Qwen3-8B tokenizer): mean 3088 tokens, median 2599, p90 4252,
p99 11134, max 28558; trained tokens mean 788 / max 2591 (25.5% of all tokens, the rest being
the tool-schema block and observations). `max_length=16384` keeps 529/531 rows whole.
Pass `--no-pretokenize` to emit `messages`/`tools` for the online path instead, accepting the
injected think blocks.

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

## Alternative runner: Harbor

`run_mcp_atlas.sh` above drives one shared sandbox through `MCPAtlasGenerator`. There is a
second path that runs the same benchmark through [Harbor](https://github.com/harbor-framework/harbor)
instead, which buys three things the shared-sandbox path cannot give:

- **Per-task containers with seeded state**, so tasks that mutate their environment don't
  contaminate each other (the shared sandbox never resets).
- **Exact per-turn token IDs and logprobs**, so training is step-wise with off-policy
  correction (TIS) rather than re-tokenizing a finished conversation.
- **Harbor's verifier**, so reward is produced inside the task container.

It needs no new SkyRL generator — the existing Harbor entrypoint plus a trial config that
selects the agent is enough:

```bash
# 1. Generate Harbor bundles (in the harbor checkout)
cd /path/to/harbor/adapters/mcp_atlas
uv run run_adapter.py --hf --image <image exposing MCP> \
  --mcp-url http://localhost:1984/mcp --out $HOME/data/mcp_atlas_harbor/tasks

# 2. Sanity check the bundles
harbor run -p $HOME/data/mcp_atlas_harbor/tasks -a oracle   # expect reward 1.0
harbor run -p $HOME/data/mcp_atlas_harbor/tasks -a nop      # expect reward 0.0

# 3. Train
export EVAL_LLM_BASE_URL=... EVAL_LLM_API_KEY=... EVAL_LLM_MODEL=...
bash examples/train/mcp_atlas/run_mcp_atlas_harbor.sh
```

The agent and adapter live in Harbor, not here: `harbor.agents.installed.mcp_atlas` (registered
as `mcp-atlas`) and `adapters/mcp_atlas/`. `harbor_trial_config.yaml` selects it with
`agent.name: mcp-atlas`.

Three requirements that are easy to miss:

1. **The `harbor` dependency must contain the agent.** `pyproject.toml` currently pins
   upstream Harbor, where `mcp-atlas` is an unknown agent name and trials fail at
   construction. Point it at the fork carrying the agent (`AfterQuery/harbor-aq`, branch
   `aq`). Note that fork is Harbor 0.18.0 while the pin is 0.13.1, so this also upgrades
   Harbor for the existing `train_integrations/harbor` examples — re-run
   `run_codecontest.sh` as a regression check before relying on it.
2. **Do not enable a server-side tool-call parser.** `run_mcp_atlas_harbor.sh` deliberately
   omits `enable_auto_tool_choice` / `tool_call_parser`, unlike `run_mcp_atlas.sh`. Harbor's
   `LLMResponse` has no `tool_calls` field, so the agent parses
   `<tool_call>{...}</tool_call>` from the response text; a server-side parser moves those
   calls out of `content` and the loop sees none.
3. **The environment image must expose MCP** over sse or streamable-http. Upstream's sandbox
   serves a REST API (`/list-tools`, `/call-tool`) instead. `adapters/mcp_atlas/mcp_bridge.py`
   in the Harbor fork re-exposes that REST API as MCP, which is enough to run locally:

   ```bash
   uv run --with mcp --with httpx mcp_bridge.py --sandbox http://localhost:1984 --port 1985
   ```

   Bind it to the interface the agent connects from -- FastMCP rejects mismatched Host
   headers with HTTP 421.

4. **Environments default to Daytona** (`environment.type` in `harbor_trial_config.yaml`),
   which gives one cloud sandbox per trial and lets rollouts parallelise beyond this machine.
   Set `DAYTONA_API_KEY` (or `DAYTONA_JWT_TOKEN` + `DAYTONA_ORGANIZATION_ID`) in
   `.env.mcp_atlas`. Switch to `type: docker` to run locally.

   **Two things do not survive the move to cloud**, because a Daytona sandbox cannot reach
   this machine's localhost: the judge in `verifier.env` must be a publicly reachable
   endpoint, and the MCP endpoint should live inside the task image rather than behind a
   host-local bridge.

5. **Tool state is shared, not per-task.** Harbor gives each trial its own container, but
   every task's tool calls still land in the one shared MCP-Atlas sandbox -- same filesystem,
   memory graph, and response cache. Read-only tasks are unaffected, which is what
   `--exclude-mutating` is for; genuine isolation needs the MCP server running inside each
   task's own environment.
