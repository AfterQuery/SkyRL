# Training on MCP-Atlas

RL training on MCP-Atlas — tool-use tasks over simulated services, scored by claim-coverage
LLM-as-judge. Everything runs through [Harbor](https://github.com/AfterQuery/harbor-aq): one
container per trial with its own seeded world.

The task set is **AQ-1000**: bundles ship pre-built, each container runs the
`mcp-atlas-runtime` image, and tools are served by a container-local REST gateway on `:1984`.
Tool state is per task. The upstream 500-task set is not supported — its adapter generated
bundles for an MCP-served sandbox this agent cannot drive, and was removed.

## Architecture

```
SkyRL trainer ──► Harbor Trial.run()
                    │
                    ├─ environment: build hb__<hash> from the task's 5-line Dockerfile
                    │               (FROM mcp-atlas-runtime + AQ_SIM_* vars), run with
                    │               CMD replaced by a keepalive
                    │
                    ├─ agent (harbor.agents.installed.mcp_atlas, name `mcp-atlas`)
                    │    setup: exec run_agent_environment.sh --> seeds the task's world,
                    │           execs `uvicorn atlas_sim_env:app` on :1984; poll /health;
                    │           assert /reset-state reports selected_overlay
                    │    run:   upload mcp_atlas_runner.py, exec it IN the container
                    │             loop: POST {api_base}/chat/completions with the task's
                    │                   tools --> native tool_calls
                    │                   POST 127.0.0.1:1984/call-tool
                    │             until a turn makes no calls | max_steps | deadline
                    │           write the final message to /app/answer.md
                    │
                    └─ verifier: tests/test.sh -> tests/grade.py, in the container
                         one judge call per rubric claim (1.0 / 0.5 / 0.0), averaged
                         --> raw coverage in [0,1] --> /logs/verifier/reward.json
```

Four details that are easy to get wrong:

**The loop runs inside the container.** `mcp_atlas_runner.py` (stdlib only) is uploaded and
exec'd in the environment, so tool calls are localhost. The agent exposes exactly the task's
served tools and nothing else — no system prompt, no shell, no file tools. That matters because
MCP-Atlas grades tool selection among distractors, and the gateway listens on localhost inside
the same container: give the model a shell and it can curl the gateway directly, and the
allowlist stops binding.

**The agent starts the gateway, because Harbor displaced it.**
`run_agent_environment.sh` is the image's own `CMD`; Harbor replaces `CMD` with a keepalive so
it has a long-lived container to `exec` into, so nothing starts the tool API but `setup()`.
It then asserts `/reset-state` reports `selected_overlay` — a sandbox that silently falls back
to the base seed boots perfectly and grades meaningless answers.

**A loopback `api_base` is rewritten.** `http://127.0.0.1:8000/v1` means the container itself
once the loop runs in-container, so the agent resolves the host from `/proc/net/route`. Note
Harbor gives each trial its own network, so the gateway differs per trial (172.20.0.1,
172.21.0.1, …). Serve vLLM with `--host 0.0.0.0` or the container cannot reach it.

**Steps and tool output are uncapped by default**, matching upstream's `--max-turns` and
`--tool-output-cap` defaults of `None`. The loop takes a wall-clock deadline derived from
Harbor's agent timeout, so a non-terminating rollout still yields its trajectory.

The tool allowlist is enforced server-side, not by agent etiquette: `AQ_SIM_ENABLED_SERVERS`
decides which services boot, so only the task's tools exist at all (typically ~65).

## Prerequisites

1. **The `harbor` dependency must be our fork.** `pyproject.toml` pins
   `AfterQuery/harbor-aq`, which carries the `mcp-atlas` agent; on upstream Harbor that agent
   name is unknown and trials fail at construction. Bump the `rev` and re-run `uv lock` after
   every harbor push, or the new kwargs get silently swallowed by the old pin.

2. **The runtime image, locally.** Task Dockerfiles say `FROM mcp-atlas-runtime:<tag>` with no
   registry, so the tag must resolve on the build host:

   ```bash
   gcloud auth activate-service-account --key-file ~/.gcp/afterquery-image-pull.json
   gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin \
     https://us-east1-docker.pkg.dev
   docker pull us-east1-docker.pkg.dev/afterqueryai/mcp-atlas-redelivery-staging/runtime:redeliv-final3-20260622
   docker tag  us-east1-docker.pkg.dev/afterqueryai/mcp-atlas-redelivery-staging/runtime:redeliv-final3-20260622 \
               mcp-atlas-runtime:delivery7-20260625
   ```

   Note the substitution: `delivery7-20260625` is not published anywhere reachable, so the
   June 22 staging runtime stands in under that tag. 995/1000 task overlays are present in it
   and a 40-task oracle sweep scored 1.0, so it is a sound stand-in — but it is a local alias,
   and it is why `environment.type: docker` is the default while
   **Daytona cannot work yet**: a remote builder has no such tag, and fixing that means
   pushing the runtime to a registry and rewriting the task `FROM` lines fully-qualified.

3. **Patch the bundles to use the LLM judge.** AQ bundles ship a deterministic proxy grader
   whose own docstring invites replacement. It is maximised by restating claim text, so a
   policy trained against it learns to parrot claim-shaped sentences.

   Their `tests/test.sh` already execs `tests/grade.py`, so the swap is one file. Do it into
   a **copy** -- patching the pristine set in place is how a task set ends up with stale
   graders:

   ```bash
   cp -r $HOME/AQ-MCP-Atlas-1000-Tasks/tasks $HOME/data/mcp_atlas_tasks
   for d in $HOME/data/mcp_atlas_tasks/*/; do
     cp /path/to/harbor/adapters/mcp_atlas/template/tests/grade.py "$d/tests/grade.py"
   done
   ```

   `rubric.json` and the `AQ_SIM_*` Dockerfile are untouched.

4. **API keys**, in `.env.mcp_atlas.local` (gitignored, mode 600). The committed
   `.env.mcp_atlas` holds placeholders only.

   - **Judge — required.** `MCP_ATLAS_JUDGE_BASE_URL` / `MCP_ATLAS_JUDGE_KEY` /
     `MCP_ATLAS_JUDGE_MODEL`. This *is* the reward, so nothing trains without it, and it must
     never point at the policy being trained.
   - **Agent inference key.** `MCP_ATLAS_API_KEY`. A local vLLM ignores it, but the runner
     refuses to start without one.
   - **MCP server keys — optional** (`GITHUB_TOKEN`, `NOTION_TOKEN`, `BRAVE_API_KEY`, …).
     Mostly irrelevant for AQ-1000, whose services are simulated in-image; they matter for the
     upstream 500.

   Keys reach the container as `${VAR}` templates in `verifier.env` / `environment.env`;
   Harbor resolves them from the host at trial start and re-serializes sensitive values back
   to `${VAR}` when persisting the config, so they stay out of trial logs. Optional keys use
   `${VAR:-}` so a missing one leaves that server disabled instead of failing the trial.

5. **A vLLM server *with* native tool calling.** The runner reads structured `tool_calls`, and
   vLLM rejects a request carrying `tools=` under the default `tool_choice="auto"` unless both
   flags are set:

   ```bash
   uv run --isolated --extra fsdp python -m vllm.entrypoints.openai.api_server \
     --model <policy> --served-model-name Qwen3-30B-A3B \
     --enable-auto-tool-choice --tool-call-parser hermes \
     --data-parallel-size 4 --tensor-parallel-size 2 \
     --max-model-len 32768 --host 0.0.0.0 --port 8000
   ```

   This inverts the previous host-side agent, which required those flags to be *off*.

## Eval

Sanity-check the bundles, then run the policy:

```bash
harbor run -p $HOME/data/mcp_atlas_eval20 -a oracle   # expect 1.0
harbor run -p $HOME/data/mcp_atlas_eval20 -a nop      # expect 0.0

set -a; . examples/train/mcp_atlas/.env.mcp_atlas.local; set +a
uv run --extra harbor harbor run -p $HOME/data/mcp_atlas_eval20 \
  --config examples/train/mcp_atlas/harbor_eval_config.yaml
```

`harbor_eval_config.yaml` holds the agent kwargs (`api_base`, `api_key`, `max_steps`,
`tool_output_cap`, `max_tokens`) and the judge under `verifier.env`.

### Measured

| run | coverage | notes |
|---|---|---|
| `oracle` | **1.000** | bounds the scale from above |
| `nop` | **0.000** | bounds it from below |
| Qwen3-30B-A3B, untrained, 20 tasks | **0.160** | 0 exceptions; 1/20 above the 0.75 threshold |

The base model's failure mode is *not calling tools at all*: 14 of 20 trials made zero tool
calls and answered from parametric knowledge. Every trial ended `complete` — no truncation, no
context overflow — so 0.160 is a clean baseline.

## Warm start: SFT on GLM-5.2 teacher trajectories

RL from a cold start is expensive here: a model that emits malformed tool calls earns coverage
0 on every rollout, so GRPO sees no reward variance and no gradient. Distilling a stronger
teacher's tool-use behaviour first gives RL a policy that already calls tools correctly.

Inputs are the two AfterQuery bundles (1000 Harbor-schema tasks, and 3 GLM-5.2 rollouts per
task graded with the same claim-coverage judge). Unzip both, then:

```bash
uv run examples/train/mcp_atlas/prepare_glm_sft_dataset.py \
  --trajectories-dir ~/AQ-MCP-Atlas-1000-Trajectories-GLM-5.2 \
  --tasks-dir ~/AQ-MCP-Atlas-1000-Tasks \
  --output-dir ~/data/mcp_atlas_sft \
  --min-coverage 1.0 --one-per-task

bash examples/train/mcp_atlas/run_sft_glm_warmstart.sh   # Qwen3-30B-A3B, LoRA, max_length=32768
```

LoRA checkpoints hold a PEFT adapter at
`$CKPT_PATH/global_step_<N>/policy/lora_adapter/`, not a merged model — SkyRL has no
`merge_and_unload`, so `hf_save_interval` stays 0 rather than producing a misleading "export".
Merge before RL, or RL silently starts from base weights:

```bash
uv run examples/train/mcp_atlas/merge_lora_adapter.py \
  --base Qwen/Qwen3-30B-A3B \
  --adapter $HOME/mcp_atlas_sft_run/ckpts/global_step_32/policy/lora_adapter \
  --output $HOME/mcp_atlas_sft_run/merged
```

The converter handles three things that matter:

- **Anthropic → OpenAI messages.** The bundle stores assistant `content` as a block list
  (`text` / `tool_use` with a JSON-string `input`) and results under `tool_use_id`;
  `SFTTrainer` needs text content plus a `tool_calls` list and `tool_call_id`.
- **Observations flattened exactly as the agent flattens them** (MCP blocks joined on `text`,
  capped at `--max-tool-output-chars`, default 10000 = the agent's `tool_output_cap`). A
  different observation format in SFT than RL produces is the main avoidable train/rollout
  mismatch.
- **Tool schemas reconstructed from teacher usage**, because neither bundle ships them (they
  live in the runtime image). Argument keys are unioned per tool name; every tool in the task's
  `enabled_tools` is included — *including distractors the teacher never called* — so the
  student faces the same choice the teacher did.

Selection knobs: `--min-coverage 1.0 --one-per-task` yields 531 rollouts over 531 tasks
(505 train / 26 validation). Dropping `--one-per-task` gives 1207 rows, but 256 of those 534
tasks have all three runs perfect, so easy tasks would supply 64% of the data; measured across
pairs of perfect runs the extra rollouts are largely redundant (mean tool-set Jaccard 0.855).
`--prefer shortest` (default) breaks the frequent coverage-1.0 ties toward the fewest-tool-call
demonstration. Rollouts with `status != graded` are always excluded — those are GLM never
terminating (one hit 1053 tool calls), which carry no answer to learn from.

**Reasoning and the chat template.** GLM's reasoning traces were not saved in the bundle, so
the stock Qwen3 template injects an empty `<think>\n\n</think>` into the **final** assistant
turn of every row — its `loop.last or reasoning_content` branch fires with nothing to put
inside. Those tokens land in the trained span (`loss_mask=1`), teaching the model to open and
immediately close its reasoning on exactly the turn that produces the graded answer.
`enable_thinking=False` does not suppress it; that flag only affects the generation prompt.

Two further stock behaviours matter for RL, both avoided by the same template:

- It **rewrites** think blocks it does keep (`<think>R</think>X` becomes
  `<think>\nR\n</think>\n\nX`), so re-rendering a rollout yields different tokens than the
  policy emitted.
- It **strips** reasoning from assistant turns at or before `last_query_index` — the last
  *genuine* user message, where a wholly `<tool_response>`-wrapped user message does not
  count. Pure tool-calling rollouts have one genuine query at index 0, so nothing is stripped
  there; it bites only when real user turns are interleaved.

The fix is `skyrl/train/utils/templates/qwen3_acc_thinking.jinja2`, which never injects a think
block and never strips reasoning from earlier turns. Since `SFTConfig` has no `chat_template`
field (the online-tokenization worker passes a fixed argument tuple), the prep script emits the
dataset **pre-tokenized** — `input_ids` plus a full-sequence `loss_mask`, consumed via
`pretokenized_dataset_paths`. `run_mcp_atlas_harbor.sh` serves the same template to vLLM via
`engine_init_kwargs.chat_template`, so SFT and rollout serialization match.

Matching matters beyond the empty block: the stock template strips reasoning from all but the
last assistant turn, so a thinking policy's rollout context and its re-tokenized training
sequence would disagree — RL would compute logprobs on a context the policy never had.

Pass `--no-pretokenize` to emit `messages`/`tools` for the online path instead, accepting the
injected think blocks.

## Training

```bash
export MCP_ATLAS_JUDGE_BASE_URL=... MCP_ATLAS_JUDGE_KEY=... MCP_ATLAS_JUDGE_MODEL=... WANDB_API_KEY=...
bash examples/train/mcp_atlas/run_mcp_atlas_harbor.sh \
  trainer.policy.model.path=$HOME/mcp_atlas_sft_run/merged
```

**No step-wise training on these rollouts.** The in-container loop calls the model API
directly, so Harbor's LLM layer never sees the calls and no per-turn token IDs or logprobs
reach `context.rollout_details`. Off-policy correction (TIS) needs those; training here means
re-tokenizing the finished conversation from `trajectory.json`, which the agent downloads per
trial.

Reward is continuous coverage in [0,1], which GRPO needs variance within a group to use; the
judge's 0.5-per-claim granularity is what supplies it. Judge cost is one LLM call per claim per
rollout (~2-6). Judge failures make the verifier fail loudly rather than emit a misleading 0.0
— "unscored" and "answered wrong" are different outcomes, and conflating them poisons the
signal.

## Known gaps

- **Truncated turns are mistaken for final answers.** If the model is still emitting a tool
  call when it hits `max_tokens`, the JSON is cut mid-string, vLLM's hermes parser throws, and
  the calls stay in `content`. The runner sees no `tool_calls`, concludes the model is done,
  and writes that raw text as the answer. The fix is to check `finish_reason == "length"`;
  it is not yet implemented, and it depresses a tool-happy policy specifically.
- **Long rollouts overflow the context.** Every turn re-sends the whole history plus all tool
  output, so with uncapped output the prompt eventually exceeds `--max-model-len` and the
  endpoint returns HTTP 400. Cap `tool_output_cap` for a 32k window.
- The runtime image is the June 22 substitute described above, and 5 of 1000 tasks have no
  overlay in it.
- Daytona is configured but blocked on that image being registry-hosted with a fully-qualified
  `FROM`.
- `train_integrations/harbor/run_codecontest.sh` has not been re-run since the Harbor
  0.13→0.18 bump that the fork brings; check it before relying on that example.
