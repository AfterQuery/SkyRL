# Harbor-native Toolathlon tasks

This adapter runs the JSON-world Toolathlon delivery through Harbor using the
generic MCP agent in `examples/train_integrations/harbor/mcp_agent.py`. The
model loop runs in the Harbor process and talks directly to the configured
inference endpoint. Tool actions cross Harbor's environment interface to a
small bridge that owns the container-local `t3.mcp_server`; Harbor continues
to own the environment lifecycle and deterministic verifier.

## Local evaluation

Docker and Harbor must be installed. Point the launcher at any
OpenAI-compatible endpoint that supports native `tool_calls`:

```bash
export TOOLATHLON_API_BASE=http://127.0.0.1:8000/v1
export TOOLATHLON_API_KEY=dummy
export TOOLATHLON_MODEL=Qwen3-8B

bash examples/train/toolathlon_harbor/run_eval.sh \
  -i 401k-watchlist-recency-window-refresh
```

The default task root is `toolathlon-tasks/tasks`. Override it with
`TOOLATHLON_TASKS_DIR`. The launcher builds `toolathlon-json-runtime:v1` from
the bundled runtime archive as `linux/amd64` when the image is missing.
Additional arguments are forwarded to `harbor run`, so omitting `-i` runs the
dataset and flags such as `--n-concurrent` and `--n-attempts` work normally.

The benchmark's intended prompt contract is preserved: the complete
`instruction.md` is sent as one user message and no system prompt is added.
The model endpoint and API key stay on the Harbor host and are not injected
into the task container. Because tool RPC uses Harbor's file-transfer and exec
interfaces instead of an exposed container port, the same agent operates local
Docker and remote Compute environments without changing the agent loop.

## AfterQuery Compute evaluation

Compute runs each task container remotely while Docker builds and pushes its
content-addressed image from the submission node. Before launching, set the
Compute credentials and authenticate Docker to Artifact Registry:

```bash
export COMPUTE_API_KEY=aq_your_key_here
export COMPUTE_API_URL=https://compute-api.afterquery.com

TOKEN=$(curl -sS -X POST "$COMPUTE_API_URL/api/registry/push-token" \
  -H "Authorization: Bearer $COMPUTE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"image":"harbor","tag":"latest"}' | jq -r '.token')
echo "$TOKEN" | docker login us-docker.pkg.dev \
  -u oauth2accesstoken --password-stdin

bash examples/train/toolathlon_harbor/run_compute_eval.sh \
  -i 401k-watchlist-recency-window-refresh
```

Use `COMPUTE_PROVIDER` to target a dedicated cluster. For pass@k, add
`--n-attempts K`; combine it with `--n-concurrent N` to control parallelism.
Registry tokens last approximately 15 minutes, so refresh the login before a
later run. `gcloud auth configure-docker us-docker.pkg.dev` is also supported.

## Infrastructure checks

The reference and no-op paths do not call a model:

```bash
cd toolathlon-tasks
harbor run --path tasks -i 401k-watchlist-recency-window-refresh --agent oracle
harbor run --path tasks -i 401k-watchlist-recency-window-refresh --agent nop
```

Expect rewards `1.0` and `0.0`, respectively.

## SkyRL generation and training

Use the existing Harbor entrypoint with this adapter's trial configuration and
the task directory as the dataset:

```bash
uv run --isolated --extra fsdp --extra harbor \
  -m examples.train_integrations.harbor.entrypoints.main_harbor_generate \
  data.train_data="['$PWD/toolathlon-tasks/tasks']" \
  harbor_trial_config=examples/train/toolathlon_harbor/harbor_trial_config.yaml \
  trainer.policy.model.path=Qwen/Qwen3-8B \
  generator.inference_engine.served_model_name=Qwen3-8B \
  trainer.algorithm.max_seq_len=32768 \
  generator.inference_engine.engine_init_kwargs.max_model_len=32768 \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=hermes
```

For remote task containers, set the Compute credentials and registry login as
above, then replace the config argument with:

```bash
harbor_trial_config=examples/train/toolathlon_harbor/harbor_compute_trial_config.yaml
```

The Slurm process, vLLM server, and `HarborMCPAgent` remain on the GPU cluster;
only task commands, MCP tools, and verification execute in Compute sandboxes.

SkyRL mode is deliberately strict: the vLLM endpoint must return prompt token
IDs, completion token IDs, and aligned per-token log probabilities. Evaluation
mode does not request or require those provider extensions.
