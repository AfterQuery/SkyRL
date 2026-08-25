# Harbor-native Toolathlon tasks

This adapter runs the JSON-world Toolathlon delivery through Harbor using the
generic MCP agent in `examples/train_integrations/harbor/mcp_agent.py`. The
model receives only the task instruction and the tools discovered from the
container-local `t3.mcp_server`; Harbor continues to own the Docker lifecycle
and deterministic verifier.

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

The default task root is `toolathlon-tasks/tasks/tasks`. Override it with
`TOOLATHLON_TASKS_DIR`. The launcher builds `toolathlon-json-runtime:v1` from
the bundled runtime archive when the image is missing. Additional arguments
are forwarded to `harbor run`, so omitting `-i` runs the dataset and flags such
as `--n-concurrent` work normally.

The benchmark's intended prompt contract is preserved: the complete
`instruction.md` is sent as one user message and no system prompt is added.

## Infrastructure checks

The reference and no-op paths do not call a model:

```bash
cd toolathlon-tasks/tasks
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
  data.train_data="['$PWD/toolathlon-tasks/tasks/tasks']" \
  harbor_trial_config=examples/train/toolathlon_harbor/harbor_trial_config.yaml \
  trainer.policy.model.path=Qwen/Qwen3-8B \
  generator.inference_engine.served_model_name=Qwen3-8B \
  trainer.algorithm.max_seq_len=32768 \
  generator.inference_engine.engine_init_kwargs.max_model_len=32768 \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=hermes
```

SkyRL mode is deliberately strict: the vLLM endpoint must return prompt token
IDs, completion token IDs, and aligned per-token log probabilities. Evaluation
mode does not request or require those provider extensions.
