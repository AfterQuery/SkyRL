import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from examples.train.toolathlon_harbor.toolathlon_context_manager import (
    create_managed_context,
)

RUNNER = Path(__file__).parents[2] / "examples/train_integrations/harbor/mcp_runner.py"
SPEC = importlib.util.spec_from_file_location("harbor_mcp_runner", RUNNER)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_openai_tools_preserves_mcp_schema():
    tools = [
        {
            "name": "calendar__create",
            "description": "Create an event",
            "inputSchema": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
            },
        }
    ]
    assert runner.openai_tools(tools) == [
        {
            "type": "function",
            "function": {
                "name": "calendar__create",
                "description": "Create an event",
                "parameters": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                },
            },
        }
    ]


def test_token_data_reads_vllm_extensions():
    response = {
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "token_ids": [3, 4],
                "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
            }
        ],
    }
    assert runner.token_data(response) == ([1, 2], [3, 4], [-0.1, -0.2])


def test_tool_error_is_visible_to_model():
    result = {"content": [{"type": "text", "text": "bad arguments"}], "is_error": True}
    assert runner.tool_result_text(result) == "Error: bad arguments"


@pytest.mark.asyncio
async def test_completion_routes_session_and_keeps_loopback_url_on_host():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await runner.completion(
            client,
            api_base="http://127.0.0.1:8000/v1/",
            api_key="secret",
            model="Qwen3-8B",
            messages=[{"role": "user", "content": "task"}],
            tools=[],
            max_tokens=32,
            temperature=0.0,
            collect_rollout_details=True,
            session_id="session-123",
        )

    assert seen["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert seen["headers"]["x-session-id"] == "session-123"
    assert seen["payload"]["session_id"] == "session-123"
    assert seen["payload"]["return_token_ids"] is True


@pytest.mark.asyncio
async def test_completion_classifies_vllm_context_rejection():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "maximum context length is 32768 tokens"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(runner.ContextLengthExceeded):
            await runner.completion(
                client,
                api_base="http://model/v1",
                api_key="dummy",
                model="model",
                messages=[{"role": "user", "content": "task"}],
                tools=[],
                max_tokens=32,
                temperature=0.0,
                collect_rollout_details=False,
                session_id=None,
            )


@pytest.mark.asyncio
async def test_run_loop_recovers_from_context_rejection(monkeypatch, tmp_path):
    requests = []

    async def fake_completion(_client, **kwargs):
        requests.append(kwargs["messages"])
        if len(requests) == 1:
            raise runner.ContextLengthExceeded("too long")
        return {
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 1},
        }

    checkpoints = []

    async def checkpoint(value):
        checkpoints.append(value)

    async def call_tool(_name, _arguments):
        raise AssertionError("no tool call expected")

    monkeypatch.setattr(runner, "completion", fake_completion)
    result = await runner.run_loop(
        instruction="task",
        bridge_tools=[],
        call_tool=call_tool,
        checkpoint=checkpoint,
        api_base="http://model/v1",
        api_key="dummy",
        model="model",
        max_turns=3,
        deadline_sec=60,
        request_timeout_sec=30,
        max_tokens=32,
        temperature=0.0,
        collect_rollout_details=False,
        strict_rollout_details=False,
        artifact_dir=tmp_path / "context",
        context_manager_factory=create_managed_context,
        max_context_tokens=256,
        context_safety_tokens=16,
    )

    assert result["stop_reason"] == "complete"
    assert any(event["type"] == "emergency_reset" for event in result["context_events"])
    assert "Context reset" in requests[1][1]["content"]
    assert checkpoints


def test_agent_uses_environment_rpc_without_container_model_credentials():
    agent_source = (RUNNER.parent / "mcp_agent.py").read_text()
    bridge_source = (RUNNER.parent / "remote_mcp.py").read_text()
    assert "environment.upload_file" in bridge_source
    assert "environment.download_file" in bridge_source
    assert "RemoteMCPBridge" in agent_source
    assert "container_reachable_url" not in agent_source
    assert 'env={"OPENAI_BASE_URL"' not in agent_source


def test_remote_bridge_omits_null_mcp_content_fields():
    bridge_source = (RUNNER.parent / "mcp_bridge.py").read_text()
    assert 'model_dump(mode="json", exclude_none=True)' in bridge_source
