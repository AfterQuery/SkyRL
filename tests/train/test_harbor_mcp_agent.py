import importlib.util
import json
from pathlib import Path

import httpx
import pytest

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


def test_agent_uses_environment_rpc_without_container_model_credentials():
    agent_source = (RUNNER.parent / "mcp_agent.py").read_text()
    assert "environment.upload_file" in agent_source
    assert "environment.download_file" in agent_source
    assert "container_reachable_url" not in agent_source
    assert 'env={"OPENAI_BASE_URL"' not in agent_source
