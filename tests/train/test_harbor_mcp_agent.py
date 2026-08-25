import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


RUNNER = Path(__file__).parents[2] / "examples/train_integrations/harbor/mcp_runner.py"
SPEC = importlib.util.spec_from_file_location("harbor_mcp_runner", RUNNER)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_openai_tools_preserves_mcp_schema():
    tools = [
        SimpleNamespace(
            name="calendar__create",
            description="Create an event",
            inputSchema={"type": "object", "properties": {"title": {"type": "string"}}},
        )
    ]
    assert runner.openai_tools(tools) == [
        {
            "type": "function",
            "function": {
                "name": "calendar__create",
                "description": "Create an event",
                "parameters": {"type": "object", "properties": {"title": {"type": "string"}}},
            },
        }
    ]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.openai.com/v1", "https://api.openai.com/v1"),
        ("http://model.internal:8000/v1/", "http://model.internal:8000/v1"),
    ],
)
def test_non_loopback_urls_are_unchanged(url, expected):
    assert runner.container_reachable_url(url) == expected


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
    assert runner._token_data(response) == ([1, 2], [3, 4], [-0.1, -0.2])


def test_tool_error_is_visible_to_model():
    result = SimpleNamespace(content=[SimpleNamespace(text="bad arguments")], isError=True)
    assert runner._tool_result_text(result) == "Error: bad arguments"
