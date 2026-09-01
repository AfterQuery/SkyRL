import json

from examples.train.toolathlon_harbor.toolathlon_context_manager import (
    ContextPolicy,
    ManagedContext,
)


def _manager(tmp_path, **overrides):
    values = {
        "max_context_tokens": 10_000,
        "max_output_tokens": 1_000,
        "safety_tokens": 0,
        "warning_ratio": 0.50,
        "compact_ratio": 0.60,
        "target_ratio": 0.40,
        "keep_recent_turns": 2,
        "keep_reasoning_turns": 1,
        "inline_tool_output_chars": 100,
        "preview_chars": 20,
    }
    values.update(overrides)
    return ManagedContext("do the task", tmp_path / "context", ContextPolicy(**values))


def _append_exchange(manager, index, size=600):
    manager.append(
        {
            "role": "assistant",
            "content": f"turn {index}",
            "reasoning_content": "r" * size,
            "tool_calls": [{"id": f"call-{index}", "function": {"name": "work", "arguments": "{}"}}],
        }
    )
    manager.append_tool_result(
        {"role": "tool", "tool_call_id": f"call-{index}", "name": "work", "content": "x" * size}
    )


def test_automatic_compaction_prunes_reasoning_and_atomic_exchanges(tmp_path):
    manager = _manager(tmp_path)
    for index in range(6):
        _append_exchange(manager, index)

    full_before = json.loads(json.dumps(manager.full_messages))
    manager.latest_prompt_tokens = 8_000
    manager.prepare_for_request()

    assert manager.full_messages == full_before
    active_assistants = [m for m in manager.active_messages if m["role"] == "assistant"]
    assert len(active_assistants) >= 2
    assert "reasoning_content" not in active_assistants[0]
    calls = {call["id"] for m in active_assistants for call in m.get("tool_calls", [])}
    results = {m["tool_call_id"] for m in manager.active_messages if m["role"] == "tool"}
    assert calls == results
    assert any(event["type"] == "automatic_compaction" for event in manager.events)


def test_oversized_tool_output_is_offloaded_and_searchable(tmp_path):
    manager = _manager(tmp_path)
    content = "prefix NEEDLE " + "z" * 200 + " NEEDLE " + "q" * 200
    compact = manager.append_tool_result(
        {"role": "tool", "tool_call_id": "call-1", "name": "work", "content": content}
    )

    artifact_id = compact["artifact_id"]
    assert "artifact_id" not in manager.active_messages[-1]
    assert len(compact["content"]) < len(content) + 200
    found = manager.call_local_tool(
        "tool_output_search",
        {"artifact_id": artifact_id, "pattern": "needle", "page_size": 1, "context_size": 20},
    )
    assert found["results"]
    assert found["total_matches"] == 2
    assert found["current_page"] == 1
    next_search = manager.call_local_tool(
        "tool_output_search_navigate",
        {"search_session_id": found["search_session_id"], "action": "next_page"},
    )
    assert next_search["current_page"] == 2
    viewed = manager.call_local_tool(
        "tool_output_view", {"artifact_id": artifact_id, "page_size": 100}
    )
    assert viewed["content"] == content[:100]
    next_view = manager.call_local_tool(
        "tool_output_view_navigate",
        {"view_session_id": viewed["view_session_id"], "action": "next_page"},
    )
    assert next_view["content"] == content[100:200]


def test_tool_output_search_validates_toolathlon_limits(tmp_path):
    manager = _manager(tmp_path)
    compact = manager.append_tool_result(
        {"role": "tool", "tool_call_id": "call-1", "name": "work", "content": "x" * 200}
    )
    artifact_id = compact["artifact_id"]

    for args in (
        {"artifact_id": artifact_id, "pattern": "["},
        {"artifact_id": artifact_id, "pattern": "x", "page_size": 51},
        {"artifact_id": artifact_id, "pattern": "x", "context_size": 0},
    ):
        try:
            manager.call_local_tool("tool_output_search", args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid search arguments accepted: {args}")


def test_manual_compaction_is_scheduled_until_exchange_finishes(tmp_path):
    manager = _manager(tmp_path)
    for index in range(5):
        _append_exchange(manager, index, size=20)
    before = list(manager.active_messages)

    result = manager.call_local_tool(
        "context_manage", {"method": "delete_first_turns", "value": 2}
    )
    assert result["status"] == "scheduled"
    assert manager.active_messages == before
    manager.apply_pending()
    assert len([m for m in manager.active_messages if m["role"] == "assistant"]) == 3


def test_emergency_reset_is_bounded_and_preserves_durable_history(tmp_path):
    manager = _manager(tmp_path, max_resets=2)
    _append_exchange(manager, 1, size=20)
    full_before = list(manager.full_messages)

    assert manager.emergency_reset() is True
    assert manager.emergency_reset() is True
    assert manager.emergency_reset() is False
    assert manager.full_messages[: len(full_before)] == full_before
    assert len(manager.full_messages) == len(full_before) + 2
    assert manager.active_messages[0] == {"role": "user", "content": "do the task"}
    assert "Context reset" in manager.active_messages[1]["content"]


def test_invalid_context_policy_is_rejected(tmp_path):
    policy = ContextPolicy(max_context_tokens=1_000, max_output_tokens=900, safety_tokens=100)
    try:
        ManagedContext("task", tmp_path / "context", policy)
    except ValueError as exc:
        assert "must exceed" in str(exc)
    else:
        raise AssertionError("invalid context budget was accepted")
