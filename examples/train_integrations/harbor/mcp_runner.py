"""Host-side OpenAI-compatible loop for Harbor tasks exposing MCP tools."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
Checkpoint = Callable[[dict[str, Any]], Awaitable[None]]


def openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert bridge MCP tool descriptions to OpenAI function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": tool.get("inputSchema")
                or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


def tool_result_text(result: dict[str, Any]) -> str:
    blocks: list[str] = []
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("text") is not None:
            blocks.append(str(block["text"]))
        else:
            blocks.append(json.dumps(block, ensure_ascii=False))
    text = "\n".join(blocks)
    return f"Error: {text}" if result.get("is_error") else text


def token_data(
    response: dict[str, Any],
) -> tuple[list[int] | None, list[int] | None, list[float] | None]:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    prompt_ids = response.get("prompt_token_ids")
    completion_ids = choice.get("token_ids")
    if completion_ids is None:
        completion_ids = (choice.get("provider_specific_fields") or {}).get("token_ids")
    logprobs_content = (choice.get("logprobs") or {}).get("content") or []
    logprobs = [item["logprob"] for item in logprobs_content if "logprob" in item]
    return prompt_ids, completion_ids, logprobs or None


async def completion(
    client: httpx.AsyncClient,
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    collect_rollout_details: bool,
    session_id: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if collect_rollout_details:
        payload["logprobs"] = True
        payload["return_token_ids"] = True
    if session_id:
        payload["session_id"] = session_id
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if session_id:
        headers["X-Session-ID"] = session_id

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = await client.post(
                f"{api_base.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            if (
                response.status_code not in {408, 409, 429}
                and response.status_code < 500
            ):
                response.raise_for_status()
                return response.json()
            last_error = RuntimeError(
                f"model API returned {response.status_code}: {response.text[:1000]}"
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
        if attempt < 3:
            await asyncio.sleep(2**attempt)
    raise RuntimeError(f"model API failed after retries: {last_error}")


def _trajectory(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    calls_log: list[dict[str, Any]],
    usage: dict[str, int],
    started: float,
    stop_reason: str,
    prompt_ids: list[list[int]],
    completion_ids: list[list[int]],
    logprobs: list[list[float]],
    error: str | None = None,
) -> dict[str, Any]:
    trajectory: dict[str, Any] = {
        "schema_version": 2,
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_calls": calls_log,
        "usage": usage,
        "n_turns": sum(1 for message in messages if message.get("role") == "assistant"),
        "stop_reason": stop_reason,
        "elapsed_seconds": time.monotonic() - started,
    }
    if error:
        trajectory["error"] = error
    if prompt_ids or completion_ids or logprobs:
        trajectory["rollout_details"] = {
            "prompt_token_ids": prompt_ids,
            "completion_token_ids": completion_ids,
            "logprobs": logprobs,
        }
    return trajectory


async def run_loop(
    *,
    instruction: str,
    bridge_tools: list[dict[str, Any]],
    call_tool: ToolCaller,
    checkpoint: Checkpoint,
    api_base: str,
    api_key: str,
    model: str,
    max_turns: int,
    deadline_sec: float,
    request_timeout_sec: float,
    max_tokens: int,
    temperature: float,
    collect_rollout_details: bool,
    strict_rollout_details: bool,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run the model locally while dispatching tool actions through a bridge."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
    calls_log: list[dict[str, Any]] = []
    prompt_ids_per_turn: list[list[int]] = []
    completion_ids_per_turn: list[list[int]] = []
    logprobs_per_turn: list[list[float]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    tools = openai_tools(bridge_tools)
    tool_names = {tool["function"]["name"] for tool in tools}
    started = time.monotonic()
    stop_reason = "running"
    error: str | None = None

    def snapshot() -> dict[str, Any]:
        return _trajectory(
            model=model,
            messages=messages,
            tools=tools,
            calls_log=calls_log,
            usage=usage,
            started=started,
            stop_reason=stop_reason,
            prompt_ids=prompt_ids_per_turn,
            completion_ids=completion_ids_per_turn,
            logprobs=logprobs_per_turn,
            error=error,
        )

    try:
        async with httpx.AsyncClient(timeout=request_timeout_sec) as client:
            for _turn in range(max_turns):
                if time.monotonic() - started >= deadline_sec:
                    stop_reason = "deadline"
                    break
                response = await completion(
                    client,
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    collect_rollout_details=collect_rollout_details,
                    session_id=session_id,
                )
                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                assistant: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("content"),
                }
                if message.get("reasoning_content") is not None:
                    assistant["reasoning_content"] = message["reasoning_content"]
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    assistant["tool_calls"] = tool_calls
                messages.append(assistant)

                turn_usage = response.get("usage") or {}
                usage["prompt_tokens"] += int(turn_usage.get("prompt_tokens") or 0)
                usage["completion_tokens"] += int(
                    turn_usage.get("completion_tokens") or 0
                )
                details = turn_usage.get("prompt_tokens_details") or {}
                usage["cached_tokens"] += int(details.get("cached_tokens") or 0)

                if collect_rollout_details:
                    prompt_ids, completion_ids, logprobs = token_data(response)
                    if strict_rollout_details and (
                        not isinstance(prompt_ids, list)
                        or not isinstance(completion_ids, list)
                        or not isinstance(logprobs, list)
                        or len(completion_ids) != len(logprobs)
                    ):
                        raise RuntimeError(
                            "strict rollout collection requires aligned prompt_token_ids, "
                            "completion token_ids, and per-token logprobs"
                        )
                    if isinstance(prompt_ids, list):
                        prompt_ids_per_turn.append(prompt_ids)
                    if isinstance(completion_ids, list):
                        completion_ids_per_turn.append(completion_ids)
                    if isinstance(logprobs, list):
                        logprobs_per_turn.append(logprobs)

                if choice.get("finish_reason") == "length":
                    stop_reason = "output_length"
                    await checkpoint(snapshot())
                    break
                if not tool_calls:
                    stop_reason = "complete"
                    await checkpoint(snapshot())
                    break

                for call in tool_calls:
                    call_id = call.get("id") or f"call_{len(calls_log)}"
                    fn = call.get("function") or {}
                    name = fn.get("name") or ""
                    raw_args = fn.get("arguments") or "{}"
                    record: dict[str, Any] = {
                        "id": call_id,
                        "name": name,
                        "arguments": raw_args,
                    }
                    try:
                        parsed_args = (
                            raw_args
                            if isinstance(raw_args, dict)
                            else json.loads(raw_args)
                        )
                        if not isinstance(parsed_args, dict):
                            raise TypeError("tool arguments must decode to an object")
                        if name not in tool_names:
                            raise ValueError(f"unknown tool: {name}")
                        result = await call_tool(name, parsed_args)
                        result_text = tool_result_text(result)
                        record["is_error"] = bool(result.get("is_error"))
                    except Exception as exc:  # noqa: BLE001 - tool errors are model-visible
                        result_text = f"Error: {type(exc).__name__}: {exc}"
                        record["is_error"] = True
                    record["result"] = result_text
                    calls_log.append(record)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": result_text,
                        }
                    )
                await checkpoint(snapshot())
            else:
                stop_reason = "max_turns"
    except asyncio.CancelledError:
        stop_reason = "cancelled"
        error = "CancelledError: agent loop cancelled"
        await checkpoint(snapshot())
        raise
    except Exception as exc:  # noqa: BLE001 - fatal errors are persisted in the trajectory
        stop_reason = "error"
        error = f"{type(exc).__name__}: {exc}"

    final = snapshot()
    await checkpoint(final)
    return final
