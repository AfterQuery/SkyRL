#!/usr/bin/env python3
"""Small OpenAI-compatible agent loop for stdio MCP servers.

This file is uploaded into a Harbor task container by ``HarborMCPAgent``.  It
intentionally depends only on packages already present in MCP task images
(``mcp`` and ``httpx``), and never exposes a shell to the model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def container_reachable_url(url: str) -> str:
    """Rewrite a loopback model URL to an address reachable from a container."""
    parsed = urlsplit(url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return url.rstrip("/")
    host = "host.docker.internal"
    try:
        socket.getaddrinfo(host, parsed.port or 80)
    except OSError:
        # Docker on Linux commonly exposes the bridge gateway but not the
        # Docker Desktop hostname.
        try:
            for line in Path("/proc/net/route").read_text().splitlines()[1:]:
                fields = line.split()
                if len(fields) >= 3 and fields[1] == "00000000":
                    raw = bytes.fromhex(fields[2])
                    host = socket.inet_ntoa(raw[::-1])
                    break
        except (OSError, ValueError):
            pass
    netloc = host if parsed.port is None else f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)).rstrip("/")


def openai_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool descriptions to the OpenAI chat-completions shape."""
    out = []
    for tool in tools:
        schema = getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            }
        )
    return out


def _tool_result_text(result: Any) -> str:
    blocks: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            blocks.append(text)
        elif hasattr(block, "model_dump"):
            blocks.append(json.dumps(block.model_dump(mode="json"), ensure_ascii=False))
        else:
            blocks.append(str(block))
    text = "\n".join(blocks)
    if getattr(result, "isError", False):
        return f"Error: {text}"
    return text


def _token_data(response: dict[str, Any]) -> tuple[list[int] | None, list[int] | None, list[float] | None]:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    prompt_ids = response.get("prompt_token_ids")
    completion_ids = choice.get("token_ids")
    if completion_ids is None:
        completion_ids = (choice.get("provider_specific_fields") or {}).get("token_ids")
    logprobs_content = (choice.get("logprobs") or {}).get("content") or []
    logprobs = [item["logprob"] for item in logprobs_content if "logprob" in item]
    return prompt_ids, completion_ids, logprobs or None


async def _completion(
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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = await client.post(f"{api_base}/chat/completions", headers=headers, json=payload)
            if response.status_code not in {408, 409, 429} and response.status_code < 500:
                response.raise_for_status()
                return response.json()
            last_error = RuntimeError(f"model API returned {response.status_code}: {response.text[:1000]}")
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
        if attempt < 3:
            await asyncio.sleep(2**attempt)
    raise RuntimeError(f"model API failed after retries: {last_error}")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    instruction = Path(args.instruction_file).read_text()
    messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
    calls_log: list[dict[str, Any]] = []
    prompt_ids_per_turn: list[list[int]] = []
    completion_ids_per_turn: list[list[int]] = []
    logprobs_per_turn: list[list[float]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    started = time.monotonic()
    stop_reason = "complete"
    error: str | None = None
    tools: list[dict[str, Any]] = []

    server = StdioServerParameters(command=args.mcp_command, args=args.mcp_arg, env=dict(os.environ))
    try:
        async with stdio_client(server) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                listed = await session.list_tools()
                tools = openai_tools(listed.tools)
                tool_names = {tool["function"]["name"] for tool in tools}
                async with httpx.AsyncClient(timeout=args.request_timeout) as client:
                    for turn in range(args.max_turns):
                        if time.monotonic() - started >= args.deadline:
                            stop_reason = "deadline"
                            break
                        response = await _completion(
                            client,
                            api_base=container_reachable_url(args.api_base),
                            api_key=args.api_key,
                            model=args.model,
                            messages=messages,
                            tools=tools,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            collect_rollout_details=args.collect_rollout_details,
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
                        usage["completion_tokens"] += int(turn_usage.get("completion_tokens") or 0)
                        details = turn_usage.get("prompt_tokens_details") or {}
                        usage["cached_tokens"] += int(details.get("cached_tokens") or 0)

                        if args.collect_rollout_details:
                            prompt_ids, completion_ids, logprobs = _token_data(response)
                            if args.strict_rollout_details and (
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
                            break

                        if not tool_calls:
                            stop_reason = "complete"
                            break

                        for call in tool_calls:
                            call_id = call.get("id") or f"call_{len(calls_log)}"
                            fn = call.get("function") or {}
                            name = fn.get("name") or ""
                            raw_args = fn.get("arguments") or "{}"
                            call_record: dict[str, Any] = {"id": call_id, "name": name, "arguments": raw_args}
                            try:
                                parsed_args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
                                if not isinstance(parsed_args, dict):
                                    raise ValueError("tool arguments must decode to an object")
                                if name not in tool_names:
                                    raise ValueError(f"unknown tool: {name}")
                                result = await session.call_tool(name, parsed_args)
                                result_text = _tool_result_text(result)
                                call_record["is_error"] = bool(getattr(result, "isError", False))
                            except Exception as exc:  # tool errors belong in the conversation
                                result_text = f"Error: {type(exc).__name__}: {exc}"
                                call_record["is_error"] = True
                            call_record["result"] = result_text
                            calls_log.append(call_record)
                            messages.append(
                                {"role": "tool", "tool_call_id": call_id, "name": name, "content": result_text}
                            )
                    else:
                        stop_reason = "max_turns"
    except Exception as exc:
        stop_reason = "error"
        error = f"{type(exc).__name__}: {exc}"

    trajectory: dict[str, Any] = {
        "schema_version": 1,
        "model": args.model,
        "messages": messages,
        "tools": tools,
        "tool_calls": calls_log,
        "usage": usage,
        "n_turns": sum(1 for m in messages if m.get("role") == "assistant"),
        "stop_reason": stop_reason,
        "elapsed_seconds": time.monotonic() - started,
    }
    if error:
        trajectory["error"] = error
    if prompt_ids_per_turn or completion_ids_per_turn or logprobs_per_turn:
        trajectory["rollout_details"] = {
            "prompt_token_ids": prompt_ids_per_turn,
            "completion_token_ids": completion_ids_per_turn,
            "logprobs": logprobs_per_turn,
        }
    return trajectory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--trajectory-file", required=True)
    parser.add_argument("--mcp-command", required=True)
    parser.add_argument("--mcp-arg", action="append", default=[])
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--deadline", type=float, default=7000)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--collect-rollout-details", action="store_true")
    parser.add_argument("--strict-rollout-details", action="store_true")
    args = parser.parse_args()
    if not args.api_base:
        parser.error("--api-base or OPENAI_BASE_URL is required")

    trajectory = asyncio.run(run(args))
    path = Path(args.trajectory_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False))
    os.replace(tmp, path)
    if trajectory.get("error"):
        print(trajectory["error"], flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
