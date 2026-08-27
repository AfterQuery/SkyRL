"""Stateful stdio-MCP bridge that runs inside a Harbor task environment."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_MAX_MESSAGE_BYTES = 100_000_000


def _content_json(block: Any) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json")
    text = getattr(block, "text", None)
    return {"type": "text", "text": str(text if text is not None else block)}


async def serve(args: argparse.Namespace) -> None:
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    stop = asyncio.Event()
    shutdown_reply_sent = asyncio.Event()
    session_lock = asyncio.Lock()
    parameters = StdioServerParameters(
        command=args.mcp_command,
        args=args.mcp_arg,
        env=dict(os.environ),
    )

    async with (
        stdio_client(parameters) as streams,
        ClientSession(*streams) as session,
    ):
        await session.initialize()

        async def handle(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            is_shutdown = False
            try:
                raw = await reader.readline()
                if not raw or len(raw) > _MAX_MESSAGE_BYTES:
                    raise ValueError("invalid or oversized bridge request")
                request_data = json.loads(raw)
                op = request_data.get("op")
                if op == "ping":
                    result: dict[str, Any] = {
                        "status": "ok",
                        "result": {"ready": True},
                    }
                elif op == "list_tools":
                    async with session_lock:
                        listed = await session.list_tools()
                    result = {
                        "status": "ok",
                        "result": {
                            "tools": [
                                {
                                    "name": tool.name,
                                    "description": tool.description or "",
                                    "inputSchema": tool.inputSchema
                                    or {"type": "object", "properties": {}},
                                }
                                for tool in listed.tools
                            ]
                        },
                    }
                elif op == "call_tool":
                    name = request_data.get("name")
                    arguments = request_data.get("arguments")
                    if not isinstance(name, str) or not isinstance(arguments, dict):
                        raise ValueError(
                            "call_tool requires string name and object arguments"
                        )
                    async with session_lock:
                        tool_result = await session.call_tool(name, arguments)
                    result = {
                        "status": "ok",
                        "result": {
                            "content": [
                                _content_json(block)
                                for block in tool_result.content or []
                            ],
                            "is_error": bool(tool_result.isError),
                        },
                    }
                elif op == "shutdown":
                    is_shutdown = True
                    stop.set()
                    result = {"status": "ok", "result": {"stopping": True}}
                else:
                    raise ValueError(f"unknown bridge operation: {op}")
            except Exception as exc:  # noqa: BLE001 - errors cross the RPC boundary
                result = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            writer.write((json.dumps(result, ensure_ascii=False) + "\n").encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            if is_shutdown:
                shutdown_reply_sent.set()

        server = await asyncio.start_unix_server(
            handle, path=socket_path, limit=_MAX_MESSAGE_BYTES
        )
        await stop.wait()
        server.close()
        await server.wait_closed()
        await shutdown_reply_sent.wait()
    # Closing the MCP contexts terminates the child and lets it flush state.
    socket_path.unlink(missing_ok=True)


async def request(args: argparse.Namespace) -> None:
    request_path = Path(args.request_file)
    response_path = Path(args.response_file)
    raw = request_path.read_bytes()
    if len(raw) > _MAX_MESSAGE_BYTES:
        raise ValueError("bridge request exceeds 100 MB")
    reader, writer = await asyncio.open_unix_connection(
        args.socket, limit=_MAX_MESSAGE_BYTES
    )
    writer.write(raw.rstrip(b"\n") + b"\n")
    await writer.drain()
    response = await reader.readline()
    writer.close()
    await writer.wait_closed()
    if not response or len(response) > _MAX_MESSAGE_BYTES:
        raise RuntimeError("bridge returned an invalid or oversized response")
    response_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = response_path.with_suffix(response_path.suffix + ".tmp")
    temporary.write_bytes(response)
    os.replace(temporary, response_path)
    request_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--socket", required=True)
    serve_parser.add_argument("--mcp-command", required=True)
    serve_parser.add_argument("--mcp-arg", action="append", default=[])
    request_parser = subparsers.add_parser("request")
    request_parser.add_argument("--socket", required=True)
    request_parser.add_argument("--request-file", required=True)
    request_parser.add_argument("--response-file", required=True)
    args = parser.parse_args()
    asyncio.run(serve(args) if args.mode == "serve" else request(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
