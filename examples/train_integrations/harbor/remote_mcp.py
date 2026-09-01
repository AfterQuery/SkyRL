"""Host-side client for a stateful stdio MCP server in a Harbor environment."""

from __future__ import annotations

import json
import shlex
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from harbor.environments.base import BaseEnvironment
from harbor.models.task.config import MCPServerConfig
from harbor.models.trial.paths import EnvironmentPaths


class RemoteMCPBridge:
    """Own the remote MCP bridge lifecycle and its file-backed RPC protocol."""

    _REMOTE_DIR = PurePosixPath("/opt/harbor-mcp-agent")
    _REMOTE_BRIDGE = _REMOTE_DIR / "bridge.py"
    _REMOTE_SOCKET = _REMOTE_DIR / "bridge.sock"
    _REMOTE_PID = _REMOTE_DIR / "bridge.pid"
    _REMOTE_RPC_DIR = _REMOTE_DIR / "rpc"
    _BRIDGE_LOG = EnvironmentPaths.agent_dir / "bridge.log"
    _MAX_RESPONSE_BYTES = 100_000_000

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = logs_dir
        self.tools: list[dict[str, Any]] = []
        self._started = False

    async def setup(self, environment: BaseEnvironment, server: MCPServerConfig) -> None:
        local_bridge = Path(__file__).with_name("mcp_bridge.py")
        directories = [
            self._REMOTE_DIR.as_posix(),
            self._REMOTE_RPC_DIR.as_posix(),
            EnvironmentPaths.agent_dir.as_posix(),
        ]
        result = await environment.exec(
            "mkdir -p " + " ".join(shlex.quote(path) for path in directories),
            user="root",
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout or "failed to create MCP bridge directory")
        await environment.upload_file(local_bridge, self._REMOTE_BRIDGE.as_posix())

        command = [
            "python3",
            self._REMOTE_BRIDGE.as_posix(),
            "serve",
            "--socket",
            self._REMOTE_SOCKET.as_posix(),
            "--mcp-command",
            server.command or "",
        ]
        for arg in server.args:
            command.append(f"--mcp-arg={arg}")
        shell_command = " ".join(shlex.quote(part) for part in command)
        start = await environment.exec(
            f"rm -f {shlex.quote(self._REMOTE_SOCKET.as_posix())}; "
            f"nohup {shell_command} > {shlex.quote(self._BRIDGE_LOG.as_posix())} 2>&1 "
            f"< /dev/null & echo $! > {shlex.quote(self._REMOTE_PID.as_posix())}",
            user="root",
            timeout_sec=30,
        )
        if start.return_code != 0:
            raise RuntimeError(start.stderr or start.stdout or "failed to start MCP bridge")
        self._started = True

        last_error: Exception | None = None
        for _ in range(30):
            try:
                response = await self.rpc(environment, {"op": "list_tools"}, timeout_sec=30)
                tools = response.get("tools")
                if not isinstance(tools, list):
                    raise TypeError("MCP bridge list_tools response omitted tools")
                self.tools = tools
                return
            except Exception as exc:  # noqa: BLE001 - readiness includes remote errors
                last_error = exc
                import asyncio

                await asyncio.sleep(1)
        raise RuntimeError(f"MCP bridge did not become ready: {last_error}")

    async def shutdown(self, environment: BaseEnvironment) -> None:
        if not self._started:
            return
        self._started = False
        await self.rpc(environment, {"op": "shutdown"}, timeout_sec=30)
        pid_file = shlex.quote(self._REMOTE_PID.as_posix())
        wait = await environment.exec(
            f"pid=$(cat {pid_file}); " 'while kill -0 "$pid" 2>/dev/null; do sleep 0.1; done',
            user="root",
            timeout_sec=30,
        )
        if wait.return_code != 0:
            raise RuntimeError(wait.stderr or wait.stdout or "MCP bridge did not stop cleanly")

    async def rpc(
        self,
        environment: BaseEnvironment,
        request: dict[str, Any],
        *,
        timeout_sec: int,
    ) -> dict[str, Any]:
        rpc_id = uuid4().hex
        local_request = self.logs_dir / f"bridge-request-{rpc_id}.json"
        local_response = self.logs_dir / f"bridge-response-{rpc_id}.json"
        remote_request = self._REMOTE_RPC_DIR / f"{rpc_id}.request.json"
        remote_response = self._REMOTE_RPC_DIR / f"{rpc_id}.response.json"
        local_request.write_text(json.dumps(request, ensure_ascii=False))
        try:
            await environment.upload_file(local_request, remote_request.as_posix())
            command = " ".join(
                shlex.quote(part)
                for part in [
                    "python3",
                    self._REMOTE_BRIDGE.as_posix(),
                    "request",
                    "--socket",
                    self._REMOTE_SOCKET.as_posix(),
                    "--request-file",
                    remote_request.as_posix(),
                    "--response-file",
                    remote_response.as_posix(),
                ]
            )
            result = await environment.exec(command, user="root", timeout_sec=timeout_sec)
            if result.return_code != 0:
                raise RuntimeError(result.stderr or result.stdout or "MCP bridge request failed")
            await environment.download_file(remote_response.as_posix(), local_response)
            if local_response.stat().st_size > self._MAX_RESPONSE_BYTES:
                raise RuntimeError("MCP bridge response exceeds 100 MB")
            envelope = json.loads(local_response.read_text())
            if envelope.get("status") != "ok":
                raise RuntimeError(str(envelope.get("error") or "MCP bridge returned an error"))
            payload = envelope.get("result")
            if not isinstance(payload, dict):
                raise TypeError("MCP bridge returned an invalid result")
            return payload
        finally:
            local_request.unlink(missing_ok=True)
            local_response.unlink(missing_ok=True)
