"""Generic host-side OpenAI agent for Harbor tasks exposing stdio MCP tools."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from .mcp_runner import run_loop


class HarborMCPAgent(BaseAgent):
    """Run the model locally and dispatch configured stdio MCP tools remotely."""

    SUPPORTS_ATIF = False
    _REMOTE_DIR = PurePosixPath("/opt/harbor-mcp-agent")
    _REMOTE_BRIDGE = _REMOTE_DIR / "bridge.py"
    _REMOTE_SOCKET = _REMOTE_DIR / "bridge.sock"
    _REMOTE_PID = _REMOTE_DIR / "bridge.pid"
    _REMOTE_RPC_DIR = _REMOTE_DIR / "rpc"
    _BRIDGE_LOG = EnvironmentPaths.agent_dir / "bridge.log"
    _MAX_TRAJECTORY_BYTES = 100_000_000

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        session_id: str | None = None,
        max_turns: int = 64,
        deadline_sec: float = 7000,
        request_timeout_sec: float = 900,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        collect_rollout_details: bool = False,
        strict_rollout_details: bool = False,
        extra_env: dict[str, str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, extra_env=extra_env, **kwargs)
        self.api_base = api_base or self.extra_env.get("OPENAI_BASE_URL")
        self.api_key = api_key or self.extra_env.get("OPENAI_API_KEY", "dummy")
        self.session_id = session_id
        self.max_turns = max_turns
        self.deadline_sec = deadline_sec
        self.request_timeout_sec = request_timeout_sec
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.collect_rollout_details = collect_rollout_details
        self.strict_rollout_details = strict_rollout_details
        self._bridge_tools: list[dict[str, Any]] = []
        if not self.api_base:
            raise ValueError("HarborMCPAgent requires api_base or OPENAI_BASE_URL")
        if not self.model_name:
            raise ValueError("HarborMCPAgent requires agent.model_name")
        if len(self.mcp_servers) != 1 or self.mcp_servers[0].transport != "stdio":
            raise ValueError(
                "HarborMCPAgent currently requires exactly one stdio MCP server"
            )
        if not self.mcp_servers[0].command:
            raise ValueError("HarborMCPAgent requires an MCP server command")

    @staticmethod
    def name() -> str:
        return "harbor-mcp"

    def version(self) -> str | None:
        return "2"

    def _request_model(self) -> str:
        assert self.model_name is not None
        return self.model_name.split("/", 1)[-1]

    async def setup(self, environment: BaseEnvironment) -> None:
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
            raise RuntimeError(
                result.stderr
                or result.stdout
                or "failed to create MCP bridge directory"
            )
        await environment.upload_file(local_bridge, self._REMOTE_BRIDGE.as_posix())

        server = self.mcp_servers[0]
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
            raise RuntimeError(
                start.stderr or start.stdout or "failed to start MCP bridge"
            )

        last_error: Exception | None = None
        for _ in range(30):
            try:
                response = await self._bridge_rpc(
                    environment, {"op": "list_tools"}, timeout_sec=30
                )
                tools = response.get("tools")
                if not isinstance(tools, list):
                    raise TypeError("MCP bridge list_tools response omitted tools")
                self._bridge_tools = tools
                return
            except Exception as exc:  # noqa: BLE001 - readiness retries include provider errors
                last_error = exc
                await asyncio.sleep(1)
        raise RuntimeError(f"MCP bridge did not become ready: {last_error}")

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        trajectory: dict[str, Any] | None = None

        async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return await self._bridge_rpc(
                environment,
                {"op": "call_tool", "name": name, "arguments": arguments},
                timeout_sec=int(self.request_timeout_sec),
            )

        async def checkpoint(value: dict[str, Any]) -> None:
            self._write_trajectory(value)
            self._apply_trajectory(value, context)

        try:
            trajectory = await run_loop(
                instruction=instruction,
                bridge_tools=self._bridge_tools,
                call_tool=call_tool,
                checkpoint=checkpoint,
                api_base=self.api_base or "",
                api_key=self.api_key,
                model=self._request_model(),
                max_turns=self.max_turns,
                deadline_sec=self.deadline_sec,
                request_timeout_sec=self.request_timeout_sec,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                collect_rollout_details=self.collect_rollout_details,
                strict_rollout_details=self.strict_rollout_details,
                session_id=self.session_id,
            )
        finally:
            shutdown = asyncio.create_task(self._shutdown_bridge(environment))
            try:
                await asyncio.shield(shutdown)
            except Exception as exc:  # noqa: BLE001 - environment teardown is authoritative cleanup
                self.logger.warning(
                    "Bridge shutdown failed; environment teardown will clean it up: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        if trajectory and trajectory.get("error"):
            raise RuntimeError(str(trajectory["error"]))

    async def _shutdown_bridge(self, environment: BaseEnvironment) -> None:
        await self._bridge_rpc(environment, {"op": "shutdown"}, timeout_sec=30)
        pid_file = shlex.quote(self._REMOTE_PID.as_posix())
        wait = await environment.exec(
            f"pid=$(cat {pid_file}); "
            'while kill -0 "$pid" 2>/dev/null; do sleep 0.1; done',
            user="root",
            timeout_sec=30,
        )
        if wait.return_code != 0:
            raise RuntimeError(
                wait.stderr or wait.stdout or "MCP bridge did not stop cleanly"
            )

    async def _bridge_rpc(
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
            result = await environment.exec(
                command, user="root", timeout_sec=timeout_sec
            )
            if result.return_code != 0:
                raise RuntimeError(
                    result.stderr or result.stdout or "MCP bridge request failed"
                )
            await environment.download_file(remote_response.as_posix(), local_response)
            if local_response.stat().st_size > self._MAX_TRAJECTORY_BYTES:
                raise RuntimeError("MCP bridge response exceeds 100 MB")
            envelope = json.loads(local_response.read_text())
            if envelope.get("status") != "ok":
                raise RuntimeError(
                    str(envelope.get("error") or "MCP bridge returned an error")
                )
            payload = envelope.get("result")
            if not isinstance(payload, dict):
                raise TypeError("MCP bridge returned an invalid result")
            return payload
        finally:
            local_request.unlink(missing_ok=True)
            local_response.unlink(missing_ok=True)

    def _write_trajectory(self, trajectory: dict[str, Any]) -> None:
        local = self.logs_dir / "trajectory.json"
        encoded = json.dumps(trajectory, indent=2, ensure_ascii=False)
        if len(encoded.encode()) > self._MAX_TRAJECTORY_BYTES:
            raise RuntimeError("MCP trajectory exceeds 100 MB")
        temporary = local.with_suffix(".tmp")
        temporary.write_text(encoded)
        os.replace(temporary, local)

    @staticmethod
    def _apply_trajectory(trajectory: dict[str, Any], context: AgentContext) -> None:
        usage = trajectory.get("usage") or {}
        context.n_input_tokens = int(usage.get("prompt_tokens") or 0)
        context.n_output_tokens = int(usage.get("completion_tokens") or 0)
        context.n_cache_tokens = int(usage.get("cached_tokens") or 0)
        context.metadata = {
            "n_episodes": int(trajectory.get("n_turns") or 0),
            "n_tool_calls": len(trajectory.get("tool_calls") or []),
            "agent_stop_reason": trajectory.get("stop_reason", "error"),
            "all_messages": trajectory.get("messages") or [],
            "elapsed_seconds": trajectory.get("elapsed_seconds"),
        }
        details = trajectory.get("rollout_details")
        if details:
            context.rollout_details = [details]
