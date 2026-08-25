"""Generic OpenAI-compatible agent for Harbor tasks exposing stdio MCP tools."""

from __future__ import annotations

import json
import shlex
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths


class HarborMCPAgent(BaseAgent):
    """Drive configured stdio MCP tools without exposing a shell to the model."""

    SUPPORTS_ATIF = False
    _REMOTE_DIR = PurePosixPath("/opt/harbor-mcp-agent")
    _REMOTE_RUNNER = _REMOTE_DIR / "runner.py"
    _REMOTE_INSTRUCTION = _REMOTE_DIR / "instruction.md"
    _TRAJECTORY = EnvironmentPaths.agent_dir / "trajectory.json"
    _MAX_TRAJECTORY_BYTES = 100_000_000

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
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
        super().__init__(*args, **kwargs)
        self.extra_env = extra_env or {}
        self.api_base = api_base or self.extra_env.get("OPENAI_BASE_URL")
        self.api_key = api_key or self.extra_env.get("OPENAI_API_KEY", "dummy")
        self.max_turns = max_turns
        self.deadline_sec = deadline_sec
        self.request_timeout_sec = request_timeout_sec
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.collect_rollout_details = collect_rollout_details
        self.strict_rollout_details = strict_rollout_details
        if not self.api_base:
            raise ValueError("HarborMCPAgent requires api_base or OPENAI_BASE_URL")
        if not self.model_name:
            raise ValueError("HarborMCPAgent requires agent.model_name")
        if len(self.mcp_servers) != 1 or self.mcp_servers[0].transport != "stdio":
            raise ValueError("HarborMCPAgent currently requires exactly one stdio MCP server")

    @staticmethod
    def name() -> str:
        return "harbor-mcp"

    def version(self) -> str | None:
        return "1"

    def _request_model(self) -> str:
        assert self.model_name is not None
        return self.model_name.split("/", 1)[-1]

    async def setup(self, environment: BaseEnvironment) -> None:
        local_runner = Path(__file__).with_name("mcp_runner.py")
        result = await environment.exec(
            f"mkdir -p {shlex.quote(self._REMOTE_DIR.as_posix())} {shlex.quote(EnvironmentPaths.agent_dir.as_posix())}",
            user="root",
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout or "failed to create MCP agent directory")
        await environment.upload_file(local_runner, self._REMOTE_RUNNER.as_posix())

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        local_instruction = self.logs_dir / "instruction.md"
        local_instruction.write_text(instruction)
        await environment.upload_file(local_instruction, self._REMOTE_INSTRUCTION.as_posix())

        server = self.mcp_servers[0]
        command = [
            "python3",
            self._REMOTE_RUNNER.as_posix(),
            "--instruction-file",
            self._REMOTE_INSTRUCTION.as_posix(),
            "--trajectory-file",
            self._TRAJECTORY.as_posix(),
            "--mcp-command",
            server.command or "",
            "--model",
            self._request_model(),
            "--max-turns",
            str(self.max_turns),
            "--deadline",
            str(self.deadline_sec),
            "--request-timeout",
            str(self.request_timeout_sec),
            "--max-tokens",
            str(self.max_tokens),
            "--temperature",
            str(self.temperature),
        ]
        for arg in server.args:
            command.append(f"--mcp-arg={arg}")
        if self.collect_rollout_details:
            command.append("--collect-rollout-details")
        if self.strict_rollout_details:
            command.append("--strict-rollout-details")
        shell_command = " ".join(shlex.quote(part) for part in command)
        result = await environment.exec(
            f"{shell_command} > {shlex.quote((EnvironmentPaths.agent_dir / 'runner.log').as_posix())} 2>&1",
            env={"OPENAI_BASE_URL": self.api_base or "", "OPENAI_API_KEY": self.api_key},
            timeout_sec=int(self.deadline_sec + 60),
        )
        await self._apply_trajectory(environment, context)
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout or "MCP agent runner failed")

    async def _apply_trajectory(self, environment: BaseEnvironment, context: AgentContext) -> None:
        local = self.logs_dir / "trajectory.json"
        await environment.download_file(self._TRAJECTORY.as_posix(), local)
        if local.stat().st_size > self._MAX_TRAJECTORY_BYTES:
            raise RuntimeError("MCP trajectory exceeds 100 MB")
        trajectory = json.loads(local.read_text())
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
