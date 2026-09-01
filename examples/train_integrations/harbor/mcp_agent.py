"""Generic host-side OpenAI agent for Harbor tasks exposing stdio MCP tools."""

from __future__ import annotations

import asyncio
import json
import os
from importlib import import_module
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .mcp_runner import ContextManagerFactory, run_loop
from .remote_mcp import RemoteMCPBridge


class ContextLengthExceededError(RuntimeError):
    """Harbor-visible terminal context overflow with usable rollout details."""


def _load_context_manager_factory(import_path: str | None) -> ContextManagerFactory | None:
    if import_path is None:
        return None
    module_name, separator, attribute = import_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("context_manager_factory must use module.path:attribute syntax")
    factory = getattr(import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"context manager factory is not callable: {import_path}")
    return factory


class HarborMCPAgent(BaseAgent):
    """Run the model locally and dispatch configured stdio MCP tools remotely."""

    SUPPORTS_ATIF = False
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
        context_manager_factory: str | None = None,
        max_context_tokens: int | None = None,
        context_safety_tokens: int = 2048,
        context_warning_ratio: float = 0.75,
        context_compact_ratio: float = 0.85,
        context_target_ratio: float = 0.70,
        context_keep_recent_turns: int = 4,
        context_keep_reasoning_turns: int = 1,
        inline_tool_output_chars: int = 12_000,
        max_context_resets: int = 2,
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
        self.context_manager_factory = _load_context_manager_factory(context_manager_factory)
        self.max_context_tokens = max_context_tokens
        self.context_safety_tokens = context_safety_tokens
        self.context_warning_ratio = context_warning_ratio
        self.context_compact_ratio = context_compact_ratio
        self.context_target_ratio = context_target_ratio
        self.context_keep_recent_turns = context_keep_recent_turns
        self.context_keep_reasoning_turns = context_keep_reasoning_turns
        self.inline_tool_output_chars = inline_tool_output_chars
        self.max_context_resets = max_context_resets
        self._bridge_tools: list[dict[str, Any]] = []
        self._remote_bridge = RemoteMCPBridge(self.logs_dir)
        if not self.api_base:
            raise ValueError("HarborMCPAgent requires api_base or OPENAI_BASE_URL")
        if not self.model_name:
            raise ValueError("HarborMCPAgent requires agent.model_name")
        if len(self.mcp_servers) != 1 or self.mcp_servers[0].transport != "stdio":
            raise ValueError("HarborMCPAgent currently requires exactly one stdio MCP server")
        if not self.mcp_servers[0].command:
            raise ValueError("HarborMCPAgent requires an MCP server command")

    @staticmethod
    def name() -> str:
        return "harbor-mcp"

    def version(self) -> str | None:
        return "3"

    def _request_model(self) -> str:
        assert self.model_name is not None
        return self.model_name.split("/", 1)[-1]

    async def setup(self, environment: BaseEnvironment) -> None:
        await self._remote_bridge.setup(environment, self.mcp_servers[0])
        self._bridge_tools = self._remote_bridge.tools

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
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
                context_manager_factory=self.context_manager_factory,
                artifact_dir=self.logs_dir / "context",
                max_context_tokens=self.max_context_tokens,
                context_safety_tokens=self.context_safety_tokens,
                context_warning_ratio=self.context_warning_ratio,
                context_compact_ratio=self.context_compact_ratio,
                context_target_ratio=self.context_target_ratio,
                context_keep_recent_turns=self.context_keep_recent_turns,
                context_keep_reasoning_turns=self.context_keep_reasoning_turns,
                inline_tool_output_chars=self.inline_tool_output_chars,
                max_context_resets=self.max_context_resets,
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

        if trajectory and trajectory.get("stop_reason") == "context_length":
            raise ContextLengthExceededError(str(trajectory.get("error") or "context recovery exhausted"))
        if trajectory and trajectory.get("error"):
            raise RuntimeError(str(trajectory["error"]))

    async def _shutdown_bridge(self, environment: BaseEnvironment) -> None:
        await self._remote_bridge.shutdown(environment)

    async def _bridge_rpc(
        self,
        environment: BaseEnvironment,
        request: dict[str, Any],
        *,
        timeout_sec: int,
    ) -> dict[str, Any]:
        return await self._remote_bridge.rpc(environment, request, timeout_sec=timeout_sec)

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
            "context_events": trajectory.get("context_events") or [],
            "active_messages": trajectory.get("active_messages") or [],
            "context_artifact_dir": trajectory.get("context_artifact_dir"),
        }
        details = trajectory.get("rollout_details")
        if details:
            context.rollout_details = [details]
