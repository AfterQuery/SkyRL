"""Generator that runs MCP-Atlas tasks against SkyRL's OpenAI-compatible inference endpoint.

The generator owns the agent loop (the benchmark's TypeScript harness is bypassed): each turn
it calls SkyRL's ``/v1/chat/completions`` with the task's tool schemas, executes the returned
tool calls against the long-running MCP-Atlas sandbox (``POST /call-tool``), and appends the
observations. Rewards come from the claim-coverage LLM judge (``judge.py``), the benchmark's
own metric. Conversations are re-tokenized into training data with tool observations
loss-masked.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiohttp
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm

from skyrl.backends.skyrl_train.inference_servers.base import ConversationType, InferenceEngineInterface
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import (
    get_response_ids_and_loss_mask_from_messages,
    get_rollout_metrics,
)

from .judge import ClaimCoverageJudge, JudgeError

# Retries for the whole rollout on unexpected errors (sandbox hiccups etc.). Judge failures
# and legitimate low-coverage answers are never retried.
MAX_NUM_RETRIES_PER_TASK = 2

# Retries per chat-completions call inside the agent loop (matches the official harness).
MAX_LLM_RETRIES = 3

# Sampling params forwarded from the trainer to /chat/completions.
_ALLOWED_SAMPLING_KEYS = ("temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "stop")


@dataclass
class AtlasTrajectoryOutput:
    """One trajectory's conversation and reward."""

    trajectory_id: TrajectoryID
    # Full conversation excluding the initial prompt messages; None for failed rollouts.
    messages: Optional[List[Dict[str, Any]]] = None
    prompt_messages: List[Dict[str, Any]] = field(default_factory=list)
    # OpenAI function-tool schemas offered to the model, used for prompt re-tokenization.
    tools: List[Dict[str, Any]] = field(default_factory=list)
    reward: float = 0.0
    # One of: "complete", "error". "error" trajectories are loss-masked.
    stop_reason: str = "complete"
    num_turns: int = 0
    e2e_time: Optional[float] = None


class MCPAtlasGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg,
        atlas_cfg: Dict[str, Any],
        inference_engine_client: InferenceEngineInterface,
        tokenizer,
        max_seq_len: int,
    ):
        """
        Args:
            generator_cfg: Generator configuration (``cfg.generator``).
            atlas_cfg: MCP-Atlas configuration (see ``mcp_atlas_config.yaml``).
            inference_engine_client: Client for the inference engines; provides the HTTP endpoint.
            tokenizer: Tokenizer used to re-tokenize conversations into training data.
            max_seq_len: Maximum total sequence length (prompt + response) for truncation.
        """
        self.generator_cfg = generator_cfg
        self.atlas_cfg = atlas_cfg
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.base_url = inference_engine_client.get_endpoint_url()

        self._served_model_name = generator_cfg.inference_engine.served_model_name
        assert self._served_model_name is not None, "generator.inference_engine.served_model_name must be set"

        self.sandbox_url = atlas_cfg["sandbox_url"].rstrip("/")
        self.dump_root = Path(atlas_cfg["dump_root"]).expanduser().resolve()
        self.dump_root.mkdir(parents=True, exist_ok=True)

        self.judge = ClaimCoverageJudge(atlas_cfg["judge"]) if atlas_cfg["judge"]["enabled"] else None
        if self.judge is None:
            logger.warning("MCP-Atlas judge disabled: all rewards will be 0.0 (debugging only).")

        # The sandbox is one shared, stateful container: cap concurrent rollouts, and serialize
        # repetitions of the same task since they'd race on the same filesystem paths, memory
        # graph, and git trees.
        self._semaphore = asyncio.Semaphore(int(atlas_cfg["max_concurrent_tasks"]))
        self._task_locks: Dict[str, asyncio.Lock] = {}

        # Tool schemas fetched from the sandbox once, on first use.
        self._all_tools: Optional[Dict[str, Dict[str, Any]]] = None
        self._tools_lock = asyncio.Lock()

        if getattr(generator_cfg, "step_wise_trajectories", False):
            raise ValueError(
                "MCPAtlasGenerator uses re-tokenization and does not support step-wise training. "
                "Set generator.step_wise_trajectories=false."
            )

    def _get_task_lock(self, task_id: str) -> asyncio.Lock:
        if task_id not in self._task_locks:
            self._task_locks[task_id] = asyncio.Lock()
        return self._task_locks[task_id]

    async def _get_all_tools(self, session: aiohttp.ClientSession) -> Dict[str, Dict[str, Any]]:
        """Fetch all tool schemas from the sandbox once and map them to OpenAI function format."""
        async with self._tools_lock:
            if self._all_tools is None:
                timeout = aiohttp.ClientTimeout(total=float(self.atlas_cfg["list_tools_timeout_seconds"]))
                async with session.post(f"{self.sandbox_url}/list-tools", timeout=timeout) as resp:
                    resp.raise_for_status()
                    tools = await resp.json()
                self._all_tools = {
                    t["name"]: {
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
                        },
                    }
                    for t in tools
                }
                logger.info(f"Fetched {len(self._all_tools)} tool schemas from the MCP-Atlas sandbox")
            return self._all_tools

    async def _call_tool(self, session: aiohttp.ClientSession, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute one tool call against the sandbox; errors come back as observation text."""
        timeout = aiohttp.ClientTimeout(total=float(self.atlas_cfg["tool_call_timeout_seconds"]))
        try:
            async with session.post(
                f"{self.sandbox_url}/call-tool",
                json={"tool_name": tool_name, "tool_args": tool_args},
                timeout=timeout,
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    return f"Error: {body[:1000]}"
        except asyncio.TimeoutError:
            return f"Error: tool call {tool_name} timed out"
        try:
            blocks = json.loads(body)
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        except (json.JSONDecodeError, AttributeError, TypeError):
            text = body
        cap = self.atlas_cfg.get("tool_output_cap")
        if cap and len(text) > int(cap):
            text = text[: int(cap)] + f"\n[... truncated to {cap} characters]"
        return text

    def _build_sampling_kwargs(self, sampling_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if sampling_params:
            kwargs = {k: v for k, v in sampling_params.items() if k in _ALLOWED_SAMPLING_KEYS and v is not None}
        kwargs.setdefault("max_tokens", self.generator_cfg.sampling_params.max_generate_length)
        return kwargs

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if env_extras is None:
            raise ValueError("`env_extras` is required in the input batch (task_id / enabled_tools / claims)")

        batch_metadata = input_batch.get("batch_metadata")
        if batch_metadata is not None:
            batch_tag = f"step_{batch_metadata.global_step}_{batch_metadata.training_phase}"
        else:
            batch_tag = f"batch_{uuid4().hex[:8]}"
        batch_dir = self.dump_root / batch_tag
        batch_dir.mkdir(parents=True, exist_ok=True)
        sampling_kwargs = self._build_sampling_kwargs(input_batch.get("sampling_params"))

        policy_client = AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="dummy",
            timeout=float(self.atlas_cfg["llm_timeout_seconds"]),
            max_retries=0,
        )
        all_outputs: List[AtlasTrajectoryOutput] = [None] * len(prompts)  # type: ignore[list-item]
        progress = tqdm(
            disable=disable_tqdm,
            total=len(prompts),
            desc="Generating MCP-Atlas Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async with aiohttp.ClientSession() as session:

            async def _worker(idx: int, prompt: ConversationType, extras: Dict[str, Any], tid: TrajectoryID):
                all_outputs[idx] = await self._run_one_trajectory(
                    session, policy_client, prompt, extras, tid, sampling_kwargs, batch_dir
                )
                progress.update(1)

            try:
                async with asyncio.TaskGroup() as tg:
                    for idx in range(len(prompts)):
                        tg.create_task(_worker(idx, prompts[idx], env_extras[idx], trajectory_ids[idx]))
            finally:
                progress.close()

        return self._build_generator_output(all_outputs)

    async def _run_one_trajectory(
        self,
        session: aiohttp.ClientSession,
        policy_client: AsyncOpenAI,
        prompt: ConversationType,
        extras: Dict[str, Any],
        trajectory_id: TrajectoryID,
        sampling_kwargs: Dict[str, Any],
        batch_dir: Path,
    ) -> AtlasTrajectoryOutput:
        start_time = time.monotonic()
        task_id = str(extras.get("task_id", trajectory_id.instance_id))
        enabled_tools = json.loads(extras["enabled_tools_json"])
        claims = json.loads(extras["gtfa_claims_json"])
        lock = self._get_task_lock(task_id)

        for attempt in range(MAX_NUM_RETRIES_PER_TASK):
            prefix = f"Trajectory {trajectory_id} ({task_id}) attempt {attempt + 1}/{MAX_NUM_RETRIES_PER_TASK}"
            try:
                async with lock:
                    async with self._semaphore:
                        output = await self._agent_loop(
                            session, policy_client, prompt, enabled_tools, trajectory_id, sampling_kwargs
                        )

                # Reward: claim coverage of the final assistant response (the benchmark metric).
                final_response = ""
                for msg in reversed(output.messages or []):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        final_response = msg["content"]
                        break
                if self.judge is not None:
                    output.reward = await self.judge.score(claims, final_response)

                output.e2e_time = time.monotonic() - start_time
                self._dump_trajectory(batch_dir, task_id, trajectory_id, output, final_response)
                return output
            except JudgeError as e:
                logger.warning(f"{prefix}: judge failed ({e}); masking trajectory.")
                break
            except Exception as e:  # noqa: BLE001 - retry the rollout on any harness/sandbox error
                logger.warning(f"{prefix} failed: {e}")

        return AtlasTrajectoryOutput(
            trajectory_id=trajectory_id, stop_reason="error", e2e_time=time.monotonic() - start_time
        )

    async def _agent_loop(
        self,
        session: aiohttp.ClientSession,
        policy_client: AsyncOpenAI,
        prompt: ConversationType,
        enabled_tools: List[str],
        trajectory_id: TrajectoryID,
        sampling_kwargs: Dict[str, Any],
    ) -> AtlasTrajectoryOutput:
        """Run the multi-turn tool-use loop for one task (mirrors the official harness)."""
        all_tools = await self._get_all_tools(session)
        # Intersection, like the official harness: tools from disabled/gated servers are dropped.
        tools = [all_tools[name] for name in enabled_tools if name in all_tools]
        missing = [name for name in enabled_tools if name not in all_tools]
        if missing:
            logger.warning(f"Trajectory {trajectory_id}: {len(missing)} enabled tools unavailable: {missing[:5]}")

        prompt_messages = list(prompt)
        system_prompt = self.atlas_cfg.get("system_prompt")
        if system_prompt and not any(m["role"] == "system" for m in prompt_messages):
            prompt_messages = [{"role": "system", "content": system_prompt}] + prompt_messages

        messages: List[Dict[str, Any]] = []
        max_turns = int(self.atlas_cfg["max_turns"])
        max_tool_calls = int(self.atlas_cfg["max_tool_calls"])
        total_tool_calls = 0
        num_turns = 0
        stop_reason = "complete"

        for _ in range(max_turns):
            completion = None
            for llm_attempt in range(MAX_LLM_RETRIES):
                try:
                    completion = await policy_client.chat.completions.create(
                        model=self._served_model_name,
                        messages=prompt_messages + messages,
                        tools=tools or None,
                        **sampling_kwargs,
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    if llm_attempt == MAX_LLM_RETRIES - 1:
                        raise
                    logger.debug(f"Trajectory {trajectory_id}: LLM call failed ({e}), retrying")
                    await asyncio.sleep(2**llm_attempt)

            assistant = completion.choices[0].message
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": assistant.content or ""}
            if assistant.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in assistant.tool_calls
                ]
            messages.append(assistant_msg)
            num_turns += 1

            tool_calls = assistant_msg.get("tool_calls", [])
            if not tool_calls:
                break

            for tc in tool_calls:
                if total_tool_calls >= max_tool_calls:
                    break
                total_tool_calls += 1
                try:
                    tool_args = json.loads(tc["function"]["arguments"])
                    observation = await self._call_tool(session, tc["function"]["name"], tool_args)
                except json.JSONDecodeError as e:
                    observation = f"Error: invalid tool arguments JSON: {e}"
                messages.append({"role": "tool", "content": observation, "tool_call_id": tc["id"]})
            if total_tool_calls >= max_tool_calls:
                break

        return AtlasTrajectoryOutput(
            trajectory_id=trajectory_id,
            messages=messages,
            prompt_messages=prompt_messages,
            tools=tools,
            stop_reason=stop_reason,
            num_turns=num_turns,
        )

    def _dump_trajectory(
        self,
        batch_dir: Path,
        task_id: str,
        trajectory_id: TrajectoryID,
        output: AtlasTrajectoryOutput,
        final_response: str,
    ) -> None:
        path = batch_dir / f"{task_id}_rep{trajectory_id.repetition_id}.json"
        path.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "reward": output.reward,
                    "num_turns": output.num_turns,
                    "final_response": final_response,
                    "prompt_messages": output.prompt_messages,
                    "messages": output.messages,
                },
                indent=2,
            )
        )

    def _tokenize_trajectory(self, traj: AtlasTrajectoryOutput):
        """Re-tokenize one conversation into (prompt_ids, response_ids, loss_mask, stop_reason)."""
        response_messages = list(traj.messages)
        while response_messages and response_messages[-1]["role"] != "assistant":
            response_messages.pop()
        if not response_messages:
            return [0], [0], [0], "error"

        prompt_ids = self.tokenizer.apply_chat_template(
            traj.prompt_messages,
            tools=traj.tools or None,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
        )
        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
            response_messages, self.tokenizer, assistant_logprobs=None
        )

        stop_reason = traj.stop_reason
        max_response_tokens = self.max_seq_len - len(prompt_ids)
        if max_response_tokens <= 0:
            logger.warning(f"Trajectory {traj.trajectory_id}: prompt alone exceeds max_seq_len; masking.")
            return [0], [0], [0], "error"
        if len(response_ids) > max_response_tokens:
            response_ids = response_ids[:max_response_tokens]
            loss_mask = loss_mask[:max_response_tokens]
            stop_reason = "length"
        return prompt_ids, response_ids, loss_mask, stop_reason

    def _build_generator_output(self, trajectory_outputs: List[AtlasTrajectoryOutput]) -> GeneratorOutput:
        prompt_token_ids: List[List[int]] = []
        response_ids: List[List[int]] = []
        rewards: List[float] = []
        loss_masks: List[List[int]] = []
        stop_reasons: List[str] = []
        trajectory_generation_times: List[Optional[float]] = []

        successful: List[AtlasTrajectoryOutput] = []
        response_ids_for_metrics: List[List[int]] = []
        num_error_trajectories = 0
        num_truncated_trajectories = 0

        for traj in trajectory_outputs:
            trajectory_generation_times.append(traj.e2e_time)
            if traj.stop_reason == "error" or traj.messages is None:
                num_error_trajectories += 1
                prompt_token_ids.append([0])
                response_ids.append([0])
                rewards.append(0.0)
                loss_masks.append([0])
                stop_reasons.append("error")
                continue

            p_ids, r_ids, mask, stop_reason = self._tokenize_trajectory(traj)
            if stop_reason == "error":
                num_error_trajectories += 1
            else:
                successful.append(traj)
                response_ids_for_metrics.append(r_ids)
                if stop_reason == "length":
                    num_truncated_trajectories += 1
            prompt_token_ids.append(p_ids)
            response_ids.append(r_ids)
            rewards.append(traj.reward if stop_reason != "error" else 0.0)
            loss_masks.append(mask)
            stop_reasons.append(stop_reason)

        if any(t is None for t in trajectory_generation_times):
            trajectory_generation_times = None

        if successful:
            rollout_metrics = get_rollout_metrics(
                response_ids_for_metrics,
                [t.reward for t in successful],
                trajectory_completion_times=(
                    [t.e2e_time for t in successful] if all(t.e2e_time is not None for t in successful) else None
                ),
            )
            rollout_metrics["generate/avg_num_turns"] = sum(t.num_turns for t in successful) / len(successful)
            rollout_metrics["generate/avg_coverage_score"] = sum(t.reward for t in successful) / len(successful)
        else:
            rollout_metrics = {}
        rollout_metrics["generate/num_error_trajectories"] = num_error_trajectories
        rollout_metrics["generate/num_truncated_trajectories"] = num_truncated_trajectories

        return GeneratorOutput(
            prompt_token_ids=prompt_token_ids,
            response_ids=response_ids,
            rewards=rewards,
            loss_masks=loss_masks,
            stop_reasons=stop_reasons,
            rollout_metrics=rollout_metrics,
            rollout_logprobs=None,
            trajectory_generation_times=trajectory_generation_times,
        )
