"""Generator that runs Toolathlon tasks against SkyRL's OpenAI-compatible inference endpoint.

Each trajectory shells out to Toolathlon's decoupled runner
(``scripts/run_single_decoupled.sh``): the task container does preprocess and eval, and the
host-side agent loop talks to SkyRL's inference endpoint via ``TOOLATHLON_OPENAI_BASE_URL``.
Afterwards we read the trajectory (``traj_log.json``) and reward (``eval_res.json``) from the
dump directory and re-tokenize the conversation to build training data.
"""

import asyncio
import json
import os
import shutil
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger
from tqdm import tqdm

from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineInterface
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

# Retries per trajectory on harness errors (container startup failures etc.). Legitimate task
# failures (evaluator returns pass=false) and max-turns terminations are never retried.
MAX_NUM_RETRIES_PER_TASK = 2

# Message keys forwarded to the chat template during re-tokenization. Toolathlon adds
# non-standard keys (e.g. "thinking") that chat templates would reject or mis-render.
_MESSAGE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")


@dataclass
class ToolathlonTrajectoryOutput:
    """One trajectory's parsed output from a Toolathlon run."""

    trajectory_id: TrajectoryID
    # OpenAI-format conversation excluding the system prompt; None for failed runs.
    messages: Optional[List[Dict[str, Any]]] = None
    system_prompt: str = ""
    # OpenAI function-tool schemas the agent ran with, used for prompt re-tokenization.
    tools: List[Dict[str, Any]] = field(default_factory=list)
    reward: float = 0.0
    # One of: "complete", "error". "error" trajectories are loss-masked.
    stop_reason: str = "complete"
    num_turns: int = 0
    e2e_time: Optional[float] = None


class ToolathlonGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg,
        toolathlon_cfg: Dict[str, Any],
        inference_engine_client: InferenceEngineInterface,
        tokenizer,
        max_seq_len: int,
    ):
        """
        Args:
            generator_cfg: Generator configuration (``cfg.generator``).
            toolathlon_cfg: Toolathlon harness configuration (see ``toolathlon_config.yaml``).
            inference_engine_client: Client for the inference engines; provides the HTTP endpoint.
            tokenizer: Tokenizer used to re-tokenize conversations into training data.
            max_seq_len: Maximum total sequence length (prompt + response) for truncation.
        """
        self.generator_cfg = generator_cfg
        self.toolathlon_cfg = toolathlon_cfg
        self.inference_engine_client = inference_engine_client
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.base_url = inference_engine_client.get_endpoint_url()

        self._served_model_name = generator_cfg.inference_engine.served_model_name
        assert self._served_model_name is not None, "generator.inference_engine.served_model_name must be set"

        self.repo_path = Path(toolathlon_cfg["repo_path"]).expanduser().resolve()
        runner = self.repo_path / toolathlon_cfg["runner_script"]
        if not runner.is_file():
            raise ValueError(f"Toolathlon runner script not found: {runner}")
        # The runner invokes `uv run` on the host; Ray workers may have a minimal PATH.
        uv_path = shutil.which("uv") or (
            str(Path.home() / ".local" / "bin" / "uv") if (Path.home() / ".local" / "bin" / "uv").is_file() else None
        )
        if uv_path is None:
            raise ValueError("`uv` not found on PATH; the Toolathlon runner requires it on the host.")
        self._uv_bin_dir = str(Path(uv_path).parent)
        global_configs = self.repo_path / "configs" / "global_configs.py"
        if not global_configs.is_file():
            raise ValueError(
                f"{global_configs} not found. Copy configs/global_configs_example.py to "
                "configs/global_configs.py in the Toolathlon repo and set podman_or_docker."
            )

        self.dump_root = Path(toolathlon_cfg["dump_root"]).expanduser().resolve()
        self.dump_root.mkdir(parents=True, exist_ok=True)

        # Concurrency cap on simultaneous task containers.
        self._semaphore = asyncio.Semaphore(int(toolathlon_cfg["max_concurrent_tasks"]))
        # Tasks in the same conflict group mutate shared external app state (same Canvas course,
        # same mailbox, ...) and must not run concurrently. Repetitions of the same task also
        # serialize on these locks: each run's preprocess resets the task's app state, so
        # concurrent repetitions would corrupt each other.
        self._task_locks = self._build_task_locks()

        if getattr(generator_cfg, "step_wise_trajectories", False):
            raise ValueError(
                "ToolathlonGenerator uses re-tokenization and does not support step-wise training. "
                "Set generator.step_wise_trajectories=false."
            )

    def _build_task_locks(self) -> Dict[str, asyncio.Lock]:
        """Map each task name to a lock, shared across tasks in the same conflict group."""
        locks: Dict[str, asyncio.Lock] = {}
        conflict_file = self.repo_path / "tasks" / self.toolathlon_cfg["tasks_domain"] / "task_conflict.json"
        if conflict_file.is_file():
            groups = json.loads(conflict_file.read_text()).get("conflict_groups", [])
            for group in groups:
                shared_lock = asyncio.Lock()
                for task_name in group:
                    locks[task_name] = shared_lock
        return locks

    def _get_task_lock(self, task_name: str) -> asyncio.Lock:
        if task_name not in self._task_locks:
            self._task_locks[task_name] = asyncio.Lock()
        return self._task_locks[task_name]

    def _write_model_params_file(self, batch_dir: Path, sampling_params: Optional[Dict[str, Any]]) -> Path:
        """Write the sampling params consumed via ``TOOLATHLON_MODEL_PARAMS_FILE``.

        Toolathlon merges this JSON into every /chat/completions request body, so the policy is
        sampled with the trainer's sampling params rather than Toolathlon's defaults.
        """
        allowed_keys = ("temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "logprobs")
        params: Dict[str, Any] = {}
        if sampling_params:
            params = {k: v for k, v in sampling_params.items() if k in allowed_keys and v is not None}
        params.setdefault("max_tokens", self.generator_cfg.sampling_params.max_generate_length)
        batch_dir.mkdir(parents=True, exist_ok=True)
        params_file = batch_dir / "model_params.json"
        params_file.write_text(json.dumps(params, indent=2))
        return params_file

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]
        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")

        batch_metadata = input_batch.get("batch_metadata")
        if batch_metadata is not None:
            batch_tag = f"step_{batch_metadata.global_step}_{batch_metadata.training_phase}"
        else:
            batch_tag = f"batch_{uuid4().hex[:8]}"
        batch_dir = self.dump_root / batch_tag
        params_file = self._write_model_params_file(batch_dir, input_batch.get("sampling_params"))

        all_outputs: List[ToolathlonTrajectoryOutput] = [None] * len(prompts)  # type: ignore[list-item]
        progress = tqdm(
            disable=disable_tqdm,
            total=len(prompts),
            desc="Generating Toolathlon Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async def _worker(idx: int, task_name: str, trajectory_id: TrajectoryID):
            all_outputs[idx] = await self._run_one_trajectory(task_name, trajectory_id, batch_dir, params_file)
            progress.update(1)

        try:
            async with asyncio.TaskGroup() as tg:
                for idx, (task_name, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                    tg.create_task(_worker(idx, task_name, trajectory_id))
        finally:
            progress.close()

        return self._build_generator_output(all_outputs)

    async def _run_one_trajectory(
        self, task_name: str, trajectory_id: TrajectoryID, batch_dir: Path, params_file: Path
    ) -> ToolathlonTrajectoryOutput:
        """Run one Toolathlon task end-to-end and parse its trajectory and reward."""
        start_time = time.monotonic()
        cfg = self.toolathlon_cfg
        task_dir_arg = f"{cfg['tasks_domain']}/{task_name}"
        lock = self._get_task_lock(task_name)

        for attempt in range(MAX_NUM_RETRIES_PER_TASK):
            prefix = f"Trajectory {trajectory_id} ({task_name}) attempt {attempt + 1}/{MAX_NUM_RETRIES_PER_TASK}"
            # Unique dump dir per (trajectory, attempt); the runner writes results under
            # <dump>/<domain>/<task>/ so distinct roots keep repetitions apart.
            dump_dir = batch_dir / f"{task_name}_rep{trajectory_id.repetition_id}_try{attempt}"
            try:
                # Lock first so trajectories queued on a conflicting task don't hold a
                # container slot while waiting.
                async with lock:
                    async with self._semaphore:
                        returncode = await self._run_toolathlon_subprocess(task_dir_arg, dump_dir, params_file)
                output = self._parse_task_output(dump_dir / cfg["tasks_domain"] / task_name, trajectory_id)
                if output is not None:
                    output.e2e_time = time.monotonic() - start_time
                    return output
                logger.warning(f"{prefix} produced no usable trajectory (runner exit code {returncode}); retrying.")
            except Exception as e:
                logger.warning(f"{prefix} failed: {e}")

        logger.warning(f"Trajectory {trajectory_id} ({task_name}) failed; setting loss mask to [0].")
        return ToolathlonTrajectoryOutput(
            trajectory_id=trajectory_id, stop_reason="error", e2e_time=time.monotonic() - start_time
        )

    async def _run_toolathlon_subprocess(self, task_dir_arg: str, dump_dir: Path, params_file: Path) -> int:
        """Invoke the decoupled runner for one task; returns the process exit code."""
        cfg = self.toolathlon_cfg
        dump_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "bash",
            str(self.repo_path / cfg["runner_script"]),
            task_dir_arg,
            cfg["runmode"],
            str(dump_dir),
            self._served_model_name,
            cfg["provider"],
            str(cfg["max_steps"]),
            cfg["eval_config"],
            cfg["image_name"],
            cfg["agent_framework"],
        ]
        env = os.environ.copy()
        env["PATH"] = self._uv_bin_dir + os.pathsep + env.get("PATH", "")
        env["TOOLATHLON_OPENAI_BASE_URL"] = f"{self.base_url}/v1"
        env["TOOLATHLON_OPENAI_API_KEY"] = cfg["api_key"]
        env["TOOLATHLON_MODEL_PARAMS_FILE"] = str(params_file)

        runner_log = dump_dir / "runner.log"
        with open(runner_log, "wb") as log_file:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.repo_path),
                env=env,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=float(cfg["task_timeout_seconds"]))
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # SIGTERM the whole process group so the runner's cleanup trap stops the
                # task container; escalate to SIGKILL if it doesn't exit.
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    await asyncio.wait_for(proc.wait(), timeout=60)
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await proc.wait()
                raise asyncio.TimeoutError(f"Toolathlon runner timed out (log: {runner_log})")
        return proc.returncode

    def _parse_task_output(
        self, task_output_dir: Path, trajectory_id: TrajectoryID
    ) -> Optional[ToolathlonTrajectoryOutput]:
        """Parse traj_log.json + eval_res.json into a trajectory output.

        Returns None when the trajectory is unusable and worth retrying (missing/empty logs).
        """
        traj_log_file = task_output_dir / "traj_log.json"
        eval_res_file = task_output_dir / "eval_res.json"
        if not traj_log_file.is_file():
            return None
        traj_log = json.loads(traj_log_file.read_text())
        messages = traj_log.get("messages") or []
        if not any(m.get("role") == "assistant" for m in messages):
            return None

        status = traj_log.get("status", "failed")
        eval_pass = None
        if eval_res_file.is_file():
            eval_pass = json.loads(eval_res_file.read_text()).get("pass")

        # Reward mapping: evaluator pass -> 1.0, fail -> 0.0. `pass: null` means the evaluator
        # did not score the run: for max_turns_reached that is a legitimate policy failure
        # (train with reward 0); for crashes/interruptions the trajectory is loss-masked.
        if eval_pass is True:
            reward, stop_reason = 1.0, "complete"
        elif eval_pass is False:
            reward, stop_reason = 0.0, "complete"
        elif status == "max_turns_reached":
            reward, stop_reason = 0.0, "complete"
        else:
            logger.warning(f"Trajectory {trajectory_id}: unscored run (status={status}, pass={eval_pass}); masking.")
            reward, stop_reason = 0.0, "error"

        return ToolathlonTrajectoryOutput(
            trajectory_id=trajectory_id,
            messages=messages,
            system_prompt=traj_log.get("config", {}).get("system_prompts", {}).get("agent", ""),
            tools=traj_log.get("tool_calls", {}).get("tools", []),
            reward=reward,
            stop_reason=stop_reason,
            num_turns=sum(1 for m in messages if m.get("role") == "assistant"),
        )

    @staticmethod
    def _sanitize_message(message: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only chat-template-compatible keys; chat templates require string content."""
        sanitized = {k: v for k, v in message.items() if k in _MESSAGE_KEYS and v is not None}
        sanitized.setdefault("content", "")
        return sanitized

    def _tokenize_trajectory(self, traj: ToolathlonTrajectoryOutput):
        """Re-tokenize one conversation into (prompt_ids, response_ids, loss_mask, stop_reason).

        The prompt is the system prompt plus all messages before the first assistant turn,
        rendered with the tool schemas the agent ran with. Everything from the first assistant
        turn on is the response; tool/user observation tokens are loss-masked by
        ``get_response_ids_and_loss_mask_from_messages``.
        """
        messages = [self._sanitize_message(m) for m in traj.messages]
        first_assistant_idx = next(i for i, m in enumerate(messages) if m["role"] == "assistant")

        prompt_messages = [{"role": "system", "content": traj.system_prompt}] + messages[:first_assistant_idx]
        response_messages = messages[first_assistant_idx:]
        # Drop trailing non-assistant messages (observations after the final assistant turn).
        while response_messages and response_messages[-1]["role"] != "assistant":
            response_messages.pop()

        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
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

    def _build_generator_output(self, trajectory_outputs: List[ToolathlonTrajectoryOutput]) -> GeneratorOutput:
        prompt_token_ids: List[List[int]] = []
        response_ids: List[List[int]] = []
        rewards: List[float] = []
        loss_masks: List[List[int]] = []
        stop_reasons: List[str] = []
        trajectory_generation_times: List[Optional[float]] = []

        successful: List[ToolathlonTrajectoryOutput] = []
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
