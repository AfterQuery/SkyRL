"""Harbor variant of the mini-swe-agent generator.

Reuses ``examples/train/mini_swe_agent`` wherever possible:

* ``MiniSweAgentGenerator`` -- subclassed; only ``minisweagent_agent_loop`` is overridden,
  so the message-to-token conversion and ``generate()`` batching are inherited verbatim.
* ``get_sb_environment`` -- imported unchanged. It calls ``get_docker_image_name``, which
  short-circuits on ``instance["image_name"]``; ``preprocess_harbor.py`` pre-builds each
  Harbor image and writes its tag there, so the SWE-Bench registry-naming branch never
  runs and ``data_source`` is inert (we pass ``"harbor"``).
* ``get_response_ids_and_loss_mask_from_messages`` / ``generate()`` -- inherited.

``DefaultAgentWithReminder`` is deliberately *not* reused: theirs targets the v1 agent
API and silently does nothing under the locked v2. See the class below.

Only ``init_and_run`` is reimplemented, for two reasons that cannot be patched around
(Ray re-imports the defining module in the worker, so monkeypatching the original module's
globals from the driver would not take effect):

1. **Grading target.** SWE-Bench replays ``git diff --cached`` into a *fresh* container.
   Harbor has no patch -- the container the agent mutated *is* the submission -- so the live
   environment must reach the verifier.
2. **v2 API.** ``uv.lock`` resolves ``mini-swe-agent==2.4.2``, where ``agent.run()``
   returns a **dict**. The original unpacks it as ``exit_status, result = agent.run(...)``,
   which yields the dict's *keys*, making its "patch" the literal string ``"submission"``.

Inherited limitation worth stating plainly: ``response_ids`` are reconstructed by
re-encoding ``agent.messages`` and ``rollout_logprobs`` is ``None``, so these are not the
exact tokens the policy emitted. ``mini-swe-tinker/`` in this workspace captures ids and
logprobs at the sampler instead.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ray
import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_path
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj

from skyrl.train.config import GeneratorConfig
from skyrl.train.generators.base import BatchMetadata, TrainingPhase, TrajectoryID
from skyrl.train.generators.skyrl_gym_generator import ConversationType
from skyrl.train.generators.utils import get_response_ids_and_loss_mask_from_messages

# Reused as-is from the SWE-Bench example.
from examples.train.mini_swe_agent.mini_swe_generator import (
    MiniSweAgentGenerator,
    MiniSWEGeneratorConfig,
)
from examples.train.mini_swe_agent.mini_swe_utils import get_sb_environment

from .harbor_tasks import evaluate_harbor_task



class DefaultAgentWithReminder(DefaultAgent):
    """Tell the model how many turns remain.

    NOT reused from the SWE-Bench example: that version overrides ``get_observation`` and
    reaches for ``self.model.n_calls`` / ``add_message`` / ``action_observation_template``,
    all of which are the **v1** API. Under the locked ``mini-swe-agent==2.4.2`` there is no
    ``get_observation`` hook at all, so its reminder never fires -- silently, since an
    unused override raises nothing. In v2 the observation is appended by
    ``execute_actions``, and ``n_calls`` lives on the agent.

    The reminder matters: without it a model under a hard step limit tends to keep
    exploring and never submit.
    """

    def execute_actions(self, message: dict) -> list[dict]:
        observations = super().execute_actions(message)
        if not self.config.step_limit:
            return observations
        remaining = self.config.step_limit - self.n_calls
        if remaining == 1:
            note = "REMINDER: You only have 1 turn left. Please provide the final answer"
        elif remaining > 1:
            note = f"REMINDER: You have {remaining} turns left to arrive at the solution."
        else:
            note = "REMINDER: This is your final turn."
        return observations + self.add_messages({"role": "user", "content": note})


@dataclass
class MiniSWEHarborGeneratorConfig(MiniSWEGeneratorConfig):
    """Adds the Harbor verifier timeout; inherits miniswe_config_path / miniswe_traj_dir."""

    harbor_verifier_timeout: float = 600.0


@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    sweagent_config: dict,
    generator_cfg: GeneratorConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
    base_url: str,
) -> Tuple[List[dict], float, Optional[str]]:
    """Run one episode in a fresh Harbor container and grade it in place.

    ``data_source`` is accepted but unused. In the SWE-Bench example it selects the Docker
    registry and the instance_id mangling (``swe-gym`` -> xingyaoww/``__``->``_s_``;
    ``swe-bench`` -> swebench/``__``->``_1776_``). Harbor carries a pre-built
    ``instance["image_name"]``, which short-circuits that lookup. Keeping the parameter
    means the call site stays byte-identical to the original method.
    """
    from loguru import logger

    # mini-swe-agent reaches the inference router via LiteLLM's ``openai/`` provider, which
    # POSTs to ``{OPENAI_BASE_URL}/chat/completions``; the router serves OpenAI under /v1.
    os.environ["OPENAI_BASE_URL"] = f"{base_url}/v1"

    model_config = sweagent_config.get("model", {})
    model_config.setdefault("model_kwargs", {}).update(sampling_params)
    model = get_model(litellm_model_name, model_config)

    agent = env = extra_info = None
    exit_status, submission = "unknown", ""
    reward, error, eval_error = 0.0, None, None

    try:
        env = get_sb_environment(sweagent_config, instance, data_source)
        agent = DefaultAgentWithReminder(model, env, **sweagent_config.get("agent", {}))
        # v2: run() returns the terminal message's `extra` dict.
        result = agent.run(instance["problem_statement"])
        exit_status = str(result.get("exit_status", "unknown"))
        submission = str(result.get("submission", ""))
    except Exception as e:
        logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
        exit_status, error = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        path = Path(generator_cfg.miniswe_traj_dir) / f"step_{global_step}" / training_phase
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"{instance['instance_id']}_{trajectory_id.repetition_id}.json"

        if agent is not None and env is not None:
            try:
                # Grade the container the agent just mutated -- no patch to replay.
                evaluation = evaluate_harbor_task(
                    instance,
                    env,
                    timeout=getattr(generator_cfg, "harbor_verifier_timeout", 600.0),
                    container_tool=sweagent_config.get("environment", {}).get("executable", "podman"),
                )
                reward = float(evaluation["reward"])
                eval_error = evaluation["eval_error"]
                if eval_error:
                    error = error or eval_error
                    logger.debug(f"Error during evaluation {eval_error}")
            except Exception as e:
                logger.debug(f"Error during evaluation {e}\n{traceback.format_exc()}")
                eval_error = error = str(e)

            save_traj(
                agent, path, exit_status=exit_status, result=submission,
                extra_info=extra_info, reward=reward, eval_error=eval_error,
            )

        # Harbor containers are per-trajectory. The SWE-Bench example leaks them to GC,
        # which across hundreds of concurrent rollouts fills the disk.
        if env is not None:
            for hook in ("cleanup", "close", "stop"):
                fn = getattr(env, hook, None)
                if callable(fn):
                    try:
                        fn()
                        break
                    except Exception as e:
                        logger.debug(f"env.{hook}() failed: {e}")

    return (agent.messages if agent is not None else [], reward, error)


class MiniSweAgentHarborGenerator(MiniSweAgentGenerator):
    """Same generator, Harbor tasks. Only the per-trajectory Ray call differs."""

    async def minisweagent_agent_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]]]:

        sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        messages, reward, error = await init_and_run.remote(
            env_extras["instance"],
            self.litellm_model_name,
            sweagent_config,
            self.generator_cfg,
            env_extras["data_source"],
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
            self.base_url,
        )
        if not len(messages):
            return None, None, None, None, None, None

        # TODO (sumanthrh): This is currently hardcoded for SWEBench with 2 initial messages (system and user).
        response_messages = messages[2:]

        for message in messages[:2]:
            assert message["role"] in (
                "system",
                "user",
            ), "Expected the first two messages to be system and user messages"

        initial_input_ids = self.tokenizer.apply_chat_template(
            messages[:2], add_generation_prompt=False, return_dict=False, tokenize=True
        )
        initial_prompt_length = len(initial_input_ids)

        # We remove trailing `user` messages - this is added by Mini-SWE-Agent to capture the final git diff for the trajectory
        last_idx = len(response_messages) - 1
        # DEVIATION from the original: guard last_idx >= 0. Without it, an all-`user`
        # response walks negative and IndexErrors instead of raising the message below.
        while last_idx >= 0 and response_messages[last_idx]["role"] == "user":
            last_idx -= 1
        if last_idx < 0:
            raise ValueError(
                "Found no assistant messages. Please ensure that your environment is configured correctly and the `OPENAI_BASE_URL` points to the HTTP server from the inference engine client"
            )
        response_messages = response_messages[: last_idx + 1]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
            response_messages,
            self.tokenizer,
            assistant_logprobs=None,
        )

        # Extract prompt ids
        prompt_ids = initial_input_ids

        # Calculate maximum response tokens allowed
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length

        # Determine stop reason
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return (response_ids, reward, stop_reason, loss_mask, prompt_ids, None)
