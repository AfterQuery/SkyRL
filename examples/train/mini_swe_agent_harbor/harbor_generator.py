"""Harbor variant of the mini-swe-agent generator.

Reuses ``examples/train/mini_swe_agent`` wherever possible:

* ``MiniSweAgentGenerator`` -- subclassed; only ``minisweagent_agent_loop`` is overridden,
  so the message-to-token conversion and ``generate()`` batching are inherited verbatim.
* ``get_sb_environment`` -- imported unchanged. It calls ``get_docker_image_name``, which
  short-circuits on ``instance["image_name"]``; ``preprocess_harbor.py`` pre-builds each
  Harbor image and writes its tag there, so the SWE-Bench registry-naming branch never
  runs and ``data_source`` is inert (we pass ``"harbor"``).
* ``get_response_ids_and_loss_mask_from_messages`` / ``generate()`` -- inherited.

* ``DefaultAgentWithReminder`` -- imported unchanged. It targets the v1 agent API
  (``get_observation`` / ``self.model.n_calls`` / ``add_message``), which is what the
  pinned fork provides.

Only ``init_and_run`` is reimplemented, and for exactly one reason: **the grading target**.
SWE-Bench replays ``git diff --cached`` into a *fresh* container; Harbor has no patch --
the container the agent mutated *is* the submission -- so the live environment must reach
the verifier. ``init_and_run`` is the only place holding the live ``env``, and Ray
re-imports the defining module in the worker, so monkeypatching the original from the
driver would not take effect.

**Version pin.** This module targets mini-swe-agent v1, pinned in
``requirements-miniswe.txt`` to the scaleapi fork @ d74716a (1.15.0) and applied via
``--with-requirements`` in the run script. ``uv.lock`` otherwise resolves 2.4.2, whose
breaking changes are all silent here: ``run()`` returns a dict whose keys would be
unpacked as if they were the exit status and submission, ``env.execute`` takes a dict
rather than a string, and the ``get_observation`` hook no longer exists so the turn
reminder would simply never fire.

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
from loguru import logger
from minisweagent.agents.default import FormatError
from minisweagent.config import get_config_path
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj

from skyrl.train.config import GeneratorConfig
from skyrl.train.generators.base import BatchMetadata, TrainingPhase, TrajectoryID
from skyrl.train.generators.skyrl_gym_generator import ConversationType
from skyrl.train.generators.utils import get_response_ids_and_loss_mask_from_messages

# Reused as-is from the SWE-Bench example. ``DefaultAgentWithReminder`` targets the v1
# agent API (``get_observation`` / ``self.model.n_calls`` / ``add_message``), which is what
# the pinned fork provides, so it is imported rather than reimplemented.
from examples.train.mini_swe_agent.mini_swe_generator import (
    DefaultAgentWithReminder,
    MiniSweAgentGenerator,
    MiniSWEGeneratorConfig,
)
from examples.train.mini_swe_agent.mini_swe_utils import get_sb_environment

from .harbor_tasks import evaluate_harbor_task


@dataclass
class MiniSWEHarborGeneratorConfig(MiniSWEGeneratorConfig):
    """Harbor knobs; inherits miniswe_config_path / miniswe_traj_dir."""

    # Fallback only: the per-task `[verifier] timeout_sec` from task.toml wins when set.
    harbor_verifier_timeout: float = 600.0
    # Per-malformed-turn reward penalty, and the total it may reach across an episode.
    # The cap keeps shaping from outweighing the task signal: at 0.01/error an unbounded
    # 50-error spiral would reach -0.5 and swamp the 0..1 solve reward.
    harbor_format_error_penalty: float = 0.01
    harbor_format_error_penalty_cap: float = 0.3


class FormatErrorCountingAgent(DefaultAgentWithReminder):
    """Counts malformed turns so the reward can penalise format spirals.

    mini-swe-agent owns the loop here (unlike tinker_cookbook's hand-rolled one, which can
    apply a per-step reward inline), and v1 swallows ``FormatError`` in ``run()`` as a
    ``NonTerminatingException`` -- the template is bounced back to the model and the turn
    is otherwise invisible. Counting in ``parse_action`` is the one hook that sees every
    occurrence; the total is converted into a single scalar penalty after ``run()``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.format_errors = 0

    def parse_action(self, response: dict) -> dict:
        try:
            return super().parse_action(response)
        except FormatError:
            self.format_errors += 1
            raise


def _openai_sampling_params(sampling_params: dict) -> dict:
    """Strip vLLM-native keys that the OpenAI-compatible wire format rejects.

    ``get_sampling_params_for_backend`` builds params for vLLM's *native* API, but
    mini-swe-agent reaches the engine through LiteLLM's ``openai/`` provider, i.e. via
    ``/v1/chat/completions``. Most vLLM-only keys (``min_tokens``, ``skip_special_tokens``,
    ``include_stop_str_in_output``, ``top_k``, ``min_p``) are handled by ``drop_params:
    true`` in the model config, which drops what the provider does not support.

    ``logprobs`` is the exception, and it is fatal: OpenAI *does* support the name, so
    drop_params keeps it -- but there it is a **bool**, with the count in ``top_logprobs``,
    while SamplingParams.logprobs defaults to the integer ``1``. vLLM's deserializer then
    rejects every request::

        BadRequestError: Failed to deserialize the JSON body into the target type:
        logprobs: invalid type: integer `1`, expected a boolean

    Dropping rather than translating to ``logprobs=True, top_logprobs=n`` is deliberate:
    this generator reconstructs ``response_ids`` by re-encoding ``agent.messages`` and
    passes ``assistant_logprobs=None``, so sampler logprobs are discarded anyway. Asking
    for them would only inflate every response.

    NOTE: ``examples/train/mini_swe_agent`` feeds the same unfiltered dict to LiteLLM and
    has the same bug.
    """
    return {k: v for k, v in sampling_params.items() if k != "logprobs"}


_UNAME_KEYS = ("system", "release", "version", "machine")


def _container_uname(env) -> dict[str, str]:
    """``platform.uname()``-shaped template vars, read from inside the container.

    ``instance_template`` interpolates these; ``LocalEnvironment`` supplies them via
    ``platform.uname()`` but ``DockerEnvironment`` does not, and Jinja runs with
    ``StrictUndefined``. Falls back to placeholders rather than letting a flaky ``uname``
    call take down an otherwise fine episode -- a wrong OS string costs a little prompt
    accuracy, an exception costs the whole trajectory.
    """
    try:
        # One field per line, not `uname -s -r -v -m`: the version field itself contains
        # spaces ("#49~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Wed Jan 28 ..."), so positional
        # splitting silently shifts the machine value into it.
        out = env.execute("uname -s; uname -r; uname -v; uname -m")
        if out.get("returncode") == 0:
            parts = (out.get("output") or "").strip().splitlines()
            if len(parts) == 4:
                return dict(zip(_UNAME_KEYS, (p.strip() for p in parts)))
    except Exception as e:  # noqa: BLE001 - prompt detail is not worth an episode
        logger.debug(f"uname lookup failed, using placeholders: {e}")
    return {"system": "Linux", "release": "unknown", "version": "unknown", "machine": "x86_64"}


def format_error_penalty(n_errors: int, per_error: float, cap: float) -> float:
    """Total (non-positive) penalty for ``n_errors`` malformed turns.

    Scalar equivalent of the reference's per-step shaping: it emits ``-per_error`` on each
    malformed turn while the running total stays under ``cap``, which sums to exactly this
    over the episode.
    """
    if n_errors <= 0 or per_error <= 0:
        return 0.0
    return -min(n_errors * per_error, cap)


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
) -> Tuple[List[dict], float, Optional[str], bool]:
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
    model_config.setdefault("model_kwargs", {}).update(_openai_sampling_params(sampling_params))
    model = get_model(litellm_model_name, model_config)

    agent = env = extra_info = None
    exit_status, submission = "unknown", ""
    reward, error, eval_error = 0.0, None, None
    # No verdict until grading says otherwise: an episode that dies before grading (image
    # pull failure, container start failure, agent crash) must not train as a zero.
    grading_failed = True

    try:
        # Each task image has its own WORKDIR, recorded by preprocess_harbor.py. It must be
        # applied per instance: mini-swe-agent always passes `-w <cwd>` to podman, and
        # podman refuses to start a container whose workdir does not exist, so a single
        # cwd in the yaml hard-fails every task that uses a different one. The yaml value
        # remains the fallback for images that declare no WORKDIR.
        if workdir := instance.get("workdir"):
            sweagent_config.setdefault("environment", {})["cwd"] = workdir

        env = get_sb_environment(sweagent_config, instance, data_source)
        agent = FormatErrorCountingAgent(model, env, **sweagent_config.get("agent", {}))
        # v1: run() returns (exit_status, submission). On v2 it returns a dict and this
        # unpacking silently yields the dict's *keys* instead of raising -- see the pin in
        # requirements-miniswe.txt.
        #
        # The uname kwargs feed `<system_information>{{system}} {{release}} {{version}}
        # {{machine}}</system_information>` in instance_template. Those come free from
        # LocalEnvironment (it merges platform.uname() into its template vars) but NOT
        # from DockerEnvironment, which exposes only asdict(config) -- and templates
        # render with StrictUndefined, so without this every episode dies on
        # `UndefinedError: 'system' is undefined`. run() folds kwargs into
        # extra_template_vars, and reading uname from the container (not the host) is
        # also the honest answer, including for the template's `system == "Darwin"` branch.
        exit_status, submission = agent.run(instance["problem_statement"], **_container_uname(env))
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
                # Prefer the task's own `[verifier] timeout_sec` over the global config.
                # A single global value badly under-serves this task set: 301 of 437 tasks
                # ask for more than 600s (233 of them want 3000s), and a verifier killed
                # early scores a slow-but-correct solution 0 -- teaching the policy that a
                # real fix failed. generator_cfg.harbor_verifier_timeout is the fallback
                # for tasks whose task.toml declares nothing (recorded as 0.0).
                verifier_timeout = instance.get("verifier_timeout_sec") or getattr(
                    generator_cfg, "harbor_verifier_timeout", 600.0
                )
                evaluation = evaluate_harbor_task(
                    instance,
                    env,
                    timeout=float(verifier_timeout),
                    container_tool=sweagent_config.get("environment", {}).get("executable", "podman"),
                )
                reward = float(evaluation["reward"])
                grading_failed = bool(evaluation["grading_failed"])
                eval_error = evaluation["eval_error"]
                if eval_error:
                    error = error or eval_error
                    logger.debug(f"Error during evaluation {eval_error}")
            except Exception as e:
                logger.debug(f"Error during evaluation {e}\n{traceback.format_exc()}")
                eval_error = error = str(e)
                grading_failed = True

            # Penalise format spirals, but only on a trajectory that will actually be
            # trained on. Applying it to a dropped one is pointless, and applying it to a
            # grading failure would smuggle a fabricated negative reward into the batch.
            if not grading_failed:
                penalty = format_error_penalty(
                    getattr(agent, "format_errors", 0),
                    getattr(generator_cfg, "harbor_format_error_penalty", 0.0),
                    getattr(generator_cfg, "harbor_format_error_penalty_cap", 0.3),
                )
                if penalty:
                    logger.debug(f"{instance['instance_id']}: {agent.format_errors} format errors -> {penalty:+.3f}")
                reward += penalty

            save_traj(
                agent,
                path,
                exit_status=exit_status,
                result=submission,
                extra_info=extra_info,
                reward=reward,
                eval_error=eval_error,
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

    return (agent.messages if agent is not None else [], reward, error, grading_failed)


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
        messages, reward, error, grading_failed = await init_and_run.remote(
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

        # The verifier never returned a verdict (dead container, upload failure, timeout,
        # or a test.sh that exited without writing a reward file). Its 0.0 is an artefact
        # of broken grading, not evidence the agent failed, so drop the trajectory rather
        # than train on a false negative. `generate()` filters `None` responses out.
        if grading_failed:
            logger.warning(
                f"Dropping trajectory for {env_extras['instance']['instance_id']}: "
                f"grading produced no verdict ({error})"
            )
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
