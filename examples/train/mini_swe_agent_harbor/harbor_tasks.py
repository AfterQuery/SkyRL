"""Harbor task discovery, image building, and verification.

Harbor (https://github.com/harbor-framework/harbor) standardises a task as a directory::

    <task>/
      environment/Dockerfile   # the sandbox image (BUILD it; not a registry pull)
      tests/test.sh            # verifier; writes /logs/verifier/reward.txt
      instruction.md           # the task statement
      task.toml                # timeouts + metadata

Two things differ from the SWE-Bench flow in ``mini_swe_agent/mini_swe_utils.py``:

1. **Images are built, not pulled.** SWE-Bench derives a published registry tag from
   ``instance_id``; Harbor ships a Dockerfile. We build it here, content-addressed, and
   stash the tag in ``instance["image_name"]`` -- which ``get_docker_image_name`` already
   honours, so no change is needed on that path.
2. **State is not transported as a patch.** SWE-Bench grades by applying
   ``git diff --cached`` inside a *fresh* container. Harbor grades the container the agent
   actually mutated, so the verifier must receive the live environment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict

logger = logging.getLogger(__name__)

REWARD_DIR = "/logs/verifier"
REWARD_TXT = f"{REWARD_DIR}/reward.txt"
REWARD_JSON = f"{REWARD_DIR}/reward.json"

DEFAULT_IMAGE_PREFIX = "skyrl-harbor"


class HarborEvaluationResult(TypedDict):
    """Mirrors ``MiniSWEEvaluationResult`` but carries a float reward.

    Harbor's ``reward.txt`` may be fractional; the SWE-Bench version types this as a bool
    and the generator does ``int(resolved)``, which would silently binarise graded tasks.

    ``grading_failed`` separates "the verifier ran and the agent failed" (reward 0, a real
    training signal) from "the verifier never produced a verdict" (reward 0 only because
    grading broke). Training on the latter teaches the policy that a possibly-correct
    solution was wrong, so the caller drops those trajectories instead.
    """

    instance_id: str
    reward: float
    resolved: bool
    reward_source: str
    eval_error: Optional[str]
    grading_failed: bool


@dataclass(frozen=True)
class HarborTask:
    name: str
    task_dir: Path
    instruction: str
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def dockerfile_dir(self) -> Path:
        return self.task_dir / "environment"

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / "tests"

    def _section_value(self, section: str, key: str, default):
        block = self.config.get(section)
        if not isinstance(block, dict):
            return default
        value = block.get(key, default)
        return default if value is None else value

    def agent_timeout(self, default: float) -> float:
        return float(self._section_value("agent", "timeout_sec", default))

    def verifier_timeout(self, default: float) -> float:
        return float(self._section_value("verifier", "timeout_sec", default))


def load_task(task_dir: Path | str) -> HarborTask:
    task_dir = Path(task_dir).resolve()
    config: dict[str, Any] = {}
    toml_path = task_dir / "task.toml"
    if toml_path.exists():
        try:
            config = tomllib.loads(toml_path.read_text())
        except tomllib.TOMLDecodeError as e:
            logger.warning("ignoring malformed task.toml in %s: %s", task_dir, e)
    return HarborTask(
        name=task_dir.name,
        task_dir=task_dir,
        instruction=(task_dir / "instruction.md").read_text(),
        config=config,
    )


def discover_tasks(root: Path | str, limit: int | None = None) -> list[HarborTask]:
    """Find every complete Harbor task under ``root``, at any nesting depth.

    Harbor's CLI nests tasks as ``<shortuuid>/<task_name>/`` while a packed HF dataset
    unpacks flat, so key off the required files rather than assuming a depth.
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    candidates = {
        dockerfile.parent.parent
        for dockerfile in root.rglob("environment/Dockerfile")
        if (dockerfile.parent.parent / "tests" / "test.sh").exists()
    }
    tasks = []
    for task_dir in sorted(candidates):
        if not (task_dir / "instruction.md").exists():
            logger.warning("skipping %s: no instruction.md", task_dir)
            continue
        tasks.append(load_task(task_dir))
    return tasks[:limit] if limit is not None else tasks


def image_tag(task: HarborTask, prefix: str = DEFAULT_IMAGE_PREFIX) -> str:
    """Content-addressed tag, so editing a Dockerfile invalidates the cached image."""
    digest = hashlib.sha256((task.dockerfile_dir / "Dockerfile").read_bytes()).hexdigest()[:12]
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in task.name.lower())
    return f"{prefix}/{safe}:{digest}"


# A stage boundary is `FROM <image>` optionally followed by `AS <name>`, at column 0.
# The trailing anchor matters: these Dockerfiles embed Python in heredocs, and a lax
# case-insensitive `^\s*FROM\s` also matches `from pathlib import Path`, which would
# truncate the "final stage" to the tail of a heredoc and lose the real WORKDIR.
_FROM_RE = re.compile(r"(?mi)^FROM\s+\S+(?:\s+AS\s+\S+)?\s*$")
_WORKDIR_RE = re.compile(r'(?mi)^\s*WORKDIR\s+"?([^"\s]+)"?\s*$')


def task_workdir(task: HarborTask) -> str | None:
    """The image's effective ``WORKDIR``, or None if its Dockerfile sets none.

    Needed because mini-swe-agent's DockerEnvironment always passes ``-w <config.cwd>`` to
    both ``run`` and ``exec``; the image's own WORKDIR is never consulted. A single cwd in
    the yaml is therefore wrong for any task that does not use it -- and podman (unlike
    docker) *refuses to start* a container whose workdir is absent::

        Error: workdir "/workspace" does not exist on container

    so a mismatch is a hard failure at container start, not a wrong-directory nuisance.

    Only the final build stage is considered: a multi-stage Dockerfile's earlier WORKDIRs
    do not carry into the produced image.
    """
    dockerfile = task.dockerfile_dir / "Dockerfile"
    if not dockerfile.exists():
        return None
    text = dockerfile.read_text()
    froms = list(_FROM_RE.finditer(text))
    final_stage = text[froms[-1].start() :] if froms else text
    matches = _WORKDIR_RE.findall(final_stage)
    return matches[-1] if matches else None


def declared_image(task: HarborTask) -> str | None:
    """Prebuilt image from ``task.toml``'s ``[environment].docker_image``, or None.

    Prefer this over building. A Harbor ``environment/Dockerfile`` is typically
    ``FROM <private repo image>`` plus a layer that applies the task's bug patch, and
    ``docker_image`` is that same Dockerfile already built and pushed. Using it:

    * removes N local builds from preprocessing (a pull, or nothing, instead),
    * drops the requirement to authenticate against the *base* repository, and
    * guarantees every worker runs the identical image the task was authored against,
      rather than whatever a local rebuild happened to resolve.

    Falling back to a local build stays useful for task trees that predate the field.
    """
    value = task._section_value("environment", "docker_image", None)
    return str(value) if value else None


def build_task_image(
    task: HarborTask,
    *,
    executable: str = "podman",
    prefix: str = DEFAULT_IMAGE_PREFIX,
    build_timeout: float = 1800.0,
    force: bool = False,
) -> str:
    """Build the task's sandbox image if absent; return its tag.

    Synchronous on purpose: this runs in ``preprocess_harbor.py`` (a single process, ahead
    of training) rather than in the hot rollout path, so a Ray task per trajectory never
    races on a build.
    """
    tag = image_tag(task, prefix)
    if not force and _image_exists(tag, executable):
        return tag

    logger.info("building %s for task %s", tag, task.name)
    result = subprocess.run(
        [executable, "build", "-t", tag, str(task.dockerfile_dir)],
        capture_output=True,
        text=True,
        timeout=build_timeout,
        check=False,
    )
    if result.returncode != 0:
        tail = (result.stdout or "")[-1500:] + (result.stderr or "")[-1500:]
        raise RuntimeError(f"image build failed for {task.name} ({tag}):\n{tail}")
    return tag


def _image_exists(tag: str, executable: str) -> bool:
    return (
        subprocess.run(
            [executable, "image", "inspect", tag],
            capture_output=True,
            timeout=120,
            check=False,
        ).returncode
        == 0
    )


def evaluate_harbor_task(
    instance: dict[str, Any],
    env: Any,
    *,
    timeout: float = 600.0,
    container_tool: str = "podman",
) -> HarborEvaluationResult:
    """Grade a finished episode by running the task's own ``tests/test.sh`` in ``env``.

    Unlike ``evaluate_trajectory`` in the SWE-Bench example, this takes the **live**
    environment the agent just worked in -- Harbor has no patch to replay into a fresh
    container, so the mutated filesystem *is* the submission.

    Never raises: a task whose verifier cannot run scores 0 with an explanation, so one
    broken task does not take down the whole training step.
    """
    instance_id = instance.get("instance_id", "unknown")
    # grading_failed starts True and is cleared only once a reward file has actually been
    # read. Every early return below is therefore a grading failure by construction --
    # fail-closed, so a new error path can never silently become a trainable zero.
    result = HarborEvaluationResult(
        instance_id=instance_id,
        reward=0.0,
        resolved=False,
        reward_source="",
        eval_error=None,
        grading_failed=True,
    )

    tests_dir = instance.get("tests_dir")
    if not tests_dir:
        result["eval_error"] = "instance has no 'tests_dir'; was it built by preprocess_harbor.py?"
        return result

    container_id = getattr(env, "container_id", None)
    if not container_id:
        result["eval_error"] = "environment exposes no container_id; cannot upload tests"
        return result

    try:
        # 1. Create the destination. podman (unlike docker) will NOT create the target
        #    directory for `cp` and fails with
        #       Error: "/tests/" could not be found on container ...
        #    which would zero the reward for every task.
        env.execute("mkdir -p /tests")

        # 2. Ship the task's tests in. The trailing /. copies directory *contents*, and
        #    `cp` preserves nested fixture trees that a heredoc would mangle.
        copy = subprocess.run(
            [container_tool, "cp", f"{tests_dir}/.", f"{container_id}:/tests/"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if copy.returncode != 0:
            result["eval_error"] = f"failed to copy tests into container: {copy.stderr.strip()[:400]}"
            return result

        # 3. Restore the executable bit. Harbor task trees ship tests/*.sh mode 664 and
        #    `cp` preserves that, but a good number of test.sh scripts invoke their
        #    sibling directly (`/tests/run_script.sh`, not `bash /tests/run_script.sh`)
        #    and would die with "Permission denied" after the tests appeared to upload
        #    fine -- a 0 reward that looks like a failed solution.
        env.execute("chmod -R +x /tests 2>/dev/null || true")

        # 4. test.sh writes its verdict here; some scripts assume the dir exists.
        #    NOTE: v1 signature is execute(command: str). On v2 it is execute(action: dict)
        #    and a bare string raises AttributeError -- see requirements-miniswe.txt.
        env.execute(f"mkdir -p {REWARD_DIR}")

        # 5. Run the verifier from /root, matching tinker_cookbook's HarborReward.
        #
        #    Its stated reason -- "test.sh checks if PWD=/ and exits early" -- does not
        #    hold for this task set: none of the 437 test.sh scripts branch on PWD=/, and
        #    87 cd somewhere themselves. Kept anyway for parity with the reference and
        #    because a non-/ PWD is the safer default for scripts that use relative paths.
        #
        #    The mkdir is the part that matters here. That reference targets Modal, whose
        #    run_command tolerates a missing workdir; podman does not -- `exec -w /root`
        #    on an image lacking it returns 127 without running anything (the container
        #    survives, so this would surface as a mysterious empty verdict rather than a
        #    crash). /root exists in every image sampled from this set and the containers
        #    run as root, so the mkdir always succeeds and costs one exec.
        env.execute("mkdir -p /root")
        obs = env.execute("bash /tests/test.sh", cwd="/root", timeout=int(timeout))
        logger.debug("verifier returncode=%s", obs.get("returncode"))

        # 6. The exit code is advisory; the reward file is authoritative.
        reward, source = _read_reward(env)
        result["reward"] = reward
        result["resolved"] = reward > 0
        result["reward_source"] = source
        # A fragile test.sh can exit 0 without ever scoring the run (failed dependency
        # install, unmet precondition). That is indistinguishable from "tests failed" in
        # the reward value alone, so only a real reward file counts as a verdict.
        result["grading_failed"] = not source.startswith("reward.")
        if result["grading_failed"]:
            result["eval_error"] = f"(truncated)\n{(obs.get('output') or '')[-1000:]}"
        return result
    except subprocess.TimeoutExpired:
        result["eval_error"] = f"verifier timed out after {timeout}s"
        return result
    except Exception as e:  # noqa: BLE001 - grading must never kill the rollout
        logger.warning("verifier failed for %s: %s", instance_id, e)
        result["eval_error"] = f"verifier error: {e}"
        return result


def _read_reward(env: Any) -> tuple[float, str]:
    text = _cat(env, REWARD_TXT)
    if text:
        try:
            return float(text), "reward.txt"
        except ValueError:
            logger.warning("unparseable reward.txt: %r", text[:100])

    raw = _cat(env, REWARD_JSON)
    if raw:
        try:
            return float(json.loads(raw).get("reward", 0.0)), "reward.json"
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("unparseable reward.json: %r", raw[:100])

    return 0.0, "no reward file written by test.sh"


def _cat(env: Any, path: str) -> str:
    try:
        obs = env.execute(f"cat {path} 2>/dev/null")
    except Exception:  # noqa: BLE001
        return ""
    if obs.get("returncode", 1) != 0:
        return ""
    return (obs.get("output") or "").strip()
