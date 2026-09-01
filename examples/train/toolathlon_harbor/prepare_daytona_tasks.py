"""Stage Toolathlon tasks for a shared Daytona runtime snapshot/image."""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from pathlib import Path


def _task_ids(source: Path, task_list: Path | None) -> list[str]:
    if task_list is None:
        return sorted(path.name for path in source.iterdir() if path.is_dir())
    ids = [
        line.strip()
        for line in task_list.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task IDs in {task_list}")
    return ids


def _rewrite_task_toml(path: Path, runtime_image: str) -> None:
    document = tomllib.loads(path.read_text())
    servers = document.get("metadata", {}).get("mcp_servers")
    if (
        not isinstance(servers, list)
        or not servers
        or not all(isinstance(item, str) for item in servers)
    ):
        raise ValueError(
            f"{path}: metadata.mcp_servers must be a non-empty string list"
        )

    lines = path.read_text().splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == "[environment]"), None
    )
    if start is None:
        raise ValueError(f"{path}: missing [environment] section")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")),
        len(lines),
    )
    retained = [
        line
        for line in lines[start + 1 : end]
        if not line.lstrip().startswith(("docker_image =", "workdir =", "env ="))
    ]
    while retained and not retained[-1].strip():
        retained.pop()

    q = json.dumps
    environment_vars = (
        "env = { "
        f"T3_BUNDLE_DIR = {q('/opt/task')}, "
        f"T3_SERVERS = {q(','.join(servers))}, "
        f"T3_WORLD_DUMP = {q('/logs/world_after.json')} "
        "}"
    )
    runtime = [
        f"docker_image = {q(runtime_image)}",
        'workdir = "/opt"',
        environment_vars,
    ]
    rewritten = lines[: start + 1] + retained + runtime + [""] + lines[end:]
    path.write_text("\n".join(rewritten).rstrip() + "\n")


def stage_tasks(
    source: Path,
    output: Path,
    runtime_image: str,
    *,
    task_list: Path | None = None,
    force: bool = False,
) -> int:
    if output.exists():
        if not force:
            raise FileExistsError(
                f"output already exists: {output}; pass --force to replace it"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)

    ids = _task_ids(source, task_list)
    for task_id in ids:
        source_task = source / task_id
        if not source_task.is_dir():
            raise FileNotFoundError(f"task not found: {source_task}")
        destination = output / task_id
        shutil.copytree(
            source_task,
            destination,
            ignore=shutil.ignore_patterns("Dockerfile", "Dockerfile.*"),
        )
        environment = destination / "environment"
        if not (environment / "task" / "initial_state.json").is_file():
            raise FileNotFoundError(f"missing task bundle: {environment / 'task'}")
        if not (environment / "mcp.json").is_file():
            raise FileNotFoundError(f"missing MCP config: {environment / 'mcp.json'}")
        _rewrite_task_toml(destination / "task.toml", runtime_image)

    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="source tasks directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="persistent staged tasks directory"
    )
    parser.add_argument(
        "--runtime-image", required=True, help="pullable fallback runtime image"
    )
    parser.add_argument(
        "--task-list", type=Path, help="optional file containing one task ID per line"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output directory"
    )
    args = parser.parse_args()
    count = stage_tasks(
        args.source.resolve(),
        args.output.resolve(),
        args.runtime_image,
        task_list=args.task_list.resolve() if args.task_list else None,
        force=args.force,
    )
    print(f"Staged {count} upload-only tasks at {args.output.resolve()}")


if __name__ == "__main__":
    main()
