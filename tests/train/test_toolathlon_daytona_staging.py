import json
import tomllib
from pathlib import Path

import pytest

from examples.train.toolathlon_harbor.prepare_daytona_tasks import stage_tasks


def _write_task(root: Path, task_id: str = "sample") -> Path:
    task = root / task_id
    environment = task / "environment"
    (environment / "task").mkdir(parents=True)
    (environment / "task" / "initial_state.json").write_text("{}")
    (environment / "mcp.json").write_text(json.dumps({"mcpServers": {}}))
    (environment / "Dockerfile").write_text("FROM local-runtime\n")
    task.joinpath("instruction.md").write_text("Do the task")
    task.joinpath("task.toml").write_text(
        """schema_version = "1.4"

[metadata]
mcp_servers = ["emails", "word"]

[environment]
cpus = 2
docker_image = "old:image"
"""
    )
    return task


def test_stages_upload_only_task_with_shared_runtime(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_task(source)

    assert stage_tasks(source, output, "registry/runtime:v1") == 1

    staged = output / "sample"
    config = tomllib.loads((staged / "task.toml").read_text())
    assert config["environment"] == {
        "cpus": 2,
        "docker_image": "registry/runtime:v1",
        "workdir": "/opt",
        "env": {
            "T3_BUNDLE_DIR": "/opt/task",
            "T3_SERVERS": "emails,word",
            "T3_WORLD_DUMP": "/logs/world_after.json",
        },
    }
    assert not (staged / "environment" / "Dockerfile").exists()
    assert (staged / "environment" / "task" / "initial_state.json").exists()
    assert (staged / "environment" / "mcp.json").exists()


def test_existing_output_requires_force(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_task(source)
    output.mkdir()

    with pytest.raises(FileExistsError, match="--force"):
        stage_tasks(source, output, "registry/runtime:v1")
