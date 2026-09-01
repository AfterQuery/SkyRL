import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
ADAPTER = ROOT / "examples/train/toolathlon_harbor"


def test_mcp_config_starts_the_bundled_runtime():
    config = json.loads((ADAPTER / "mcp.json").read_text())
    server = config["mcp_servers"]["toolathlon"]
    assert server == {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "t3.mcp_server"],
    }


def test_skyrl_config_uses_generic_agent_and_task_resources():
    config = yaml.safe_load((ADAPTER / "harbor_trial_config.yaml").read_text())
    agent = config["agent"]
    assert agent["name"] is None
    assert agent["import_path"].endswith(":HarborMCPAgent")
    assert agent["kwargs"]["strict_rollout_details"] is True
    environment = config["environment"]
    assert environment["type"] == "docker"
    assert environment["override_cpus"] is None
    assert environment["override_memory_mb"] is None
    assert environment["override_storage_mb"] is None


def test_compute_config_keeps_agent_on_host_and_selects_compute():
    config = yaml.safe_load((ADAPTER / "harbor_compute_trial_config.yaml").read_text())
    assert config["agent"]["import_path"].endswith(":HarborMCPAgent")
    assert config["environment"] == {
        "type": "compute",
        "force_build": False,
        "delete": True,
        "override_cpus": None,
        "override_memory_mb": None,
        "override_storage_mb": None,
        "kwargs": {"reap_after_minutes": 180},
    }


def test_launchers_use_restored_bundle_layout_and_compute_environment():
    local_launcher = (ADAPTER / "run_eval.sh").read_text()
    compute_launcher = (ADAPTER / "run_compute_eval.sh").read_text()
    assert "toolathlon-tasks/eval_tasks}" in local_launcher
    assert "toolathlon-tasks/runtime/" in local_launcher
    assert "--platform linux/amd64 --load" in local_launcher
    assert '"$HERE/run_eval.sh" --env compute "$@"' in compute_launcher
    assert "COMPUTE_API_KEY" in compute_launcher


def test_launcher_keeps_toolathlon_out_of_generic_agent():
    generic = (ROOT / "examples/train_integrations/harbor/mcp_agent.py").read_text().lower()
    runner = (ROOT / "examples/train_integrations/harbor/mcp_runner.py").read_text().lower()
    bridge = (ROOT / "examples/train_integrations/harbor/mcp_bridge.py").read_text().lower()
    assert "toolathlon" not in generic
    assert "toolathlon" not in runner
    assert "toolathlon" not in bridge



def test_daytona_training_uses_shared_runtime_snapshot():
    config = yaml.safe_load(
        (ADAPTER / "harbor_daytona_training_config.yaml").read_text()
    )
    kwargs = config["environment"]["kwargs"]
    assert kwargs["snapshot_template_name"] == "toolathlon-json-runtime-v1"
    assert kwargs["auto_snapshot"] is False

    launcher = (ADAPTER / "run_grpo_qwen38_27b_2node.sh").read_text()
    assert "prepare_daytona_tasks.py" in launcher
    assert "TOOLATHLON_RUNTIME_IMAGE" in launcher
    assert "DAYTONA_SNAPSHOT_TEMPLATE" in launcher
    assert "Dockerfile" not in launcher
