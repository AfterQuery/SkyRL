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


def test_launcher_keeps_toolathlon_out_of_generic_agent():
    generic = (
        (ROOT / "examples/train_integrations/harbor/mcp_agent.py").read_text().lower()
    )
    runner = (
        (ROOT / "examples/train_integrations/harbor/mcp_runner.py").read_text().lower()
    )
    bridge = (
        (ROOT / "examples/train_integrations/harbor/mcp_bridge.py").read_text().lower()
    )
    assert "toolathlon" not in generic
    assert "toolathlon" not in runner
    assert "toolathlon" not in bridge
