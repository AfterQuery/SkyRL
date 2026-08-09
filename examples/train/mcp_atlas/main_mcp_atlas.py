"""
Main entrypoint for training on MCP-Atlas tasks.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import ray
import yaml

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from .mcp_atlas_generator import MCPAtlasGenerator

MCP_ATLAS_DEFAULT_CONFIG = Path(__file__).parent / "mcp_atlas_config.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge overrides into base dict recursively, modifying base in-place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class MCPAtlasSkyRLConfig(SkyRLTrainConfig):
    """SkyRLTrainConfig with the MCP-Atlas configuration."""

    mcp_atlas_config: Dict[str, Any] = field(default_factory=dict)


class MCPAtlasExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the MCPAtlasGenerator."""
        return MCPAtlasGenerator(
            generator_cfg=cfg.generator,
            atlas_cfg=cfg.mcp_atlas_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = MCPAtlasExp(cfg)
    exp.run()


def load_mcp_atlas_config(cfg: MCPAtlasSkyRLConfig) -> None:
    """Load MCP-Atlas defaults from YAML and merge CLI overrides on top."""
    with open(MCP_ATLAS_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.mcp_atlas_config = _deep_merge(defaults, cfg.mcp_atlas_config)


def main() -> None:
    cfg = MCPAtlasSkyRLConfig.from_cli_overrides(sys.argv[1:])
    load_mcp_atlas_config(cfg)

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for MCP-Atlas training; "
            "it is required to truncate responses to the maximum allowed length."
        )
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
