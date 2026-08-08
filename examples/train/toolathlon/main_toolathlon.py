"""
Main entrypoint for training on Toolathlon tasks.
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

from .dataset import ToolathlonTaskDataset
from .toolathlon_generator import ToolathlonGenerator

TOOLATHLON_DEFAULT_CONFIG = Path(__file__).parent / "toolathlon_config.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge overrides into base dict recursively, modifying base in-place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class ToolathlonSkyRLConfig(SkyRLTrainConfig):
    """SkyRLTrainConfig with the Toolathlon harness configuration."""

    toolathlon_config: Dict[str, Any] = field(default_factory=dict)


class ToolathlonExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the ToolathlonGenerator."""
        return ToolathlonGenerator(
            generator_cfg=cfg.generator,
            toolathlon_cfg=cfg.toolathlon_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            ToolathlonTaskDataset: The training dataset.
        """
        prompts_dataset = ToolathlonTaskDataset(
            data_files=self.cfg.data.train_data,
            toolathlon_repo_path=self.cfg.toolathlon_config["repo_path"],
            tasks_domain=self.cfg.toolathlon_config["tasks_domain"],
        )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            ToolathlonTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return ToolathlonTaskDataset(
                data_files=self.cfg.data.val_data,
                toolathlon_repo_path=self.cfg.toolathlon_config["repo_path"],
                tasks_domain=self.cfg.toolathlon_config["tasks_domain"],
            )
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = ToolathlonExp(cfg)
    exp.run()


def load_toolathlon_config(cfg: ToolathlonSkyRLConfig) -> None:
    """Load Toolathlon defaults from YAML and merge CLI overrides on top."""
    with open(TOOLATHLON_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.toolathlon_config = _deep_merge(defaults, cfg.toolathlon_config)


def main() -> None:
    cfg = ToolathlonSkyRLConfig.from_cli_overrides(sys.argv[1:])
    load_toolathlon_config(cfg)

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for Toolathlon training; "
            "it is required to truncate responses to the maximum allowed length."
        )
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
