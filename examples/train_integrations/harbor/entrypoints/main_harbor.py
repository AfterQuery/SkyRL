"""
Main entrypoint for training on Harbor tasks.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ray
import yaml

from skyrl.train.config import GeneratorConfig, SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.rate_limiter import RateLimiterConfig
from skyrl.train.utils.utils import initialize_ray

from ..dataset import HarborTaskDataset
from ..harbor_generator import HarborGenerator

# NOTE (sumanthrh): We use a YAML to store the defaults for the Harbor trial configuration
# TODO: Convert to a dataclass
HARBOR_DEFAULT_CONFIG = Path(__file__).parent.parent / "harbor_trial_config" / "default.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge overrides into base dict recursively, modifying base in-place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml_mapping(path: Path, description: str) -> dict:
    """Load a YAML mapping, with a useful error for invalid trial configs."""
    with path.open() as f:
        value = yaml.safe_load(f)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{description} must contain a YAML mapping, got {type(value).__name__}: {path}")
    return value


def load_harbor_trial_config(cfg: "HarborSkyRLConfig") -> None:
    """Merge Harbor defaults, an optional config file, and CLI overrides."""
    if not isinstance(cfg.harbor_trial_config, dict):
        raise TypeError(
            "harbor_trial_config must be a mapping of CLI overrides. "
            "Use harbor_trial_config_path=/path/to/config.yaml to load a YAML file."
        )

    merged = _load_yaml_mapping(HARBOR_DEFAULT_CONFIG, "Harbor default config")
    if cfg.harbor_trial_config_path:
        config_path = Path(cfg.harbor_trial_config_path).expanduser()
        merged = _deep_merge(merged, _load_yaml_mapping(config_path, "Harbor trial config"))
    cfg.harbor_trial_config = _deep_merge(merged, cfg.harbor_trial_config)


@dataclass
class HarborGeneratorConfig(GeneratorConfig):
    """GeneratorConfig with Harbor-specific rate limiting."""

    rate_limit: RateLimiterConfig = field(default_factory=RateLimiterConfig)



@dataclass
class HarborSkyRLConfig(SkyRLTrainConfig):
    """SkyRLTrainConfig with Harbor trial configuration."""

    harbor_trial_config: dict[str, Any] = field(default_factory=dict)
    harbor_trial_config_path: str | None = None
    generator: HarborGeneratorConfig = field(default_factory=HarborGeneratorConfig)


class HarborExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the HarborGenerator.
        """
        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,  # Pass harbor config to the generator
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            HarborTaskDataset: The training dataset.
        """
        prompts_dataset = HarborTaskDataset(
            data_files=self.cfg.data.train_data,
        )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            HarborTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = HarborTaskDataset(
                data_files=self.cfg.data.val_data,
            )
            return prompts_dataset
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = HarborExp(cfg)
    exp.run()


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])

    load_harbor_trial_config(cfg)

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for Harbor training; "
            "it is required to truncate responses to the maximum allowed length."
        )
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
