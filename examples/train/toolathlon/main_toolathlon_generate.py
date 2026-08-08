"""
Main entrypoint for generating rollouts on Toolathlon tasks. For debugging purposes.
"""

import asyncio
import sys

import ray
from loguru import logger

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.generators.base import GeneratorInput, TrajectoryID
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from .dataset import ToolathlonTaskDataset
from .main_toolathlon import ToolathlonSkyRLConfig, load_toolathlon_config
from .toolathlon_generator import ToolathlonGenerator

# For debugging purposes, we only generate a few samples.
NUM_SAMPLES_TO_TEST = 4


class ToolathlonGenerateExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the ToolathlonGenerator."""
        return ToolathlonGenerator(
            generator_cfg=cfg.generator,
            toolathlon_cfg=cfg.toolathlon_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def _setup_generator(self):
        logger.info(self.get_cfg_as_str(self.cfg))

        inference_engine_client = self.get_inference_client()
        asyncio.run(inference_engine_client.wake_up())

        return self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            ToolathlonTaskDataset: The training dataset.
        """
        return ToolathlonTaskDataset(
            data_files=self.cfg.data.train_data,
            toolathlon_repo_path=self.cfg.toolathlon_config["repo_path"],
            tasks_domain=self.cfg.toolathlon_config["tasks_domain"],
        )

    def get_eval_dataset(self):
        return None

    def run(self):
        generator = self._setup_generator()

        prompts = []
        trajectory_ids = []
        for item in self.train_dataset:
            prompts.append(item["prompt"])
            trajectory_ids.append(TrajectoryID(instance_id=item["uid"], repetition_id=0))

        input_batch = GeneratorInput(
            prompts=prompts[:NUM_SAMPLES_TO_TEST],
            trajectory_ids=trajectory_ids[:NUM_SAMPLES_TO_TEST],
            env_classes=None,
            env_extras=None,
            sampling_params=None,
        )

        generator_output = asyncio.run(generator.generate(input_batch))
        logger.info(f"Rollout metrics: {generator_output['rollout_metrics']}")
        logger.info(f"Rewards: {generator_output['rewards']}")
        logger.info(f"Stop reasons: {generator_output['stop_reasons']}")


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = ToolathlonGenerateExp(cfg)
    exp.run()


def main() -> None:
    cfg = ToolathlonSkyRLConfig.from_cli_overrides(sys.argv[1:])
    load_toolathlon_config(cfg)

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for Toolathlon generation; "
            "it is required to truncate responses to the maximum allowed length."
        )
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
