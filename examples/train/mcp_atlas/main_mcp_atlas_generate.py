"""
Main entrypoint for generating rollouts on MCP-Atlas tasks. For debugging purposes.
"""

import asyncio
import sys

import ray
from loguru import logger

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.generators.base import GeneratorInput, TrajectoryID
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from .main_mcp_atlas import MCPAtlasSkyRLConfig, load_mcp_atlas_config
from .mcp_atlas_generator import MCPAtlasGenerator

# For debugging purposes, we only generate a few samples.
NUM_SAMPLES_TO_TEST = 4


class MCPAtlasGenerateExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the MCPAtlasGenerator."""
        return MCPAtlasGenerator(
            generator_cfg=cfg.generator,
            atlas_cfg=cfg.mcp_atlas_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def _setup_generator(self):
        logger.info(self.get_cfg_as_str(self.cfg))

        inference_engine_client = self.get_inference_client()
        asyncio.run(inference_engine_client.wake_up())

        return self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

    def get_eval_dataset(self):
        return None

    def run(self):
        generator = self._setup_generator()

        prompts = []
        env_extras = []
        trajectory_ids = []
        for i in range(min(NUM_SAMPLES_TO_TEST, len(self.train_dataset))):
            messages, _, extras, uid = self.train_dataset[i]
            prompts.append(messages)
            env_extras.append(extras)
            trajectory_ids.append(TrajectoryID(instance_id=uid, repetition_id=0))

        input_batch = GeneratorInput(
            prompts=prompts,
            trajectory_ids=trajectory_ids,
            env_classes=None,
            env_extras=env_extras,
            sampling_params=None,
        )

        generator_output = asyncio.run(generator.generate(input_batch))
        logger.info(f"Rollout metrics: {generator_output['rollout_metrics']}")
        logger.info(f"Rewards: {generator_output['rewards']}")
        logger.info(f"Stop reasons: {generator_output['stop_reasons']}")


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = MCPAtlasGenerateExp(cfg)
    exp.run()


def main() -> None:
    cfg = MCPAtlasSkyRLConfig.from_cli_overrides(sys.argv[1:])
    load_mcp_atlas_config(cfg)

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for MCP-Atlas generation; "
            "it is required to truncate responses to the maximum allowed length."
        )
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
