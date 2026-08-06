"""SkyRL entrypoint: mini-swe-agent on Harbor-format tasks.

Identical in shape to ``examples/train/mini_swe_agent/main_mini_swe.py`` -- only the
generator class and its config differ.
"""

import sys

import ray
from skyrl.train.config import SkyRLGymConfig, make_config
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray

from .harbor_generator import MiniSweAgentHarborGenerator, MiniSWEHarborGeneratorConfig

MiniSWEHarborConfig = make_config(generator_cls=MiniSWEHarborGeneratorConfig)


class MiniSWEHarborPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return MiniSweAgentHarborGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=SkyRLGymConfig(max_env_workers=0),
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=self.cfg.trainer.policy.model.path,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = MiniSWEHarborPPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = MiniSWEHarborConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
