"""
Fully-async entrypoint for training on Harbor tasks.

Reuses HarborExp's generator/dataset overrides and swaps in
``FullyAsyncRayPPOTrainer``. This is the moral equivalent of
``examples/train/fully_async/main_fully_async.py`` for harbor.
"""

import sys

import ray

from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from .main_harbor import HarborExp, HarborSkyRLConfig, load_harbor_trial_config


class HarborFullyAsyncExp(HarborExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = HarborFullyAsyncExp(cfg)
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
