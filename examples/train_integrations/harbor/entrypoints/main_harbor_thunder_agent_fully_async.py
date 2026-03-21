"""
Main entrypoint for fully async Harbor training with ThunderAgent routing.
"""

import sys
from dataclasses import dataclass, field

import ray
import yaml

from examples.train.thunder_agent.config import ThunderAgentInferenceEngineConfig
from examples.train.thunder_agent.main_thunder_agent import ThunderAgentExp
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from .main_harbor import HARBOR_DEFAULT_CONFIG, HarborGeneratorConfig, HarborSkyRLConfig, _deep_merge


@dataclass
class HarborThunderAgentGeneratorConfig(HarborGeneratorConfig):
    """Harbor generator config with ThunderAgent-aware inference settings."""

    inference_engine: ThunderAgentInferenceEngineConfig = field(default_factory=ThunderAgentInferenceEngineConfig)


@dataclass
class HarborThunderAgentSkyRLConfig(HarborSkyRLConfig):
    """Harbor config with ThunderAgent-aware inference engine settings."""

    generator: HarborThunderAgentGeneratorConfig = field(default_factory=HarborThunderAgentGeneratorConfig)


class HarborThunderAgentFullyAsyncExp(ThunderAgentExp):
    """Harbor fully-async experiment that routes inference through ThunderAgent."""

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        from ..harbor_generator import HarborGenerator

        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        from ..dataset import HarborTaskDataset

        prompts_dataset = HarborTaskDataset(
            data_files=self.cfg.data.train_data,
            max_tasks=self.cfg.max_train_tasks,
        )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        from ..dataset import HarborTaskDataset

        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return HarborTaskDataset(
                data_files=self.cfg.data.val_data,
                max_tasks=self.cfg.max_eval_tasks,
            )
        return None

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
    exp = HarborThunderAgentFullyAsyncExp(cfg)
    exp.run()


def main() -> None:
    cfg = HarborThunderAgentSkyRLConfig.from_cli_overrides(sys.argv[1:])

    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
