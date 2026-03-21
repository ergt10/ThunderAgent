"""
Main entrypoint for fully async Harbor training with ThunderAgent routing.

Pins the driver actor to a Ray head node resource so Harbor's Docker control
plane stays on the dedicated CPU node.
"""

import os
import sys

import ray
import yaml

from .main_harbor import HARBOR_DEFAULT_CONFIG, _deep_merge
from .main_harbor_fully_async import HarborFullyAsyncExp
from .main_harbor_thunder_agent_fully_async import HarborThunderAgentFullyAsyncExp, HarborThunderAgentSkyRLConfig
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray


def _thunderagent_enabled() -> bool:
    raw = os.environ.get("SKYRL_DISABLE_THUNDERAGENT", "")
    return raw.lower() not in {"1", "true", "yes", "on"}


@ray.remote(num_cpus=1, resources={"harbor_head": 0.001})
def skyrl_entrypoint(cfg):
    if _thunderagent_enabled():
        exp = HarborThunderAgentFullyAsyncExp(cfg)
    else:
        exp = HarborFullyAsyncExp(cfg)
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
