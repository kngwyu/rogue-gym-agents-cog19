import os
from rainy import Config
from rainy.agents import PpoAgent
import rainy.utils.cli as cli
import ppo_naturecnn
from net import impala_conv


AGENT = PpoAgent


def config() -> Config:
    c = ppo_naturecnn.config()
    c.set_net_fn('actor-critic', impala_conv())
    return c


if __name__ == '__main__':
    cli.run_cli(config(), AGENT, script_path=os.path.realpath(__file__))
