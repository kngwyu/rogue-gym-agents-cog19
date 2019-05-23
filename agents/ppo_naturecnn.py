import os
from rainy import Config
from rainy.agents import PpoAgent
import rainy.utils.cli as cli
from rogue_gym.envs import ImageSetting, StatusFlag, DungeonType
from torch.optim import Adam
from env import set_env
from net import a2c_conv


EXPAND = ImageSetting(dungeon=DungeonType.SYMBOL, status=StatusFlag.EMPTY)
AGENT = PpoAgent


def config() -> Config:
    c = Config()
    set_env(c, EXPAND)
    c.set_optimizer(lambda params: Adam(params, lr=2.5e-4, eps=1.0e-4))
    c.set_net_fn('actor-critic', a2c_conv())
    c.grad_clip = 0.5
    c.episode_log_freq = 100
    c.eval_deterministic = False
    return c


if __name__ == '__main__':
    cli.run_cli(config(), AGENT, script_path=os.path.realpath(__file__))
