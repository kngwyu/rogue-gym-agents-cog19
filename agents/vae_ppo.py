import os
from rainy import Config
from rainy.utils import cli, Device
from rogue_gym.envs import ImageSetting, StatusFlag, DungeonType
from torch.optim import Adam
from env import set_env
import vae


EXPAND = ImageSetting(dungeon=DungeonType.SYMBOL, status=StatusFlag.EMPTY)
AGENT = vae.VaePpoAgent


def net(input_dim: tuple, action_dim: int, device: Device) -> vae.VaeActorCriticNet:
    return vae.VaeActorCriticNet(
        input_dim,
        action_dim,
        conv_channels=[32, 64, 32],
        conv_args=[(8, 1, 1), (4, 1, 1), (3, 1, 1)],
        h_dim=256,
        z_dim=64,
        device=device,
    )


def config() -> Config:
    c = vae.patched_config()
    # vae parameters
    c.vae_loss_weight = 1.0
    c.vae_loss = vae.BetaVaeLoss(beta=4.0, decoder_type='categorical_binary')
    set_env(c, EXPAND)
    c.set_optimizer(lambda params: Adam(params, lr=2.5e-4, eps=1.0e-4))
    c.set_net_fn('actor-critic', net)
    c.grad_clip = 0.5
    c.episode_log_freq = 100
    c.eval_deterministic = False
    c.network_log_freq = 100
    return c


if __name__ == '__main__':
    cli.run_cli(config(), AGENT, script_path=os.path.realpath(__file__))
