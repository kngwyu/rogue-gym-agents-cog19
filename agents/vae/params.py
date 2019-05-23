from rainy import Config
from .loss import BetaVaeLoss


def patched_config() -> Config:
    config = Config()
    setattr(config, 'vae_loss', BetaVaeLoss(beta=4.0, decoder_type='bernoulli'))
    setattr(config, 'vae_loss_weight', 1.0)
    return config
