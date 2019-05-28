from rainy import Config
from rogue_gym.envs import ImageSetting, RogueEnv, StairRewardEnv, StairRewardParallel
from rogue_gym.rainy_impls import ParallelRogueEnvExt, RogueEnvExt

CONFIG = {
    "width": 32,
    "height": 16,
    "seed_range": [0, 40],
    "hide_dungeon": True,
    "dungeon": {
        "style": "rogue",
        "room_num_x": 2,
        "room_num_y": 2,
    },
    "enemies": {
        "enemies": [],
    },
}


def set_env(config: Config, expand: ImageSetting) -> None:
    # ppo parameters
    config.nworkers = 32
    config.nsteps = 125
    config.value_loss_weight = 0.5
    config.entropy_weight = 0.01
    config.gae_tau = 0.95
    config.use_gae = True
    config.ppo_minibatch_size = 400
    config.ppo_clip = 0.1
    config.lr_decay = False
    config.set_parallel_env(lambda _env_gen, _num_w: ParallelRogueEnvExt(StairRewardParallel(
        [CONFIG] * config.nworkers,
        max_steps=500,
        stair_reward=50.0,
        image_setting=expand,
    )))
    config.eval_env = RogueEnvExt(StairRewardEnv(
        RogueEnv(
            config_dict=CONFIG,
            mex_steps=500,
            stair_reward=50.0,
            image_setting=expand
        ),
        100.0
    ))
    config.max_steps = int(2e7) * 2
    config.eval_freq = None
    config.save_freq = int(2e6)
