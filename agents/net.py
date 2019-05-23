from rainy.net import actor_critic
from rainy.prelude import NetFn


def a2c_conv() -> NetFn:
    return actor_critic.ac_conv(
        output_dim=256,
        kernel_and_strides=[(8, 1), (4, 1), (3, 1)]
    )


def impala_conv() -> NetFn:
    return actor_critic.impala_conv(
        maxpools=[(3, 2, 1)] + [(3, 1, 1)] * 2,
        use_batch_norm=False
    )
