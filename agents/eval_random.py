import click
import importlib
import numpy as np
from datetime import datetime


@click.command()
@click.option('--module', default='ppo_32_noenem_nohist')
@click.option('--l', default=1000)
@click.option('--r', default=2000)
@click.option('--n', default=10)
def eval_random(module: str, l: int, r: int, n: int) -> None:
    m = importlib.import_module(module)
    ag = m.AGENT(m.config())
    rewards = []
    for i in range(l, r):
        ag.config.seed = i
        rewards.append([ag.random_episode().reward for _ in range(n)])
    ag.close()
    r = np.array(rewards)
    print('reward sum: ', r.sum())
    np.save('eval_random.npy', r)


if __name__ == '__main__':
    eval_random()
