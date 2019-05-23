import click
from datetime import datetime
import numpy as np
from pathlib import Path
from eval_seeds import get_agent


@click.command()
@click.option('--logdir')
@click.option('--seed')
@click.option('--n', default=100)
@click.option('--random', is_flag=True)
def eval_1seed(logdir: str, seed: int, n: int, random: bool) -> None:
    ag = get_agent(logdir)
    logdir = Path(logdir)
    ag.config.seed = int(seed)
    start = datetime.now()
    if not random:
        ag.load(logdir.joinpath('rainy-agent.save').as_posix())
        res = ag.eval_parallel(n=n)
        rewards = [r.reward for r in res]
    else:
        rewards = [ag.random_episode().reward for _ in range(n)]
    ag.close()
    r = np.array(rewards)
    print('elapsed: ', (datetime.now() - start).total_seconds())
    print('reward sum: ', r.sum())
    np.save('{}/eval_1seed{}.npy'.format(logdir, n), r)


if __name__ == '__main__':
    eval_1seed()
