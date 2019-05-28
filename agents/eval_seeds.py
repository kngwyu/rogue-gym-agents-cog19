import click
from datetime import datetime
import numpy as np
from pathlib import Path
import importlib


def get_agent(logdir: str, nworkers: int = 10):
    module = logdir[: logdir.find('-')]
    module = module[module.rfind('/') + 1:]
    m = importlib.import_module(module)
    c = m.config()
    c.nworkers = nworkers
    ag = m.AGENT(c)
    return ag


@click.command()
@click.option('--logdir')
@click.option('--l', default=1000)
@click.option('--r', default=2000)
@click.option('--n', default=10)
def eval_seeds(logdir: str, l: int, r: int, n: int) -> None:
    ag = get_agent(logdir)
    logdir = Path(logdir)
    ag.load(logdir.joinpath('rainy-agent.pth').as_posix())
    rewards = []
    start = datetime.now()
    for i in range(l, r):
        ag.config.seed = i
        res = ag.eval_parallel(n=n)
        rewards.append([r.reward for r in res])
    ag.close()
    r = np.array(rewards)
    print('elapsed: ', (datetime.now() - start).total_seconds())
    print('reward sum: ', r.sum())
    np.save('{}/eval_seeds{}.npy'.format(logdir, n), r)


if __name__ == '__main__':
    eval_seeds()
