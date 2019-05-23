import click
from datetime import datetime
import numpy as np
from pathlib import Path
from eval_seeds import get_agent


@click.command()
@click.option('--logdir')
@click.option('--l', default=1000)
@click.option('--r', default=2000)
@click.option('--n', default=10)
def eval_cog(logdir: str, l: int, r: int, n: int) -> None:
    ag = get_agent(logdir)
    logdir = Path(logdir)
    save_files = [f for f in logdir.glob('rainy-agent.pth.*')]
    save_files.sort()
    start = datetime.now()
    for i, f in enumerate(save_files):
        ag.load(f)
        rewards = []
        result_file = logdir.joinpath('eval_seeds{}-{}.npy'.format(n, i))
        if result_file.exists():
            continue
        for s in range(l, r):
            ag.config.seed = s
            res = ag.eval_parallel(n=n)
            rewards.append([r.reward for r in res])
        rew = np.array(rewards)
        print('elapsed: ', (datetime.now() - start).total_seconds())
        print('file: {} reward sum: {}'.format(i, rew.sum()))
        np.save(result_file.as_posix(), rew)
    ag.close()


if __name__ == '__main__':
    eval_cog()
