import click
from datetime import datetime
import numpy as np
from pathlib import Path
from eval_seeds import get_agent


@click.command()
@click.option('--logdir')
@click.option('--l', default=0)
@click.option('--r', default=10)
@click.option('--n', default=10)
def entropy(logdir: str, l: int, r: int, n: int) -> None:
    ag = get_agent(logdir)
    logdir = Path(logdir)
    save_files = [f for f in logdir.glob('rainy-agent.pth.*')]
    save_files.sort()
    start = datetime.now()
    for i, f in enumerate(save_files):
        ag.load(f)
        ent = np.zeros(10)
        for s in range(l, r):
            ag.config.seed = s
            _ = ag.eval_parallel(n=n, entropy=ent)
        print('elapsed: ', (datetime.now() - start).total_seconds())
        print('entropy: {}'.format(ent / (500 * 10)))
        np.save('{}/entropy-{}.npy'.format(logdir, i), ent)
    ag.close()


if __name__ == '__main__':
    entropy()
