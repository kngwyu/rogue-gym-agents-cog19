import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Callable, Tuple
from .net import VaeOutPut


def bernoulli_recons(a: Tensor, b: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(a, b, reduction='sum')


def categorical_binary(a: Tensor, b: Tensor) -> Tensor:
    assert a.size(1) == b.size(1)
    value = b.max(dim=1, keepdim=True)[1]
    logits = a - a.logsumexp(dim=1, keepdim=True)
    log_probs = torch.gather(logits, 1, value)
    return -log_probs.sum()


def categorical_gray(a: Tensor, b: Tensor) -> Tensor:
    assert b.size(1) == 1
    categ = a.size(1)
    value = (b * float(categ)).round().long()
    logits = a - a.logsumexp(dim=1, keepdim=True)
    log_probs = torch.gather(logits, 1, value)
    return -log_probs.sum()


# Only for debugging categorical_gray
def __check_categ_loss(a: Tensor, b: Tensor) -> Tensor:
    assert b.size(1) == 1
    categ = a.size(1)
    value = (b * float(categ)).round().long()
    s = torch.zeros(1)
    for i in range(a.size(0)):
        logits = a[i].view(a[i].size(0), -1).transpose(1, 0)
        d = torch.distributions.Categorical(logits=logits)
        s += d.log_prob(value[i].view(value[i].size(0), -1)).sum()
    return s


def gaussian_recons(a: Tensor, b: Tensor) -> Tensor:
    return F.mse_loss(torch.sigmoid(a), b, reduction='sum')


def _recons_fn(decoder_type: str = 'bernoulli') -> Callable[[Tensor, Tensor], Tensor]:
    if decoder_type == 'bernoulli':
        recons_loss = bernoulli_recons
    elif decoder_type == 'gaussian':
        recons_loss = gaussian_recons
    elif decoder_type == 'categorical_gray':
        recons_loss = categorical_gray
    elif decoder_type == 'categorical_binary':
        recons_loss = categorical_binary
    else:
        raise ValueError('Currently only bernoulli and gaussian are supported as decoder head')
    return recons_loss


class BetaVaeLoss:
    def __init__(self, beta: float = 4.0, decoder_type: str = 'bernoulli') -> None:
        self.recons_loss = _recons_fn(decoder_type)
        self.beta = beta

    def __call__(self, res: VaeOutPut, img: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = float(img.size(0))
        recons = self.recons_loss(res.x, img).div_(batch_size)
        kld = -0.5 * \
            torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp()).div_(batch_size)
        return recons, self.beta * kld


class GammaVaeLoss:
    def __init__(
            self,
            gamma: float = 100.0,
            capacity_start: float = 0.0,
            capacity_max: float = 20.0,
            num_epochs: int = 25000,
            decoder_type: str = 'bernoulli',
    ) -> None:
        self.gamma = gamma
        self.recons_loss = _recons_fn(decoder_type)
        self.capacity = capacity_start
        self.delta = (capacity_max - capacity_start) / float(num_epochs)
        self.capacity_max = capacity_max

    def update(self) -> None:
        self.capacity = min(self.capacity_max, self.capacity + self.delta)

    def __call__(self, res: VaeOutPut, img: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = float(img.size(0))
        recons = self.recons_loss(res.x, img).div_(batch_size)
        kld = -0.5 * \
            torch.sum(1.0 + res.logvar - res.mu.pow(2.0) - res.logvar.exp()).div_(batch_size)
        latent = self.gamma * (kld - self.capacity).abs()
        return recons, latent
