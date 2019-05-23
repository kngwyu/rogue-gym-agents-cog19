from rainy.utils import Device
import torch
from torch import nn
import pytest
from .net import VaeActorCriticNet


ACTION_DIM = 10
BATCH_SIZE = 64


@pytest.mark.parametrize('net_gen, input_dim', [
    (VaeActorCriticNet, (17, 32, 16)),
])
def test_dim(net_gen: callable, input_dim: tuple) -> None:
    device = Device()
    vae_net = net_gen(input_dim, ACTION_DIM, device=device)
    batch = torch.randn(BATCH_SIZE,  *input_dim)
    with torch.no_grad():
        vae, policy, value = vae_net(device.tensor(batch))
    assert vae.x.shape == torch.Size((BATCH_SIZE, *input_dim))
    assert policy.dist.probs.shape == torch.Size((BATCH_SIZE, ACTION_DIM))
    assert value.shape == torch.Size((BATCH_SIZE,))
    print(vae_net)
