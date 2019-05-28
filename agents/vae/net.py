from numpy import ndarray
from rainy.net import ActorCriticNet, LinearHead, DummyRnn
from rainy.net.init import Initializer, orthogonal
from rainy.net.policy import CategoricalHead, Policy
from rainy.utils import Device
import torch
from torch import nn, Tensor
from typing import List, NamedTuple, Tuple, Union

CNN_INIT = Initializer(nonlinearity='relu')


class VaeOutPut(NamedTuple):
    x: Tensor
    mu: Tensor
    logvar: Tensor


class Flatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, shape: Tuple[int, int]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), -1, *self.shape)


class VaeActorCriticNet(ActorCriticNet):
    def __init__(
            self,
            input_dim: Tuple[int, int, int],
            action_dim: int,
            conv_channels: List[int] = [32, 32, 32],
            conv_args: List[tuple] = [(4, 2, 1), (3, 1, 1), (3, 1, 1)],
            h_dim: int = 256,
            z_dim: int = 64,
            output_channels: int = 0,
            device: Device = Device(),
    ) -> None:
        super(ActorCriticNet, self).__init__()
        cnn_hidden = calc_cnn_hidden(conv_args, *input_dim[1:])
        conved = cnn_hidden[0] * cnn_hidden[1] * conv_channels[-1]
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim[0], conv_channels[0], *conv_args[0]),
            nn.ReLU(True),
            nn.Conv2d(conv_channels[0], conv_channels[1], *conv_args[1]),
            nn.ReLU(True),
            nn.Conv2d(conv_channels[1], conv_channels[2], *conv_args[2]),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(conved, h_dim),
            nn.ReLU(True),
        )
        self.z_fc = LinearHead(h_dim, z_dim)
        self.logvar_fc = LinearHead(h_dim, z_dim)
        self.actor = LinearHead(z_dim, action_dim, Initializer(weight_init=orthogonal(0.01)))
        self.critic = LinearHead(z_dim, 1)
        if output_channels == 0:
            output_channels = input_dim[0]
        self.decoder = nn.Sequential(
            LinearHead(z_dim, h_dim),
            nn.ReLU(True),
            LinearHead(h_dim, conved),
            nn.ReLU(True),
            UnFlatten(cnn_hidden),
            nn.ConvTranspose2d(conv_channels[2], conv_channels[1], *conv_args[2]),
            nn.ReLU(True),
            nn.ConvTranspose2d(conv_channels[1], conv_channels[0], *conv_args[1]),
            nn.ReLU(True),
            nn.ConvTranspose2d(conv_channels[0], output_channels, *conv_args[0]),
        )
        CNN_INIT(self.encoder)
        CNN_INIT(self.decoder)
        self.encoder = device.data_parallel(self.encoder)
        self.decoder = device.data_parallel(self.decoder)
        self.device = device
        self._state_dim = input_dim
        self.policy_head = CategoricalHead(action_dim=action_dim)
        self.to(device.unwrapped)
        self._rnn = DummyRnn()

    @property
    def recurrent_body(self) -> DummyRnn:
        return self._rnn
        
    @property
    def action_dim(self) -> int:
        return self.actor.output_dim

    @property
    def state_dim(self) -> Tuple[int, ...]:
        return self._state_dim

    def latent(self, x: Union[ndarray, Tensor]) -> Tensor:
        h = self.encoder(self.device.tensor(x))
        return self.z_fc(h)

    def value(self, states: Union[ndarray, Tensor]) -> Tensor:
        return self.critic(self.latent(states)).squeeze()

    def policy(self, states: Union[ndarray, Tensor]) -> Policy:
        return self.policy_head(self.actor(self.latent(states)))

    def p_and_v(self, states: Union[ndarray, Tensor]) -> Tuple[Policy, Tensor]:
        latent = self.latent(states)
        value = self.critic(latent).squeeze()
        policy = self.policy_head(self.actor(latent))
        return policy, value

    def encode(self, x: Union[ndarray, Tensor]) -> Tuple[Tensor, Tensor]:
        h = self.encoder(self.device.tensor(x))
        return self.z_fc(h), self.logvar_fc(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x: Union[ndarray, Tensor]) -> Tuple[VaeOutPut, Policy, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        policy, value = self.actor(mu), self.critic(mu).squeeze()
        return VaeOutPut(self.decoder(z), mu, logvar), self.policy_head(policy), value


def calc_cnn_hidden(params: List[tuple], width: int, height: int) -> Tuple[int, int]:
    for kernel, stride, padding in params:
        width = (width - kernel + 2 * padding) // stride + 1
        height = (height - kernel + 2 * padding) // stride + 1
    assert width > 0 and height > 0, 'Convolution makes dim < 0!!!'
    return width, height
