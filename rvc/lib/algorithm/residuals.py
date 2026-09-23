from collections.abc import Sequence

import torch
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from rvc.lib.algorithm.commons import get_padding, init_weights
from rvc.lib.algorithm.modules import WaveNet

LRELU_SLOPE = 0.1


def create_conv1d_layer(channels: int, kernel_size: int, dilation: int) -> torch.nn.Module:
    return weight_norm(
        torch.nn.Conv1d(
            channels, channels, kernel_size, 1, dilation=dilation, padding=get_padding(kernel_size, dilation)
        )
    )


def apply_mask(tensor: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return tensor * mask if mask is not None else tensor


class ResBlock(torch.nn.Module):
    """HiFi-GAN residual block: pairs of dilated and undilated convolutions, each pair with a skip connection.

    Args:
        channels: Input and output channels.
        kernel_size: Kernel size of every convolution.
        dilations: Dilation of the first convolution in each pair.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilations: Sequence[int] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Sequence[int]) -> torch.nn.ModuleList:
        layers = torch.nn.ModuleList([create_conv1d_layer(channels, kernel_size, d) for d in dilations])
        layers.apply(init_weights)
        return layers

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            x_residual = x
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = apply_mask(x, x_mask)
            x = F.leaky_relu(conv1(x), LRELU_SLOPE)
            x = apply_mask(x, x_mask)
            x = conv2(x)
            x = x + x_residual
        return apply_mask(x, x_mask)


class Flip(torch.nn.Module):
    """Reverse the channel order, so alternate coupling layers transform alternate halves."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, [1])


class ResidualCouplingLayer(torch.nn.Module):
    """Affine coupling layer of a normalizing flow: shifts half the channels by a function of the other half.

    Args:
        channels: Channels of the input. Must be even.
        hidden_channels: Channels of the WaveNet that computes the shift.
        kernel_size: WaveNet kernel size.
        dilation_rate: WaveNet dilation rate.
        n_layers: WaveNet layers.
        p_dropout: WaveNet dropout.
        gin_channels: Channels of the global conditioning input, or 0 for none.
        mean_only: Shift only, without scaling.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ) -> None:
        if channels % 2 != 0:
            raise ValueError("channels should be divisible by 2")
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = torch.nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels
        )
        self.post = torch.nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        # Start as the identity transform.
        self.post.weight.data.zero_()
        if self.post.bias is not None:
            self.post.bias.data.zero_()

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None, reverse: bool = False
    ) -> torch.Tensor:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if self.mean_only:
            m, logs = stats, torch.zeros_like(stats)
        else:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)

        if reverse:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
        else:
            x1 = m + x1 * torch.exp(logs) * x_mask
        return torch.cat([x0, x1], 1)


class ResidualCouplingBlock(torch.nn.Module):
    """Normalizing flow of coupling layers, flipping the channels after each one.

    Args:
        channels: Channels of the input.
        hidden_channels: Channels inside each coupling layer.
        kernel_size: Kernel size of the coupling layers.
        dilation_rate: Dilation rate of the coupling layers.
        n_layers: WaveNet layers per coupling layer.
        n_flows: Coupling layers.
        gin_channels: Channels of the global conditioning input, or 0 for none.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        # Coupling layers sit at even indices and flips at odd ones, which fixes the state dict keys.
        self.flows = torch.nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None, reverse: bool = False
    ) -> torch.Tensor:
        for flow in reversed(self.flows) if reverse else self.flows:
            x = flow(x, x_mask, g=g, reverse=reverse) if isinstance(flow, ResidualCouplingLayer) else flow(x)
        return x
