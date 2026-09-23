import torch
from torch.nn.utils.parametrizations import weight_norm

from rvc.lib.algorithm.commons import fused_add_tanh_sigmoid_multiply


class WaveNet(torch.nn.Module):
    """WaveNet residual blocks as used in WaveGlow, with optional global conditioning.

    Args:
        hidden_channels: Channels of the residual stream.
        kernel_size: Convolution kernel size. Must be odd.
        dilation_rate: Layer i uses dilation dilation_rate**i.
        n_layers: Number of layers.
        gin_channels: Channels of the global conditioning input, or 0 for none.
        p_dropout: Dropout after each gated activation.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Kernel size must be odd for proper padding.")

        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = torch.nn.Dropout(p_dropout)

        if gin_channels:
            self.cond_layer = weight_norm(torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1))

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                weight_norm(
                    torch.nn.Conv1d(
                        hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding
                    )
                )
            )
            # The last layer only feeds the skip output, so it needs half the channels.
            res_skip_channels = hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            self.res_skip_layers.append(weight_norm(torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        output = torch.zeros_like(x)
        if g is not None:
            g = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers, strict=True)):
            x_in = in_layer(x)
            g_l = g[:, i * 2 * self.hidden_channels : (i + 1) * 2 * self.hidden_channels, :] if g is not None else 0.0
            acts = self.drop(fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels))

            res_skip_acts = res_skip_layer(acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, : self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts

        return output * x_mask
