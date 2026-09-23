import torch
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE

# RVC v2's periods. The pretrained discriminators expect exactly these.
PERIODS = (2, 3, 5, 7, 11, 17, 23, 37)

type FeatureMaps = list[torch.Tensor]


class MultiPeriodDiscriminator(torch.nn.Module):
    """A scale discriminator plus one period discriminator per entry of PERIODS.

    Args:
        use_spectral_norm: Use spectral instead of weight normalization.
        checkpointing: Recompute activations in the backward pass to save memory.
    """

    def __init__(self, use_spectral_norm: bool = False, checkpointing: bool = False) -> None:
        super().__init__()
        self.checkpointing = checkpointing
        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in PERIODS]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[FeatureMaps], list[FeatureMaps]]:
        """Score real audio y and generated audio y_hat with every discriminator.

        Returns the scores for real audio, the scores for generated audio, and the feature maps
        of each, one entry per discriminator.
        """
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            if self.training and self.checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(d, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """Discriminator over the raw waveform, with strided grouped convolutions."""

    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, FeatureMaps]:
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorP(torch.nn.Module):
    """Discriminator over the waveform folded into rows of `period` samples, so it sees periodic structure.

    Args:
        period: Samples per row.
        kernel_size: Kernel height of the convolutions.
        use_spectral_norm: Use spectral instead of weight normalization.
    """

    def __init__(self, period: int, kernel_size: int = 5, use_spectral_norm: bool = False) -> None:
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]
        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(in_ch, out_ch, (kernel_size, 1), (s, 1), padding=(get_padding(kernel_size, 1), 0))
                )
                for in_ch, out_ch, s in zip(in_channels, out_channels, strides, strict=True)
            ]
        )
        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, FeatureMaps]:
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            x = F.pad(x, (0, self.period - (t % self.period)), "reflect")
        x = x.view(b, c, -1, self.period)
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap
