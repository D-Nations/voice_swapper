import math

import torch

from rvc.lib.algorithm.attentions import FFN, MultiHeadAttention
from rvc.lib.algorithm.commons import sequence_mask
from rvc.lib.algorithm.modules import WaveNet
from rvc.lib.algorithm.normalization import LayerNorm


class Encoder(torch.nn.Module):
    """Transformer encoder with relative position attention.

    Args:
        hidden_channels: Channels of the residual stream.
        filter_channels: Channels inside the feed-forward networks.
        n_heads: Attention heads.
        n_layers: Encoder layers.
        kernel_size: Kernel size of the feed-forward convolutions.
        p_dropout: Dropout probability.
        window_size: Window for relative position encoding.
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.drop = torch.nn.Dropout(p_dropout)

        self.attn_layers = torch.nn.ModuleList(
            [
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_layers_1 = torch.nn.ModuleList([LayerNorm(hidden_channels) for _ in range(n_layers)])
        self.ffn_layers = torch.nn.ModuleList(
            [
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm_layers_2 = torch.nn.ModuleList([LayerNorm(hidden_channels) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn, norm_1, ffn, norm_2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2, strict=True
        ):
            x = norm_1(x + self.drop(attn(x, x, attn_mask)))
            x = norm_2(x + self.drop(ffn(x, x_mask)))
        return x * x_mask


class TextEncoder(torch.nn.Module):
    """Encode content features and coarse pitch into the prior distribution of the latent audio.

    The name comes from VITS, where the input was text. In RVC it is speaker-neutral content
    features from an embedder such as ContentVec.

    Args:
        out_channels: Channels of the latent.
        hidden_channels: Channels inside the encoder.
        filter_channels: Channels inside the feed-forward networks.
        n_heads: Attention heads.
        n_layers: Encoder layers.
        kernel_size: Kernel size of the feed-forward convolutions.
        p_dropout: Dropout probability.
        embedding_dim: Size of the content features (768 for ContentVec).
        f0: Whether to add a coarse pitch embedding.
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        embedding_dim: int,
        f0: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.emb_phone = torch.nn.Linear(embedding_dim, hidden_channels)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        self.emb_pitch = torch.nn.Embedding(256, hidden_channels) if f0 else None
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, phone: torch.Tensor, pitch: torch.Tensor | None, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.emb_phone(phone)
        if pitch is not None and self.emb_pitch is not None:
            x += self.emb_pitch(pitch)
        x *= math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = x.transpose(1, -1)  # [batch, hidden, time]

        x_mask = sequence_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class PosteriorEncoder(torch.nn.Module):
    """Encode a linear spectrogram into a sample of the latent audio. Used only in training.

    Args:
        in_channels: Spectrogram bins.
        out_channels: Channels of the latent.
        hidden_channels: Channels inside the WaveNet.
        kernel_size: WaveNet kernel size.
        dilation_rate: WaveNet dilation rate.
        n_layers: WaveNet layers.
        gin_channels: Channels of the global conditioning input, or 0 for none.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.pre = torch.nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor, g: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask
