from collections.abc import Sequence
from typing import Self

import torch

from rvc.configs.config import ModelConfig
from rvc.lib.algorithm.commons import rand_slice_segments, remove_weight_norm, slice_segments
from rvc.lib.algorithm.encoders import PosteriorEncoder, TextEncoder
from rvc.lib.algorithm.generators.hifigan_nsf import HiFiGANNSFGenerator
from rvc.lib.algorithm.residuals import ResidualCouplingBlock

# Scales the prior's noise at inference, trading variety for stability.
INFERENCE_NOISE_SCALE = 0.66666

type LatentStats = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class Synthesizer(torch.nn.Module):
    """RVC v2 voice model with pitch guidance: a VITS-style conditional VAE with a HiFi-GAN NSF decoder.

    Training encodes the target audio's spectrogram to a latent (enc_q), maps it through the flow
    toward the prior predicted from content features and pitch (enc_p), and decodes a random
    slice of the latent to audio (dec). Conversion samples from the prior, inverts the flow,
    and decodes.

    The positional arguments follow the "config" list saved in exported voice models, so a model
    can be rebuilt with Synthesizer(*config, use_f0=True, text_enc_hidden_dim=768).

    Args:
        spec_channels: Linear spectrogram bins.
        segment_size: Latent frames per training slice.
        inter_channels: Channels of the latent.
        hidden_channels: Channels inside the encoders and flow.
        filter_channels: Channels inside the text encoder's feed-forward networks.
        n_heads: Attention heads.
        n_layers: Text encoder layers.
        kernel_size: Kernel size of the text encoder's feed-forward convolutions.
        p_dropout: Dropout probability.
        resblock: Residual block type. Only "1" exists in RVC v2.
        resblock_kernel_sizes: Decoder residual block kernel sizes.
        resblock_dilation_sizes: Decoder residual block dilations.
        upsample_rates: Decoder upsampling factors.
        upsample_initial_channel: Decoder channels before upsampling.
        upsample_kernel_sizes: Decoder upsampling kernel sizes.
        spk_embed_dim: Number of speakers.
        gin_channels: Size of the speaker embedding.
        sr: Output sample rate in Hz.
        use_f0: Must be True. Models without pitch guidance are not supported.
        text_enc_hidden_dim: Size of the content features (768 for ContentVec).
        checkpointing: Recompute decoder activations in the backward pass to save memory.
    """

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: Sequence[int],
        resblock_dilation_sizes: Sequence[Sequence[int]],
        upsample_rates: Sequence[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Sequence[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool = True,
        text_enc_hidden_dim: int = 768,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if not use_f0:
            raise ValueError("Only models with pitch guidance are supported.")
        del resblock  # Part of the saved config, but every RVC v2 model uses the same block.
        self.segment_size = segment_size

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            text_enc_hidden_dim,
            f0=True,
        )
        self.dec = HiFiGANNSFGenerator(
            inter_channels,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
            checkpointing=checkpointing,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels
        )
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)

    @classmethod
    def from_config(
        cls, model: ModelConfig, spec_channels: int, segment_size: int, sample_rate: int, checkpointing: bool = False
    ) -> Self:
        return cls(
            spec_channels,
            segment_size,
            model.inter_channels,
            model.hidden_channels,
            model.filter_channels,
            model.n_heads,
            model.n_layers,
            model.kernel_size,
            model.p_dropout,
            model.resblock,
            model.resblock_kernel_sizes,
            model.resblock_dilation_sizes,
            model.upsample_rates,
            model.upsample_initial_channel,
            model.upsample_kernel_sizes,
            model.spk_embed_dim,
            model.gin_channels,
            sample_rate,
            text_enc_hidden_dim=model.text_enc_hidden_dim,
            checkpointing=checkpointing,
        )

    def remove_weight_norm(self) -> None:
        remove_weight_norm(self)

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        ds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, LatentStats]:
        """Training pass: decode a random slice of the latent of spectrogram y.

        Returns the generated audio slice, the slice start frames, the prior and posterior masks,
        and (z, z_p, m_p, logs_p, m_q, logs_q) for the KL loss.
        """
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
        o = self.dec(z_slice, pitchf, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        nsff0: torch.Tensor,
        sid: torch.Tensor,
        rate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Convert content features and pitch to audio in the voice of speaker sid.

        Args:
            phone: Content features, [batch, frames, text_enc_hidden_dim].
            phone_lengths: Frames per batch item.
            pitch: Coarse pitch bins, [batch, frames].
            nsff0: Pitch in Hz, [batch, frames].
            sid: Speaker ids, [batch].
            rate: If given, decode only this final fraction of the frames.

        Returns the audio, the frame mask, and (z, z_p, m_p, logs_p).
        """
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * INFERENCE_NOISE_SCALE) * x_mask

        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]
            nsff0 = nsff0[:, head:]

        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)
