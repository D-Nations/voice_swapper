import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import init_weights
from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock


class SineGenerator(torch.nn.Module):
    """Sine waves at a pitch contour and its harmonics, with noise in unvoiced frames.

    Args:
        sampling_rate: Output sample rate in Hz.
        num_harmonics: Overtones to generate above the fundamental.
        sine_amplitude: Amplitude of the sine waves.
        noise_stddev: Standard deviation of the noise added in voiced frames.
        voiced_threshold: Frames with pitch above this, in Hz, count as voiced.
    """

    def __init__(
        self,
        sampling_rate: int,
        num_harmonics: int = 0,
        sine_amplitude: float = 0.1,
        noise_stddev: float = 0.003,
        voiced_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.sine_amplitude = sine_amplitude
        self.noise_stddev = noise_stddev
        self.voiced_threshold = voiced_threshold
        self.waveform_dim = num_harmonics + 1

    def _generate_sine_wave(self, f0: torch.Tensor, upsampling_factor: int) -> torch.Tensor:
        """Sine waves for f0 of shape [batch, frames, 1], upsampled from frames to samples."""
        batch_size = f0.shape[0]
        upsampling_grid = torch.arange(1, upsampling_factor + 1, dtype=f0.dtype, device=f0.device)

        # Phase advance per sample, carrying each frame's final phase into the next frame.
        phase_increments = (f0 / self.sampling_rate) * upsampling_grid
        phase_remainder = torch.fmod(phase_increments[:, :-1, -1:] + 0.5, 1.0) - 0.5
        cumulative_phase = phase_remainder.cumsum(dim=1).fmod(1.0).to(f0.dtype)
        phase_increments += F.pad(cumulative_phase, (0, 0, 1, 0), mode="constant")
        phase_increments = phase_increments.reshape(batch_size, -1, 1)

        harmonic_scale = torch.arange(1, self.waveform_dim + 1, dtype=f0.dtype, device=f0.device).reshape(1, 1, -1)
        phase_increments *= harmonic_scale

        # Random starting phase for the harmonics. The fundamental starts at zero.
        random_phase = torch.rand(1, 1, self.waveform_dim, device=f0.device)
        random_phase[..., 0] = 0
        phase_increments += random_phase

        return torch.sin(2 * math.pi * phase_increments)

    def forward(self, f0: torch.Tensor, upsampling_factor: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the excitation, the voiced mask, and the noise, each [batch, samples, harmonics]."""
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = self._generate_sine_wave(f0, upsampling_factor) * self.sine_amplitude
            voiced_mask = (f0 > self.voiced_threshold).float()
            voiced_mask = F.interpolate(
                voiced_mask.transpose(2, 1), scale_factor=float(upsampling_factor), mode="nearest"
            ).transpose(2, 1)
            noise_amplitude = voiced_mask * self.noise_stddev + (1 - voiced_mask) * (self.sine_amplitude / 3)
            noise = noise_amplitude * torch.randn_like(sine_waves)
            sine_waveforms = sine_waves * voiced_mask + noise
        return sine_waveforms, voiced_mask, noise


class SourceModuleHnNSF(torch.nn.Module):
    """Harmonic-plus-noise excitation for the neural source filter, merged to one channel.

    Args:
        sample_rate: Output sample rate in Hz.
        harmonic_num: Overtones to generate above the fundamental.
        sine_amp: Amplitude of the sine waves.
        add_noise_std: Standard deviation of the noise added in voiced frames.
        voiced_threshold: Frames with pitch above this, in Hz, count as voiced.
    """

    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ) -> None:
        super().__init__()
        self.l_sin_gen = SineGenerator(sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, upsample_factor: int = 1) -> torch.Tensor:
        sine_wavs, _, _ = self.l_sin_gen(x, upsample_factor)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        return self.l_tanh(self.l_linear(sine_wavs))


class HiFiGANNSFGenerator(torch.nn.Module):
    """HiFi-GAN vocoder driven by a neural source filter excitation built from the pitch contour.

    Args:
        initial_channel: Channels of the input latent.
        resblock_kernel_sizes: Kernel size of each residual block after an upsampling layer.
        resblock_dilation_sizes: Dilations of each of those residual blocks.
        upsample_rates: Upsampling factor of each layer. Their product is the hop length.
        upsample_initial_channel: Channels before the first upsampling layer. Halved by each layer.
        upsample_kernel_sizes: Kernel size of each upsampling layer.
        gin_channels: Channels of the global conditioning input, or 0 for none.
        sr: Output sample rate in Hz.
        checkpointing: Recompute activations in the backward pass to save memory.
    """

    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: Sequence[int],
        resblock_dilation_sizes: Sequence[Sequence[int]],
        upsample_rates: Sequence[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Sequence[int],
        gin_channels: int,
        sr: int,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.checkpointing = checkpointing
        self.upp = math.prod(upsample_rates)
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()
        channels = [upsample_initial_channel // (2 ** (i + 1)) for i in range(len(upsample_rates))]
        # The excitation is at the output rate, so each layer's noise convolution downsamples it
        # by the product of the upsampling still to come.
        stride_f0s = [math.prod(upsample_rates[i + 1 :]) for i in range(len(upsample_rates))]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            padding = (k - u) // 2 if u % 2 == 0 else u // 2 + u % 2
            self.ups.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i), channels[i], k, u, padding=padding, output_padding=u % 2
                    )
                )
            )
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            padding = 0 if stride == 1 else (kernel - stride) // 2
            self.noise_convs.append(torch.nn.Conv1d(1, channels[i], kernel_size=kernel, stride=stride, padding=padding))

        self.resblocks = torch.nn.ModuleList(
            [
                ResBlock(channels[i], k, d)
                for i in range(len(self.ups))
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True)
            ]
        )
        self.conv_post = torch.nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1) if gin_channels else None

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None and self.cond is not None:
            x = x + self.cond(g)

        use_checkpoints = self.training and self.checkpointing
        for i, (up, noise_conv) in enumerate(zip(self.ups, self.noise_convs, strict=True)):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = checkpoint(up, x, use_reentrant=False) if use_checkpoints else up(x)
            x = x + noise_conv(har_source)
            blocks = self.resblocks[i * self.num_kernels : (i + 1) * self.num_kernels]
            outputs = [checkpoint(block, x, use_reentrant=False) if use_checkpoints else block(x) for block in blocks]
            x = outputs[0]
            for output in outputs[1:]:
                x = x + output
            x = x / self.num_kernels

        x = F.leaky_relu(x)
        return torch.tanh(self.conv_post(x))
