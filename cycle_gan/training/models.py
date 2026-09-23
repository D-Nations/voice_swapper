from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from cycle_gan.config import CONFIG

# Generator architecture.
GENERATOR_CHANNELS = 64
NUM_RES_BLOCKS = 6
GENERATOR_KERNEL_SIZE = 7  # Kernel size of the first and last convolutions.
RES_BLOCK_KERNEL_SIZE = 3

# Discriminator architecture. Each entry is a stride-2 convolution that halves the input size.
DISCRIMINATOR_CHANNELS = (64, 128, 256, 512)
DISCRIMINATOR_KERNEL_SIZE = 4
LEAKY_RELU_SLOPE = 0.2


class VoiceDataset(Dataset):
    """Fixed-length segments cut from every .npy mel spectrogram in a folder.

    Each file is split into non-overlapping segments of segment_frames frames.
    A leftover tail shorter than one segment is dropped, as are whole files that
    are shorter than one segment.
    """

    def __init__(self, mel_spectrogram_path: str, segment_frames: int = CONFIG.data.segment_frames) -> None:
        self.mel_spectrogram_path = Path(mel_spectrogram_path)
        self.segment_frames = segment_frames
        self.files: list[Path] = sorted(self.mel_spectrogram_path.glob("*.npy"))
        self.segments: list[tuple[Path, int]] = []
        for file in self.files:
            num_frames = np.load(file, mmap_mode="r").shape[-1]
            for start in range(0, num_frames - segment_frames + 1, segment_frames):
                self.segments.append((file, start))

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return a (1, n_mels, segment_frames) tensor so batches are (batch, 1, n_mels, segment_frames)."""
        file, start = self.segments[index]
        mel_spectrogram = np.load(file, mmap_mode="r")[:, start : start + self.segment_frames]
        mel_spectrogram_tensor = torch.from_numpy(np.array(mel_spectrogram, dtype=np.float32))
        return mel_spectrogram_tensor.unsqueeze(0)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = RES_BLOCK_KERNEL_SIZE) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.InstanceNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        return out


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: int = NUM_RES_BLOCKS,
        channels: int = GENERATOR_CHANNELS,
        kernel_size: int = GENERATOR_KERNEL_SIZE,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.middle = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)],
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = DISCRIMINATOR_CHANNELS,
        kernel_size: int = DISCRIMINATOR_KERNEL_SIZE,
        leaky_relu_slope: float = LEAKY_RELU_SLOPE,
    ) -> None:
        super().__init__()

        # The first block has no normalization, as in the standard PatchGAN discriminator.
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, channels[0], kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
        ]
        for previous, current in pairwise(channels):
            layers += [
                nn.Conv2d(previous, current, kernel_size=kernel_size, stride=2, padding=1),
                nn.InstanceNorm2d(current),
                nn.LeakyReLU(leaky_relu_slope, inplace=True),
            ]
        layers.append(nn.Conv2d(channels[-1], 1, kernel_size=kernel_size, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
