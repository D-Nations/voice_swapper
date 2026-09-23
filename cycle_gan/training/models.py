from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class VoiceDataset(Dataset):
    """Fixed-length segments cut from every .npy mel spectrogram in a folder.

    Each file is split into non-overlapping segments of segment_frames frames.
    A leftover tail shorter than one segment is dropped, as are whole files that
    are shorter than one segment.
    """

    def __init__(self, mel_spectrogram_path: str, segment_frames: int = 128):
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
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU(
            inplace=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )
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
        num_res_blocks: int = 6,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                padding=3,
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.middle = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)],
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                64,
                out_channels,
                kernel_size=7,
                padding=3,
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),
            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
