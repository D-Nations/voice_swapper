import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import List


class VoiceDataset(Dataset):
    def __init__(self, mel_spectrogram_path: str):
        self.mel_spectrogram_path = Path(mel_spectrogram_path)
        self.files: List[Path] = sorted(
            list(self.mel_spectrogram_path.glob("*.npy")),
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        mel_spectrogram_file = self.files[idx]
        mel_spectrogram = np.load(mel_spectrogram_file)
        mel_spectrogram_tensor = torch.from_numpy(mel_spectrogram)
        return mel_spectrogram_tensor


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(
            in_channels,
        )
        self.relu = nn.ReLU(
            inplace=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(
            in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: int = 6,
    ):
        super(Generator, self).__init__()

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
        super(Discriminator, self).__init__()

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
