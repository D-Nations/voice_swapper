"""RMVPE pitch tracking (Wei et al., 2023): a U-Net and GRU that predict pitch from a mel spectrogram, robust to noise."""

from pathlib import Path

import numpy as np
import torch
from librosa.filters import mel
from torch import nn
from torch.nn import functional as F

from rvc.runtime import MODELS_DIR

MODEL_PATH = MODELS_DIR / "predictors" / "rmvpe.pt"
SAMPLE_RATE = 16000
HOP_LENGTH = 160  # 10 ms frames
N_MELS = 128
N_CLASS = 360  # Pitch bins, 20 cents apart.
CHUNK_FRAMES = 32000  # Frames per model call, to bound memory on long inputs.


class ConvBlockRes(nn.Module):
    """Two 3x3 convolutions with batch norm, plus a residual connection."""

    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1)) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + (self.shortcut(x) if self.shortcut is not None else x)


class ResEncoderBlock(nn.Module):
    """A stack of residual blocks, with an optional average pool applied by the caller."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | None,
        n_blocks: int = 1,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.conv = nn.ModuleList(
            [ConvBlockRes(in_channels, out_channels, momentum)]
            + [ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)]
        )
        self.pool = nn.AvgPool2d(kernel_size=kernel_size) if kernel_size is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv:
            x = block(x)
        return x


class Encoder(nn.Module):
    """The U-Net's downsampling half. Returns the bottleneck and each level's output for the skip connections."""

    def __init__(
        self,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size: tuple[int, int],
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        x = self.bn(x)
        for layer in self.layers:
            assert isinstance(layer, ResEncoderBlock) and layer.pool is not None
            skip = layer(x)
            skips.append(skip)
            x = layer.pool(skip)
        return x, skips


class Intermediate(nn.Module):
    """The U-Net's bottleneck."""

    def __init__(
        self, in_channels: int, out_channels: int, n_inters: int, n_blocks: int, momentum: float = 0.01
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)]
            + [ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum) for _ in range(n_inters - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    """Upsample, concatenate the matching skip connection, and refine with residual blocks."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: tuple[int, int], n_blocks: int = 1, momentum: float = 0.01
    ) -> None:
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList(
            [ConvBlockRes(out_channels * 2, out_channels, momentum)]
            + [ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)]
        )

    def forward(self, x: torch.Tensor, concat_tensor: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.conv1(x), concat_tensor), dim=1)
        for block in self.conv2:
            x = block(x)
        return x


class Decoder(nn.Module):
    """The U-Net's upsampling half."""

    def __init__(
        self, in_channels: int, n_decoders: int, stride: tuple[int, int], n_blocks: int, momentum: float = 0.01
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for layer, skip in zip(self.layers, reversed(skips), strict=True):
            x = layer(x, skip)
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int],
        n_blocks: int,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips = self.encoder(x)
        x = self.intermediate(x)
        return self.decoder(x, skips)


class BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


class E2E(nn.Module):
    """Mel spectrogram in, per-frame salience over the N_CLASS pitch bins out."""

    def __init__(
        self,
        n_blocks: int,
        n_gru: int,
        kernel_size: tuple[int, int],
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ) -> None:
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru), nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)


class MelSpectrogram(nn.Module):
    """Log mel spectrogram with the settings RMVPE was trained on."""

    def __init__(
        self,
        n_mel_channels: int = N_MELS,
        sample_rate: int = SAMPLE_RATE,
        win_length: int = 1024,
        hop_length: int = HOP_LENGTH,
        mel_fmin: float = 30,
        mel_fmax: float = 8000,
        clamp: float = 1e-5,
    ) -> None:
        super().__init__()
        mel_basis = mel(sr=sample_rate, n_fft=win_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        self.mel_basis: torch.Tensor
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.window: torch.Tensor
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.win_length = win_length
        self.hop_length = hop_length
        self.clamp = clamp

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        fft = torch.stft(
            audio,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        return torch.log(torch.clamp(torch.matmul(self.mel_basis, magnitude), min=self.clamp))


class RMVPE:
    """Pitch tracker. get_f0 turns 16 kHz audio into pitch in Hz every 10 ms, with 0 for unvoiced frames."""

    def __init__(self, device: str | torch.device, model_path: Path = MODEL_PATH) -> None:
        self.device = device
        self.model = E2E(4, 1, (2, 2))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval().to(device)
        self.mel_extractor = MelSpectrogram().to(device)
        # Cents of each pitch bin, padded to match the 9-bin window in _local_average_cents.
        self.cents_mapping = np.pad(20 * np.arange(N_CLASS) + 1997.3794084376191, (4, 4))

    def get_f0(self, audio: np.ndarray, threshold: float = 0.03) -> np.ndarray:
        """Pitch in Hz per 10 ms frame. Frames whose best bin has salience at or below threshold are unvoiced (0)."""
        with torch.inference_mode():
            mel_spec = self.mel_extractor(torch.from_numpy(audio).float().to(self.device).unsqueeze(0))
            salience = self._salience(mel_spec).squeeze(0).cpu().numpy()
        cents = self._local_average_cents(salience, threshold)
        f0 = 10 * (2 ** (cents / 1200))
        f0[f0 == 10] = 0
        return f0

    def _salience(self, mel_spec: torch.Tensor) -> torch.Tensor:
        n_frames = mel_spec.shape[-1]
        # The U-Net halves the time axis 5 times, so pad to a multiple of 32 frames.
        mel_spec = F.pad(mel_spec, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect")
        chunks = [
            self.model(mel_spec[..., start : start + CHUNK_FRAMES])
            for start in range(0, mel_spec.shape[-1], CHUNK_FRAMES)
        ]
        return torch.cat(chunks, dim=1)[:, :n_frames]

    def _local_average_cents(self, salience: np.ndarray, threshold: float) -> np.ndarray:
        """Salience-weighted average pitch, in cents, over the 9 bins around each frame's peak."""
        center = np.argmax(salience, axis=1) + 4
        salience = np.pad(salience, ((0, 0), (4, 4)))
        idx = center[:, None] + np.arange(-4, 5)[None, :]
        local_salience = salience[np.arange(salience.shape[0])[:, None], idx]
        cents = np.sum(local_salience * self.cents_mapping[idx], axis=1) / np.sum(local_salience, axis=1)
        cents[np.max(salience, axis=1) <= threshold] = 0
        return cents
