"""Typed access to the model and training settings in the <sample rate>.json files."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self, TypedDict

from rvc.runtime import CONFIGS_DIR


class TrainSection(TypedDict):
    """The "train" section of a config file."""

    log_interval: int
    seed: int
    learning_rate: float
    betas: list[float]
    eps: float
    lr_decay: float
    segment_size: int
    c_mel: float
    c_kl: float


class DataSection(TypedDict):
    """The "data" section of a config file."""

    max_wav_value: float
    sample_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    n_mel_channels: int
    mel_fmin: float
    mel_fmax: float | None


class ModelSection(TypedDict):
    """The "model" section of a config file."""

    inter_channels: int
    hidden_channels: int
    filter_channels: int
    text_enc_hidden_dim: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    resblock: str
    resblock_kernel_sizes: list[int]
    resblock_dilation_sizes: list[list[int]]
    upsample_rates: list[int]
    upsample_initial_channel: int
    upsample_kernel_sizes: list[int]
    use_spectral_norm: bool
    gin_channels: int
    spk_embed_dim: int


class ConfigFile(TypedDict):
    """A <sample rate>.json config file. Applio's trainer may add other top-level keys, which are ignored."""

    train: TrainSection
    data: DataSection
    model: ModelSection


@dataclass(frozen=True)
class TrainConfig:
    log_interval: int
    seed: int
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    lr_decay: float
    segment_size: int  # Samples of audio per training slice.
    c_mel: float  # Weight of the mel spectrogram loss.
    c_kl: float  # Weight of the KL divergence loss.


@dataclass(frozen=True)
class DataConfig:
    max_wav_value: float
    sample_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    n_mel_channels: int
    mel_fmin: float
    mel_fmax: float | None

    @property
    def spec_channels(self) -> int:
        return self.filter_length // 2 + 1


@dataclass(frozen=True)
class ModelConfig:
    inter_channels: int
    hidden_channels: int
    filter_channels: int
    text_enc_hidden_dim: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    resblock: str
    resblock_kernel_sizes: list[int]
    resblock_dilation_sizes: list[list[int]]
    upsample_rates: list[int]
    upsample_initial_channel: int
    upsample_kernel_sizes: list[int]
    use_spectral_norm: bool
    gin_channels: int
    spk_embed_dim: int


@dataclass(frozen=True)
class RVCConfig:
    train: TrainConfig
    data: DataConfig
    model: ModelConfig

    @classmethod
    def from_dict(cls, values: ConfigFile) -> Self:
        train = values["train"]
        beta1, beta2 = train["betas"]
        return cls(
            train=TrainConfig(
                log_interval=train["log_interval"],
                seed=train["seed"],
                learning_rate=train["learning_rate"],
                betas=(beta1, beta2),
                eps=train["eps"],
                lr_decay=train["lr_decay"],
                segment_size=train["segment_size"],
                c_mel=train["c_mel"],
                c_kl=train["c_kl"],
            ),
            data=DataConfig(**values["data"]),
            model=ModelConfig(**values["model"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path, encoding="utf-8") as file:
            values: ConfigFile = json.load(file)
        return cls.from_dict(values)

    @classmethod
    def for_sample_rate(cls, sample_rate: int) -> Self:
        return cls.load(config_path(sample_rate))

    @property
    def segment_frames(self) -> int:
        """Spectrogram frames per training slice."""
        return self.train.segment_size // self.data.hop_length


def config_path(sample_rate: int) -> Path:
    path = CONFIGS_DIR / f"{sample_rate}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No model config for {sample_rate} Hz.")
    return path
