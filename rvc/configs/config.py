"""Typed access to the model and training settings in the <sample rate>.json files."""

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from rvc.runtime import CONFIGS_DIR

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def _from_dict[T: DataclassInstance](cls: type[T], values: dict[str, Any]) -> T:
    """Build a dataclass from a dict, ignoring keys it doesn't define."""
    names = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in names})


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
    def from_dict(cls, values: dict[str, Any]) -> Self:
        train = dict(values["train"], betas=tuple(values["train"]["betas"]))
        return cls(
            train=_from_dict(TrainConfig, train),
            data=_from_dict(DataConfig, values["data"]),
            model=_from_dict(ModelConfig, values["model"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path, encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

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
