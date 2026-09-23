import datetime
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch

from rvc.configs.config import ModelConfig, RVCConfig
from rvc.runtime import MODEL_AUTHOR
from rvc.train.model_info import read_model_info
from rvc.train.utils import to_legacy_names

VERSION = "v2"
VOCODER = "HiFi-GAN"

# One of Synthesizer's positional arguments, as saved in a voice model's "config" list.
type ConfigEntry = int | float | str | list[int] | list[list[int]]


class VoiceModel(TypedDict):
    """An exported voice model file, in the format Applio and other RVC tools read."""

    weight: dict[str, torch.Tensor]  # fp16 generator weights, without the posterior encoder.
    config: list[ConfigEntry]
    epoch: int
    step: int
    sr: int
    f0: bool
    version: str
    creation_date: str
    model_hash: str
    dataset_length: str | None
    model_name: str
    author: str | None
    embedder_model: str
    speakers_id: int
    vocoder: str


def saved_config(config: RVCConfig) -> list[ConfigEntry]:
    """The "config" list saved in a voice model: Synthesizer's positional arguments, in order."""
    data, model = config.data, config.model
    return [
        data.spec_channels,
        32,  # Segment size, which only training uses.
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
        data.sample_rate,
    ]


def export_voice_model(
    generator_state: Mapping[str, torch.Tensor],
    config: RVCConfig,
    name: str,
    model_path: Path,
    epoch: int,
    step: int,
) -> None:
    """Save the small fp16 voice model that conversion needs, in the format Applio and other RVC tools read.

    Drops the posterior encoder (enc_q), which only training uses. The "config" list gives
    Synthesizer's positional arguments.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    info = read_model_info(model_path.parent)
    data = config.data
    model_config = saved_config(config)
    hash_input = f"{name}-{epoch}-{step}-{data.sample_rate}-{VERSION}-{model_config}"
    weights = {key: value.half() for key, value in generator_state.items() if "enc_q" not in key}
    exported: VoiceModel = {
        "weight": to_legacy_names(weights),
        "config": model_config,
        "epoch": epoch,
        "step": step,
        "sr": data.sample_rate,
        "f0": True,
        "version": VERSION,
        "creation_date": datetime.datetime.now().astimezone().isoformat(),
        "model_hash": hashlib.sha256(hash_input.encode()).hexdigest(),
        "dataset_length": info.get("total_dataset_duration"),
        "model_name": name,
        "author": MODEL_AUTHOR,
        "embedder_model": info.get("embedder_model", "contentvec"),
        "speakers_id": info.get("speakers_id", 1),
        "vocoder": VOCODER,
    }
    torch.save(exported, model_path)
    print(f"Saved voice model '{model_path}' (epoch {epoch}, step {step})")


@dataclass(frozen=True)
class SavedModelConfig:
    """What a voice model's "config" list says about its architecture."""

    model: ModelConfig
    spec_channels: int
    sample_rate: int


def _expect[T](value: ConfigEntry, kind: type[T]) -> T:
    if not isinstance(value, kind):
        raise TypeError(f"Expected {kind.__name__} in a voice model's config, got {value!r}.")
    return value


def _int(value: ConfigEntry) -> int:
    return _expect(value, int)


def _float(value: ConfigEntry) -> float:
    return float(value) if isinstance(value, int) else _expect(value, float)


def _ints(value: ConfigEntry) -> list[int]:
    return [_int(item) for item in _expect(value, list)]


def read_saved_config(saved: list[ConfigEntry], speakers: int) -> SavedModelConfig:
    """Parse the "config" list that saved_config makes. speakers overrides the saved speaker count."""
    resblock = _expect(saved[9], str)
    dilations = _expect(saved[11], list)
    model = ModelConfig(
        inter_channels=_int(saved[2]),
        hidden_channels=_int(saved[3]),
        filter_channels=_int(saved[4]),
        text_enc_hidden_dim=768,
        n_heads=_int(saved[5]),
        n_layers=_int(saved[6]),
        kernel_size=_int(saved[7]),
        p_dropout=_float(saved[8]),
        resblock=resblock,
        resblock_kernel_sizes=_ints(saved[10]),
        resblock_dilation_sizes=[_ints(item) for item in dilations],
        upsample_rates=_ints(saved[12]),
        upsample_initial_channel=_int(saved[13]),
        upsample_kernel_sizes=_ints(saved[14]),
        use_spectral_norm=False,
        gin_channels=_int(saved[16]),
        spk_embed_dim=speakers,
    )
    return SavedModelConfig(model, spec_channels=_int(saved[0]), sample_rate=_int(saved[17]))
