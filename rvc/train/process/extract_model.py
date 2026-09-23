import datetime
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import torch

from rvc.configs.config import RVCConfig
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
    data, model = config.data, config.model
    model_config: list[ConfigEntry] = [
        data.spec_channels,
        32,
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
