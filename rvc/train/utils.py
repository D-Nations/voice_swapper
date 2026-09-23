import os
from collections.abc import Mapping
from pathlib import Path
from typing import NotRequired, TypedDict

import matplotlib
import numpy as np
import soundfile as sf
import torch
from torch.optim.optimizer import StateDict
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Weight norm parameter names in torch's parametrizations API, and in the older API that saved
# checkpoints and exported models use, which keeps them loadable by other RVC tools.
PARAMETRIZED_NAMES = (".parametrizations.weight.original0", ".parametrizations.weight.original1")
LEGACY_NAMES = (".weight_g", ".weight_v")


class Checkpoint(TypedDict):
    """A G_*.pth or D_*.pth training checkpoint, with weights under their legacy names."""

    model: dict[str, torch.Tensor]
    iteration: int  # The epoch it was saved at.
    optimizer: StateDict
    learning_rate: float
    scaler: NotRequired[StateDict]


def _rename_keys(
    state: Mapping[str, torch.Tensor], old_parts: tuple[str, ...], new_parts: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    renamed = dict(state)
    for old, new in zip(old_parts, new_parts, strict=True):
        renamed = {key.replace(old, new): value for key, value in renamed.items()}
    return renamed


def to_legacy_names(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename weight norm parameters from the parametrizations API to the legacy names."""
    return _rename_keys(state, PARAMETRIZED_NAMES, LEGACY_NAMES)


def from_legacy_names(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename weight norm parameters from the legacy names to the parametrizations API."""
    return _rename_keys(state, LEGACY_NAMES, PARAMETRIZED_NAMES)


def load_checkpoint(
    checkpoint_path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> tuple[int, StateDict]:
    """Load a training checkpoint into model and, if given, optimizer.

    Returns the epoch the checkpoint was saved at and the gradient scaler's state.
    """
    checkpoint: Checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved = from_legacy_names(checkpoint["model"])
    # Keep the model's own values for any keys the checkpoint lacks.
    state = {key: saved.get(key, value) for key, value in model.state_dict().items()}
    model.load_state_dict(state, strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['iteration']})")
    return checkpoint["iteration"], checkpoint.get("scaler", {})


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    epoch: int,
    checkpoint_path: Path,
    scaler: torch.amp.GradScaler,
) -> None:
    checkpoint: Checkpoint = {
        "model": to_legacy_names(model.state_dict()),
        "iteration": epoch,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
        "scaler": scaler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint '{checkpoint_path}' (epoch {epoch})")


def latest_checkpoint_path(directory: Path, pattern: str) -> Path | None:
    """The most recently written file in directory matching pattern, such as "G_*.pth"."""
    checkpoints = sorted(directory.glob(pattern), key=os.path.getmtime)
    return checkpoints[-1] if checkpoints else None


def summarize(
    writer: SummaryWriter,
    global_step: int,
    scalars: Mapping[str, float | torch.Tensor] | None = None,
    images: Mapping[str, np.ndarray] | None = None,
    audios: Mapping[str, torch.Tensor] | None = None,
    audio_sample_rate: int = 22050,
) -> None:
    """Log scalars, HWC images, and audio clips to TensorBoard."""
    for key, value in (scalars or {}).items():
        writer.add_scalar(key, value, global_step)
    for key, value in (images or {}).items():
        writer.add_image(key, value, global_step, dataformats="HWC")
    for key, value in (audios or {}).items():
        writer.add_audio(key, value, global_step, audio_sample_rate)


def plot_spectrogram_to_numpy(spectrogram: np.ndarray) -> np.ndarray:
    """Render a spectrogram as an RGB image array."""
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Channels")
    fig.tight_layout()
    canvas = fig.canvas
    if not isinstance(canvas, FigureCanvasAgg):
        raise TypeError(f"Plotting needs matplotlib's Agg backend, not {type(canvas).__name__}.")
    canvas.draw()
    data = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return data


def load_wav_to_torch(path: str | Path) -> tuple[torch.Tensor, int]:
    data, sample_rate = sf.read(path, dtype="float32")
    return torch.from_numpy(data), sample_rate


def load_filelist(path: Path, split: str = "|") -> list[list[str]]:
    with open(path, encoding="utf-8") as file:
        return [line.strip().split(split) for line in file]
