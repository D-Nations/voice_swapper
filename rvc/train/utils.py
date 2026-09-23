from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import soundfile as sf
import torch
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Weight norm parameter names in torch's parametrizations API, and in the older API that saved
# checkpoints and exported models use, which keeps them loadable by other RVC tools.
PARAMETRIZED_NAMES = (".parametrizations.weight.original0", ".parametrizations.weight.original1")
LEGACY_NAMES = (".weight_g", ".weight_v")


def replace_keys_in_dict(d: Mapping[Any, Any], old_key_part: str, new_key_part: str) -> dict[Any, Any]:
    """Replace old_key_part with new_key_part in every string key, recursing into nested dicts."""
    return {
        (key.replace(old_key_part, new_key_part) if isinstance(key, str) else key): (
            replace_keys_in_dict(value, old_key_part, new_key_part) if isinstance(value, dict) else value
        )
        for key, value in d.items()
    }


def to_legacy_names(d: Mapping[Any, Any]) -> dict[Any, Any]:
    for new, old in zip(PARAMETRIZED_NAMES, LEGACY_NAMES, strict=True):
        d = replace_keys_in_dict(d, new, old)
    return dict(d)


def from_legacy_names(d: Mapping[Any, Any]) -> dict[Any, Any]:
    for new, old in zip(PARAMETRIZED_NAMES, LEGACY_NAMES, strict=True):
        d = replace_keys_in_dict(d, old, new)
    return dict(d)


def load_checkpoint(
    checkpoint_path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> tuple[int, dict[str, Any]]:
    """Load a training checkpoint into model and, if given, optimizer.

    Returns the epoch the checkpoint was saved at and the gradient scaler's state.
    """
    checkpoint = from_legacy_names(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    # Keep the model's own values for any keys the checkpoint lacks.
    state = {key: checkpoint["model"].get(key, value) for key, value in model.state_dict().items()}
    model.load_state_dict(state, strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint.get("optimizer", {}))
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
    checkpoint = {
        "model": model.state_dict(),
        "iteration": epoch,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
        "scaler": scaler.state_dict(),
    }
    torch.save(to_legacy_names(checkpoint), checkpoint_path)
    print(f"Saved checkpoint '{checkpoint_path}' (epoch {epoch})")


def latest_checkpoint_path(directory: Path, pattern: str) -> Path | None:
    """The most recently written file in directory matching pattern, such as "G_*.pth"."""
    checkpoints = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
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
    assert isinstance(canvas, FigureCanvasAgg)
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
