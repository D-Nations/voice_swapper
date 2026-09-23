from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio


def _read_mono(path: str | Path, dtype: Literal["float64", "float32"] = "float64") -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype=dtype)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T)
    return audio, sample_rate


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Load a file as mono float64 audio at sample_rate, resampling with soxr if needed."""
    audio, source_rate = _read_mono(path)
    if source_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=sample_rate, res_type="soxr_vhq")
    return audio.flatten()


def load_audio_resampled(path: str | Path, sample_rate: int) -> np.ndarray:
    """Load a file as mono float32 audio at sample_rate, resampling with torchaudio's windowed sinc if needed.

    Training preprocessing uses this resampler, as Applio does.
    """
    audio, source_rate = _read_mono(path, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if source_rate != sample_rate:
        resample = torchaudio.transforms.Resample(orig_freq=source_rate, new_freq=sample_rate, lowpass_filter_width=128)
        audio = resample(torch.from_numpy(audio).unsqueeze(0)).squeeze(0).contiguous().numpy()
    return audio
