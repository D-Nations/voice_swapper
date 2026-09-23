from pathlib import Path

import soundfile as sf
import torch
from torchaudio.transforms import GriffinLim, InverseMelScale

from cycle_gan.config import CONFIG
from cycle_gan.preprocessing.preprocess_wav_files import normalized_db_to_power

# More iterations give cleaner phase estimates but take longer.
GRIFFIN_LIM_ITERATIONS = 64


def normalized_mel_to_waveform(
    mel_spectrogram: torch.Tensor,
    n_fft: int = CONFIG.audio.n_fft,
    n_mels: int = CONFIG.audio.n_mels,
    hop_length: int = CONFIG.audio.hop_length,
    win_length: int = CONFIG.audio.win_length,
    sample_rate: int = CONFIG.audio.sample_rate,
    n_iter: int = GRIFFIN_LIM_ITERATIONS,
) -> torch.Tensor:
    """Convert a normalized (n_mels, frames) log-mel spectrogram back into a 1D waveform.

    Uses Griffin-Lim phase reconstruction. The defaults come from config.toml and match
    preprocess_wav_files. Extra leading dimensions of size 1, such as the
    batch and channel dimensions of generator output, are squeezed away.
    """
    mel_spectrogram = mel_spectrogram.detach().cpu().float()
    while mel_spectrogram.dim() > 2 and mel_spectrogram.shape[0] == 1:
        mel_spectrogram = mel_spectrogram.squeeze(0)
    if mel_spectrogram.dim() != 2 or mel_spectrogram.shape[0] != n_mels:
        raise ValueError(f"Expected a ({n_mels}, frames) spectrogram, got shape {tuple(mel_spectrogram.shape)}.")

    inverse_mel = InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate)
    griffin_lim = GriffinLim(n_fft=n_fft, n_iter=n_iter, win_length=win_length, hop_length=hop_length)

    linear_power = inverse_mel(normalized_db_to_power(mel_spectrogram))
    return griffin_lim(linear_power)


def save_waveform(waveform: torch.Tensor, path: str | Path, sample_rate: int = CONFIG.audio.sample_rate) -> None:
    """Write a 1D waveform to a WAV file, clipping it to [-1, 1]."""
    sf.write(path, waveform.clamp(-1, 1).numpy(), sample_rate)
