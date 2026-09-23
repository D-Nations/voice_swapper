from pathlib import Path

import soundfile as sf
import torch
from torchaudio.transforms import GriffinLim, InverseMelScale

from cycle_gan.preprocessing.preprocess_wav_files import normalized_db_to_power


def normalized_mel_to_waveform(
    mel_spectrogram: torch.Tensor,
    n_fft: int = 2048,
    n_mels: int = 128,
    hop_length: int = 256,
    win_length: int = 1024,
    sample_rate: int = 22050,
    n_iter: int = 64,
) -> torch.Tensor:
    """Convert a normalized (n_mels, frames) log-mel spectrogram back into a 1D waveform.

    Uses Griffin-Lim phase reconstruction. The keyword arguments must match the ones
    used by preprocess_wav_files. Extra leading dimensions of size 1, such as the
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


def save_waveform(waveform: torch.Tensor, path: str | Path, sample_rate: int = 22050) -> None:
    """Write a 1D waveform to a WAV file, clipping it to [-1, 1]."""
    sf.write(path, waveform.clamp(-1, 1).numpy(), sample_rate)
