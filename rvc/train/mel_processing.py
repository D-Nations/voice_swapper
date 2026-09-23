import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn import functional as F

# Caches keyed by size, dtype, and device, since these are rebuilt on every training step otherwise.
_mel_basis: dict[str, torch.Tensor] = {}
_hann_window: dict[str, torch.Tensor] = {}


def spectrogram_torch(y: torch.Tensor, n_fft: int, hop_size: int, win_size: int, center: bool = False) -> torch.Tensor:
    """Magnitude spectrogram of [batch, samples] audio, reflect-padded so frames line up with hops."""
    key = f"{win_size}_{y.dtype}_{y.device}"
    if key not in _hann_window:
        _hann_window[key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    pad = (n_fft - hop_size) // 2
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_hann_window[key],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)


def spec_to_mel_torch(
    spec: torch.Tensor, n_fft: int, num_mels: int, sample_rate: int, fmin: float, fmax: float | None
) -> torch.Tensor:
    """Log mel spectrogram from a magnitude spectrogram."""
    key = f"{fmax}_{spec.dtype}_{spec.device}"
    if key not in _mel_basis:
        mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis[key] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    return torch.log(torch.matmul(_mel_basis[key], spec).clamp(min=1e-5))


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sample_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float | None,
    center: bool = False,
) -> torch.Tensor:
    """Log mel spectrogram of [batch, samples] audio."""
    spec = spectrogram_torch(y, n_fft, hop_size, win_size, center)
    return spec_to_mel_torch(spec, n_fft, num_mels, sample_rate, fmin, fmax)
