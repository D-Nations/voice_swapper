from pathlib import Path

import numpy as np
import torch

from cycle_gan.config import CONFIG
from cycle_gan.preprocessing.preprocess_wav_files import (
    load_audio,
    normalized_db_to_power,
    power_to_normalized_db,
    preprocess_wav_files,
)


def test_load_audio_returns_channels_first_tensor(wav_dir: Path) -> None:
    waveform, sample_rate = load_audio(wav_dir / "clip.wav")

    assert sample_rate == 22050
    assert tuple(waveform.shape) == (1, 22050)


def test_preprocess_writes_one_spectrogram_per_wav(wav_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "numpy"

    preprocess_wav_files(wav_dir, output_dir)

    spectrogram = np.load(output_dir / "clip.npy")
    assert spectrogram.shape[0] == 128


def test_stereo_audio_is_downmixed_to_a_2d_spectrogram(make_wav, tmp_path: Path) -> None:
    wav = make_wav(channels=2)
    output_dir = tmp_path / "numpy"

    preprocess_wav_files(wav.parent, output_dir)

    spectrogram = np.load(output_dir / "clip.npy")
    assert spectrogram.ndim == 2
    assert spectrogram.shape[0] == 128


def test_audio_is_resampled_to_the_target_rate(make_wav, tmp_path: Path) -> None:
    wav = make_wav(sample_rate=48000, seconds=1.0)
    output_dir = tmp_path / "numpy"

    preprocess_wav_files(wav.parent, output_dir, sample_rate=22050, hop_length=256)

    # One second at 22,050 Hz with a centered STFT gives 1 + 22050 // 256 frames.
    spectrogram = np.load(output_dir / "clip.npy")
    assert spectrogram.shape[1] == 1 + 22050 // 256


def test_saved_spectrograms_are_scaled_to_the_tanh_range(wav_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "numpy"

    preprocess_wav_files(wav_dir, output_dir)

    spectrogram = np.load(output_dir / "clip.npy")
    assert spectrogram.min() >= -1.0
    assert spectrogram.max() <= 1.0
    # A loud tone should use the upper part of the range, not collapse to silence.
    assert spectrogram.max() > 0.0


def test_normalization_maps_the_db_range_endpoints_to_minus_one_and_one() -> None:
    min_db, max_db = CONFIG.audio.min_db, CONFIG.audio.max_db
    power = torch.tensor([0.0, 10 ** (min_db / 10), 10 ** (max_db / 10), 1e12])

    normalized = power_to_normalized_db(power)

    assert torch.allclose(normalized, torch.tensor([-1.0, -1.0, 1.0, 1.0]))


def test_normalization_round_trips_within_the_db_range() -> None:
    power = torch.tensor([1e-6, 1e-3, 1.0, 1e3, 1e4])

    restored = normalized_db_to_power(power_to_normalized_db(power))

    assert torch.allclose(restored, power, rtol=1e-4)
