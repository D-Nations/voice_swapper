from pathlib import Path

import numpy as np

from cycle_gan.preprocessing.preprocess_wav_files import load_audio, preprocess_wav_files


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
