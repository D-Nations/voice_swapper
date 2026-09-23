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
