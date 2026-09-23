from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from cycle_gan.evaluation.vocoder import normalized_mel_to_waveform, save_waveform
from cycle_gan.preprocessing.preprocess_wav_files import preprocess_wav_files


def test_vocoder_recovers_the_length_and_pitch_of_a_tone(wav_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "numpy"
    preprocess_wav_files(wav_dir, output_dir)
    mel = torch.from_numpy(np.load(output_dir / "clip.npy"))

    waveform = normalized_mel_to_waveform(mel.unsqueeze(0).unsqueeze(0), n_iter=16)

    # The source clip is one second of a 440 Hz tone at 22,050 Hz.
    assert waveform.dim() == 1
    assert abs(waveform.shape[0] - 22050) <= 256
    spectrum = torch.fft.rfft(waveform).abs()
    peak_hz = spectrum.argmax().item() * 22050 / waveform.shape[0]
    assert abs(peak_hz - 440) < 30


def test_vocoder_rejects_the_wrong_number_of_mel_bands() -> None:
    with pytest.raises(ValueError):
        normalized_mel_to_waveform(torch.zeros(64, 10))


def test_save_waveform_writes_a_readable_wav(tmp_path: Path) -> None:
    path = tmp_path / "out.wav"

    save_waveform(torch.linspace(-2, 2, 100), path)

    data, sample_rate = sf.read(path)
    assert sample_rate == 22050
    assert len(data) == 100
    assert np.abs(data).max() <= 1.0
