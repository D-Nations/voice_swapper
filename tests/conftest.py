from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

WavWriter = Callable[..., Path]


@pytest.fixture
def make_wav(tmp_path: Path) -> WavWriter:
    """Return a function that writes a 440 Hz sine wave WAV file into a temp audio folder."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    def _make_wav(name: str = "clip.wav", sample_rate: int = 22050, seconds: float = 1.0, channels: int = 1) -> Path:
        t = np.arange(int(sample_rate * seconds)) / sample_rate
        tone = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        data = np.stack([tone] * channels, axis=1)
        path = audio_dir / name
        sf.write(path, data, sample_rate)
        return path

    return _make_wav


@pytest.fixture
def wav_dir(make_wav: WavWriter) -> Path:
    return make_wav().parent
