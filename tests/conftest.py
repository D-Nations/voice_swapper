from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def write_sine_wav(path: Path, sample_rate: int = 22050, seconds: float = 1.0, channels: int = 1) -> Path:
    """Write a 440 Hz sine wave to a WAV file and return its path."""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    tone = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    data = np.stack([tone] * channels, axis=1)
    sf.write(path, data, sample_rate)
    return path


@pytest.fixture
def wav_dir(tmp_path: Path) -> Path:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    write_sine_wav(audio_dir / "clip.wav")
    return audio_dir
