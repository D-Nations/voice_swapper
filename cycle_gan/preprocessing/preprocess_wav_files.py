import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from cycle_gan.config import CONFIG

# Avoids log10(0) for fully silent bins.
POWER_FLOOR = 1e-10


def power_to_normalized_db(
    power: torch.Tensor,
    min_db: float = CONFIG.audio.min_db,
    max_db: float = CONFIG.audio.max_db,
) -> torch.Tensor:
    """Convert a power mel spectrogram to decibels scaled from [min_db, max_db] into [-1, 1]."""
    db = 10 * torch.log10(power.clamp(min=POWER_FLOOR))
    db = db.clamp(min_db, max_db)
    return 2 * (db - min_db) / (max_db - min_db) - 1


def normalized_db_to_power(
    normalized: torch.Tensor,
    min_db: float = CONFIG.audio.min_db,
    max_db: float = CONFIG.audio.max_db,
) -> torch.Tensor:
    """Invert power_to_normalized_db, up to the clipping at min_db and max_db."""
    db = (normalized.clamp(-1, 1) + 1) / 2 * (max_db - min_db) + min_db
    return torch.pow(10.0, db / 10)


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load an audio file as a (channels, samples) float tensor and its sample rate."""
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy())
    return waveform, sample_rate


def preprocess_wav_files(
    input_dir: str | Path,
    output_dir: str | Path,
    n_fft: int = CONFIG.audio.n_fft,
    n_mels: int = CONFIG.audio.n_mels,
    hop_length: int = CONFIG.audio.hop_length,
    win_length: int = CONFIG.audio.win_length,
    sample_rate: int = CONFIG.audio.sample_rate,
) -> None:
    mel_transform = MelSpectrogram(
        sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
    )

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wav_files = list(input_path.glob("*.wav"))
    for wav_file in tqdm(wav_files, desc="Processing WAV files", unit="file"):
        waveform, file_sample_rate = load_audio(wav_file)
        mono = waveform.mean(dim=0)
        if file_sample_rate != sample_rate:
            mono = resample(mono, file_sample_rate, sample_rate)
        mel_spectrogram = power_to_normalized_db(mel_transform(mono)).numpy()
        np.save(output_path / f"{wav_file.stem}.npy", mel_spectrogram)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess_wav_files.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    preprocess_wav_files(input_directory, output_directory)
