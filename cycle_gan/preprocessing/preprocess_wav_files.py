import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load an audio file as a (channels, samples) float tensor and its sample rate."""
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy())
    return waveform, sample_rate


def preprocess_wav_files(
    input_dir: str | Path,
    output_dir: str | Path,
    n_fft: int = 2048,
    n_mels: int = 128,
    hop_length: int = 256,
    win_length: int = 1024,
    sample_rate: int = 22050,
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
        waveform, _ = load_audio(wav_file)
        mel_spectrogram = mel_transform(waveform)
        mel_spectrogram = mel_spectrogram.squeeze(0).numpy()
        np.save(output_path / f"{wav_file.stem}.npy", mel_spectrogram)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess_wav_files.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    preprocess_wav_files(input_directory, output_directory)
