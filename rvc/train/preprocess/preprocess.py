"""Stage 1: cut a folder of clips into short training slices at the model's sample rate.

Each clip is split at silences, then into CHUNK_SECONDS pieces overlapping by OVERLAP_SECONDS,
and written to <experiment dir>/sliced_audios/0_<clip index>_<slice index>.wav.

Run with: python -m rvc.train.preprocess.preprocess --experiment-dir DIR --dataset DIR --sample-rate 40000
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from rvc.lib.audio import load_audio_resampled
from rvc.train.model_info import update_model_info
from rvc.train.preprocess.slicer import Slicer

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")
CHUNK_SECONDS = 3.0
OVERLAP_SECONDS = 0.3
SPEAKER_ID = 0  # One voice per model.

# Silence detection, in dB and milliseconds.
SILENCE_THRESHOLD_DB = -42
MIN_LENGTH_MS = 1500
MIN_INTERVAL_MS = 400
HOP_MS = 15
MAX_SILENCE_KEPT_MS = 500


def chunks(audio: np.ndarray, sample_rate: int, chunk_seconds: float, overlap_seconds: float) -> list[np.ndarray]:
    """Split audio at silences, then into overlapping chunk_seconds pieces. The last piece of each part runs to its end."""
    slicer = Slicer(
        sr=sample_rate,
        threshold=SILENCE_THRESHOLD_DB,
        min_length=MIN_LENGTH_MS,
        min_interval=MIN_INTERVAL_MS,
        hop_size=HOP_MS,
        max_sil_kept=MAX_SILENCE_KEPT_MS,
    )
    pieces = []
    for part in slicer.slice(audio):
        i = 0
        while True:
            start = int(sample_rate * (chunk_seconds - overlap_seconds) * i)
            i += 1
            if len(part[start:]) > (chunk_seconds + overlap_seconds) * sample_rate:
                pieces.append(part[start : start + int(chunk_seconds * sample_rate)])
            else:
                pieces.append(part[start:])
                break
    return pieces


def process_clip(
    path: Path, clip_index: int, output_dir: Path, sample_rate: int, chunk_seconds: float, overlap_seconds: float
) -> float:
    """Write the slices of one clip and return its duration in seconds."""
    audio = load_audio_resampled(path, sample_rate)
    for slice_index, piece in enumerate(chunks(audio, sample_rate, chunk_seconds, overlap_seconds)):
        wavfile.write(
            output_dir / f"{SPEAKER_ID}_{clip_index}_{slice_index}.wav", sample_rate, piece.astype(np.float32)
        )
    return len(audio) / sample_rate


def preprocess_dataset(
    dataset_dir: Path,
    experiment_dir: Path,
    sample_rate: int,
    num_processes: int,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
) -> float:
    """Slice every audio file under dataset_dir and return the total duration in seconds."""
    files = sorted(path for path in dataset_dir.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No audio files found in {dataset_dir}.")
    output_dir = experiment_dir / "sliced_audios"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    total_seconds = 0.0
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(process_clip, path, index, output_dir, sample_rate, chunk_seconds, overlap_seconds)
            for index, path in enumerate(files)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            total_seconds += future.result()

    duration = str(timedelta(seconds=int(total_seconds)))
    update_model_info(experiment_dir, total_dataset_duration=duration, total_seconds=total_seconds)
    print(f"Preprocessed {duration} of audio from {len(files)} files in {time.time() - start_time:.1f} seconds.")
    return total_seconds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="Folder of clips of one speaker")
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--processes", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunk-seconds", type=float, default=CHUNK_SECONDS)
    parser.add_argument("--overlap-seconds", type=float, default=OVERLAP_SECONDS)
    args = parser.parse_args(argv)
    preprocess_dataset(
        args.dataset, args.experiment_dir, args.sample_rate, args.processes, args.chunk_seconds, args.overlap_seconds
    )


if __name__ == "__main__":
    main()
