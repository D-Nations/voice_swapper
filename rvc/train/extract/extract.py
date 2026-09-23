"""Stage 2: extract pitch and content features from each training slice, then write the file list.

For each slice in <experiment dir>/sliced_audios, writes:
- f0_voiced/<name>.wav.npy: RMVPE pitch in Hz every 10 ms.
- f0/<name>.wav.npy: the same pitch quantized to the 255 bins the model embeds.
- extracted/<name>.npy: ContentVec features every 20 ms.

Files that already exist are skipped, so an interrupted run can resume.

Run with: python -m rvc.train.extract.extract --experiment-dir DIR --sample-rate 40000 [--device cuda:0]
"""

import argparse
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from tqdm import tqdm

from rvc.lib.audio import load_audio
from rvc.lib.embedders import load_contentvec
from rvc.lib.predictors.rmvpe import RMVPE
from rvc.train.extract.preparing_files import SUPPORTED_EMBEDDER, generate_config, generate_filelist
from rvc.train.model_info import update_model_info

SAMPLE_RATE_16K = 16000
RMVPE_THRESHOLD = 0.03
MUTE_COPIES = 2

# Coarse pitch: 255 mel-spaced bins from F0_MIN to F0_MAX Hz, with bin 1 also covering unvoiced frames.
F0_BIN = 256
F0_MIN = 50.0
F0_MAX = 1100.0
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)


def coarse_f0(f0: np.ndarray) -> np.ndarray:
    """Quantize pitch in Hz to the integer bins 1 to 255 on the mel scale."""
    f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
    f0_mel = np.clip((f0_mel - F0_MEL_MIN) * (F0_BIN - 2) / (F0_MEL_MAX - F0_MEL_MIN) + 1, 1, F0_BIN - 1)
    return np.rint(f0_mel).astype(int)


def extract_pitch(wavs: list[Path], experiment_dir: Path, device: str) -> None:
    todo = [wav for wav in wavs if not (experiment_dir / "f0_voiced" / f"{wav.name}.npy").exists()]
    if not todo:
        return
    model = RMVPE(device)
    for wav in tqdm(todo, desc="Pitch"):
        f0 = model.get_f0(load_audio(wav, SAMPLE_RATE_16K), threshold=RMVPE_THRESHOLD)
        np.save(experiment_dir / "f0_voiced" / f"{wav.name}.npy", f0, allow_pickle=False)
        np.save(experiment_dir / "f0" / f"{wav.name}.npy", coarse_f0(f0), allow_pickle=False)


def extract_features(wavs: list[Path], experiment_dir: Path, device: str) -> None:
    todo = [wav for wav in wavs if not (experiment_dir / "extracted" / f"{wav.stem}.npy").exists()]
    if not todo:
        return
    # Typed as a plain Module, since transformers' wrapped .to() confuses type checkers.
    model = cast(torch.nn.Module, load_contentvec()).to(device).float().eval()
    for wav in tqdm(todo, desc="Features"):
        audio = torch.from_numpy(load_audio(wav, SAMPLE_RATE_16K)).to(device).float().view(1, -1)
        with torch.inference_mode():
            features = model(audio)["last_hidden_state"].squeeze(0).float().cpu().numpy()
        if np.isnan(features).any():
            print(f"{wav} produced NaN features and is skipped.")
            continue
        np.save(experiment_dir / "extracted" / f"{wav.stem}.npy", features, allow_pickle=False)


def extract(experiment_dir: Path, sample_rate: int, device: str, include_mutes: int = MUTE_COPIES) -> None:
    wav_dir = experiment_dir / "sliced_audios"
    wavs = sorted(wav_dir.glob("*.wav")) if wav_dir.is_dir() else []
    if not wavs:
        raise FileNotFoundError(f"No slices found in {wav_dir}. Run the preprocess stage first.")
    for folder in ("f0", "f0_voiced", "extracted"):
        (experiment_dir / folder).mkdir(exist_ok=True)

    start_time = time.time()
    extract_pitch(wavs, experiment_dir, device)
    extract_features(wavs, experiment_dir, device)
    print(f"Extracted pitch and features from {len(wavs)} slices in {time.time() - start_time:.1f} seconds.")

    update_model_info(experiment_dir, embedder_model=SUPPORTED_EMBEDDER)
    generate_config(sample_rate, experiment_dir)
    generate_filelist(experiment_dir, sample_rate, include_mutes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mute-copies", type=int, default=MUTE_COPIES, help="Silent examples per speaker")
    args = parser.parse_args(argv)
    extract(args.experiment_dir, args.sample_rate, args.device, args.mute_copies)


if __name__ == "__main__":
    main()
