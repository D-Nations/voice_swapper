"""Train one RVC voice model from a folder of clips.

Runs the four training stages from the rvc package, each as its own process the way Applio does,
since each stage manages its own worker processes:

1. preprocess  Slice the clips and resample them to the model's sample rate.
2. extract     Track pitch with RMVPE and extract ContentVec features on the GPU.
3. train       Fine-tune the pretrained HiFi-GAN base models, exporting a small voice model
               every SAVE_EVERY_EPOCHS epochs so the best checkpoint can be picked later.
4. index       Build the retrieval index of the voice's features.

Everything for a voice lands in <logs dir>/<name>/. The logs dir defaults to logs/ and can be set
with --logs-dir or the RVC_LOGS_DIR environment variable.

Run with: python -m voice_service.train --name pizarro --dataset data/rvc/train/pizarro
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rvc.runtime import LOGS_DIR, PACKAGE_DIR
from rvc.weights import DEFAULT_SAMPLE_RATE, SAMPLE_RATES, base_model_files, missing_files

TOTAL_EPOCHS = 300
SAVE_EVERY_EPOCHS = 10
BATCH_SIZE = 8  # Fits in 8 GB of GPU memory at 40 kHz.
GPU = "0"
CPU_PROCESSES = max(1, min(8, (os.cpu_count() or 2) // 2))

# Preprocessing: automatic slicing into 3-second chunks with 0.3 seconds of overlap, and no extra
# filtering or noise reduction, since the clips are already clean speech.
CUT_MODE = "Automatic"
CHUNK_SECONDS = 3.0
OVERLAP_SECONDS = 0.3

PITCH_METHOD = "rmvpe"
EMBEDDER = "contentvec"
MUTE_COPIES = 2  # Silent examples mixed in per speaker, which helps the model stay quiet in pauses.
VOCODER = "HiFi-GAN"
INDEX_ALGORITHM = "Auto"


@dataclass(frozen=True)
class TrainingRun:
    name: str
    dataset_dir: Path
    logs_dir: Path = LOGS_DIR
    sample_rate: int = DEFAULT_SAMPLE_RATE
    total_epochs: int = TOTAL_EPOCHS
    save_every_epochs: int = SAVE_EVERY_EPOCHS
    batch_size: int = BATCH_SIZE
    gpu: str = GPU
    cpu_processes: int = CPU_PROCESSES

    @property
    def experiment_dir(self) -> Path:
        return self.logs_dir / self.name


def stage_commands(run: TrainingRun) -> list[tuple[str, list[str]]]:
    """The command line for each training stage, in order."""
    scripts = PACKAGE_DIR / "train"
    python = sys.executable
    pretrained_g, pretrained_d = (str(PACKAGE_DIR / "models" / path) for path in base_model_files(run.sample_rate))
    return [
        (
            "preprocess",
            [
                python,
                str(scripts / "preprocess" / "preprocess.py"),
                str(run.experiment_dir),
                str(run.dataset_dir),
                str(run.sample_rate),
                str(run.cpu_processes),
                CUT_MODE,
                "False",  # extra filtering effects
                "False",  # noise reduction
                "0.7",  # noise reduction strength, unused while noise reduction is off
                str(CHUNK_SECONDS),
                str(OVERLAP_SECONDS),
                "none",  # loudness normalization
            ],
        ),
        (
            "extract",
            [
                python,
                str(scripts / "extract" / "extract.py"),
                str(run.experiment_dir),
                PITCH_METHOD,
                str(run.cpu_processes),
                run.gpu,
                str(run.sample_rate),
                EMBEDDER,
                "None",  # custom embedder path
                str(MUTE_COPIES),
            ],
        ),
        (
            "train",
            [
                python,
                str(scripts / "train.py"),
                run.name,
                str(run.save_every_epochs),
                str(run.total_epochs),
                pretrained_g,
                pretrained_d,
                run.gpu,
                str(run.batch_size),
                str(run.sample_rate),
                "True",  # keep only the latest full checkpoint, which is large, for resuming
                "True",  # export a small voice model at every save
                "False",  # cache the dataset in GPU memory
                "False",  # delete a previous run's files first
                VOCODER,
                "False",  # gradient checkpointing, which saves memory at the cost of speed
            ],
        ),
        ("index", [python, str(scripts / "process" / "extract_index.py"), str(run.experiment_dir), INDEX_ALGORITHM]),
    ]


def check_inputs(run: TrainingRun) -> None:
    missing = missing_files(run.sample_rate)
    if missing:
        raise FileNotFoundError(
            f"Missing pretrained weights: {', '.join(missing)}. Download them with: python -m rvc.weights "
            f"--sample-rate {run.sample_rate}"
        )
    if not run.dataset_dir.is_dir() or not any(run.dataset_dir.glob("*.wav")):
        raise FileNotFoundError(f"No WAV clips found in {run.dataset_dir}.")


def train_voice(run: TrainingRun, stages: list[str] | None = None) -> None:
    """Run the training stages in order, stopping at the first one that fails."""
    check_inputs(run)
    run.experiment_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"RVC_LOGS_DIR": str(run.logs_dir)}
    for stage, command in stage_commands(run):
        if stages is not None and stage not in stages:
            continue
        print(f"=== {stage} ({run.name})", flush=True)
        result = subprocess.run(command, env=env, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"The {stage} stage failed for {run.name} with exit code {result.returncode}.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Model name, used for its folder, such as pizarro")
    parser.add_argument("--dataset", type=Path, required=True, help="Folder of WAV clips of one speaker")
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR, help="Parent folder for training runs")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, choices=SAMPLE_RATES)
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY_EPOCHS, help="Epochs between saved models")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gpu", default=GPU)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["preprocess", "extract", "train", "index"],
        help="Run only these stages, for example to rebuild the index",
    )
    args = parser.parse_args(argv)

    run = TrainingRun(
        name=args.name,
        dataset_dir=args.dataset.resolve(),
        logs_dir=args.logs_dir.resolve(),
        sample_rate=args.sample_rate,
        total_epochs=args.epochs,
        save_every_epochs=args.save_every,
        batch_size=args.batch_size,
        gpu=args.gpu,
    )
    train_voice(run, args.stages)
    print(f"Finished. Models and index are in {run.experiment_dir}")


if __name__ == "__main__":
    main()
