import random
import shutil
from pathlib import Path

from rvc.configs.config import config_path
from rvc.runtime import MUTE_DIR
from rvc.train.model_info import read_model_info, update_model_info

# Only the ContentVec mute set is bundled. Applio also shipped sets for its other embedders.
SUPPORTED_EMBEDDER = "contentvec"


def generate_config(sample_rate: int, experiment_dir: Path) -> None:
    """Copy the model config for sample_rate into the experiment, unless it already has one."""
    destination = experiment_dir / "config.json"
    if not destination.exists():
        shutil.copyfile(config_path(sample_rate), destination)


def _stems(folder: Path) -> set[str]:
    return {path.name.split(".")[0] for path in folder.iterdir()}


def _line(*fields: Path | str) -> str:
    return "|".join(str(field) for field in fields).replace("\\", "/")


def generate_filelist(experiment_dir: Path, sample_rate: int, include_mutes: int = 2) -> None:
    """Write filelist.txt: one line per slice with all four extracted files, plus silent examples.

    Each line is "<wav>|<features>|<coarse pitch>|<pitch in Hz>|<speaker id>", with absolute paths
    so training works from any directory. include_mutes silent examples per speaker teach the
    model to stay quiet in pauses.
    """
    wav_dir = experiment_dir / "sliced_audios"
    feature_dir = experiment_dir / "extracted"
    f0_dir = experiment_dir / "f0"
    f0nsf_dir = experiment_dir / "f0_voiced"
    names = _stems(wav_dir) & _stems(feature_dir) & _stems(f0_dir) & _stems(f0nsf_dir)

    embedder = read_model_info(experiment_dir).get("embedder_model", SUPPORTED_EMBEDDER)
    if embedder != SUPPORTED_EMBEDDER:
        raise ValueError(f"No mute files bundled for embedder {embedder!r}. Use {SUPPORTED_EMBEDDER}.")

    lines = []
    speaker_ids: list[str] = []
    for name in sorted(names):
        speaker_id = name.split("_")[0]
        if speaker_id not in speaker_ids:
            speaker_ids.append(speaker_id)
        lines.append(
            _line(
                (wav_dir / f"{name}.wav").resolve(),
                (feature_dir / f"{name}.npy").resolve(),
                (f0_dir / f"{name}.wav.npy").resolve(),
                (f0nsf_dir / f"{name}.wav.npy").resolve(),
                speaker_id,
            )
        )

    mute = (
        MUTE_DIR / "sliced_audios" / f"mute{sample_rate}.wav",
        MUTE_DIR / "extracted" / "mute.npy",
        MUTE_DIR / "f0" / "mute.wav.npy",
        MUTE_DIR / "f0_voiced" / "mute.wav.npy",
    )
    for speaker_id in speaker_ids * include_mutes:
        lines.append(_line(*(path.resolve() for path in mute), speaker_id))

    update_model_info(experiment_dir, speakers_id=len(speaker_ids))
    random.shuffle(lines)
    (experiment_dir / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")
