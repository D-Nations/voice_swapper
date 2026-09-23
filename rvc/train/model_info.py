"""model_info.json in an experiment folder: facts about the dataset that the training stages pass along."""

import json
from pathlib import Path
from typing import TypedDict, Unpack

FILENAME = "model_info.json"


class ModelInfo(TypedDict, total=False):
    total_dataset_duration: str  # Such as "0:30:05", written by preprocessing.
    total_seconds: float
    embedder_model: str  # Written by extraction.
    speakers_id: int  # Number of speakers, written with the file list.


def read_model_info(experiment_dir: Path) -> ModelInfo:
    path = experiment_dir / FILENAME
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as file:
        info: ModelInfo = json.load(file)
    return info


def update_model_info(experiment_dir: Path, **values: Unpack[ModelInfo]) -> None:
    info = read_model_info(experiment_dir)
    info.update(values)
    with open(experiment_dir / FILENAME, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=4)
