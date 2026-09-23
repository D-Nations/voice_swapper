"""model_info.json in an experiment folder: facts about the dataset that the training stages pass along."""

import json
from pathlib import Path
from typing import Any

FILENAME = "model_info.json"


def read_model_info(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / FILENAME
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def update_model_info(experiment_dir: Path, **values: Any) -> None:
    data = read_model_info(experiment_dir) | values
    with open(experiment_dir / FILENAME, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
