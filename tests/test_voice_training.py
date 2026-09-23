import subprocess
from pathlib import Path

import pytest

from rvc.weights import missing_files, required_files
from voice_service import train
from voice_service.train import TrainingRun, stage_commands, train_voice


def make_run(tmp_path: Path) -> TrainingRun:
    dataset = tmp_path / "clips"
    dataset.mkdir()
    (dataset / "a.wav").touch()
    return TrainingRun(name="dave", dataset_dir=dataset, logs_dir=tmp_path / "logs")


def test_stage_commands_run_in_order_with_the_run_settings(tmp_path: Path) -> None:
    run = make_run(tmp_path)

    commands = stage_commands(run)

    assert [stage for stage, _ in commands] == ["preprocess", "extract", "train", "index"]
    preprocess, extract, training, index = (command for _, command in commands)
    assert preprocess[2:5] == [str(run.experiment_dir), str(run.dataset_dir), "40000"]
    assert extract[2:4] == [str(run.experiment_dir), "rmvpe"]
    assert training[2] == "dave"
    assert training[5].endswith("f0G40k.pth") and training[6].endswith("f0D40k.pth")
    assert index[2:] == [str(run.experiment_dir), "Auto"]


def test_train_voice_refuses_to_start_without_pretrained_weights(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(train, "missing_files", lambda sample_rate: ["predictors/rmvpe.pt"])

    with pytest.raises(FileNotFoundError, match="rmvpe"):
        train_voice(make_run(tmp_path))


def test_train_voice_refuses_an_empty_dataset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(train, "missing_files", lambda sample_rate: [])
    run = TrainingRun(name="dave", dataset_dir=tmp_path, logs_dir=tmp_path / "logs")

    with pytest.raises(FileNotFoundError, match="No WAV clips"):
        train_voice(run)


def test_train_voice_stops_at_the_first_failed_stage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(train, "missing_files", lambda sample_rate: [])
    calls = []

    def fake_run(command, env, check):
        calls.append(Path(command[1]).name)
        assert env["RVC_LOGS_DIR"] == str(tmp_path / "logs")
        return subprocess.CompletedProcess(command, returncode=1 if len(calls) == 2 else 0)

    monkeypatch.setattr(train.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="extract"):
        train_voice(make_run(tmp_path))
    assert calls == ["preprocess.py", "extract.py"]


def test_train_voice_can_run_selected_stages(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(train, "missing_files", lambda sample_rate: [])
    calls = []
    monkeypatch.setattr(
        train.subprocess,
        "run",
        lambda command, env, check: calls.append(Path(command[1]).name) or subprocess.CompletedProcess(command, 0),
    )

    train_voice(make_run(tmp_path), stages=["index"])

    assert calls == ["extract_index.py"]


def test_weights_list_covers_shared_files_and_the_sample_rate(tmp_path: Path) -> None:
    files = required_files(48000)

    assert "predictors/rmvpe.pt" in files
    assert "pretraineds/hifi-gan/f0G48k.pth" in files
    assert missing_files(48000, models_dir=tmp_path) == list(files)
