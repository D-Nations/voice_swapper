from pathlib import Path

import pytest

from voice_service.voices import VOICES, VoiceChoice


def test_voices_are_keyed_by_their_training_folder() -> None:
    assert {key: (voice.label, voice.epoch, voice.index_rate) for key, voice in VOICES.items()} == {
        "pizarro": ("DaveBot", 130, 0.4),
        "sommers": ("TamBot", 220, 0.4),
    }


def test_model_path_finds_the_chosen_epoch(tmp_path: Path) -> None:
    folder = tmp_path / "dave"
    folder.mkdir()
    for name in ("dave_130e_12350s.pth", "dave_13e_1235s.pth", "dave_1300e_123500s.pth", "G_latest.pth"):
        (folder / name).touch()

    assert VoiceChoice("dave", "Dave", 130).model_path(tmp_path) == folder / "dave_130e_12350s.pth"


def test_model_path_explains_a_missing_epoch(tmp_path: Path) -> None:
    (tmp_path / "dave").mkdir()

    with pytest.raises(FileNotFoundError, match="epoch 130"):
        VoiceChoice("dave", "Dave", 130).model_path(tmp_path)


def test_index_path_is_none_without_an_index(tmp_path: Path) -> None:
    folder = tmp_path / "dave"
    folder.mkdir()
    voice = VoiceChoice("dave", "Dave", 130)

    assert voice.index_path(tmp_path) is None
    (folder / "dave.index").touch()
    assert voice.index_path(tmp_path) == folder / "dave.index"
