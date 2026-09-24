from pathlib import Path

import pytest

from voice_service.evaluate import (
    ClipScore,
    choose_epochs,
    normalize_words,
    read_scores,
    scores_path,
    summarize,
    voice_models,
    word_error_rate,
    write_scores,
)


def test_normalize_words_ignores_case_and_punctuation() -> None:
    assert normalize_words("Well, it's a TWO-part episode.") == ["well", "it's", "a", "two", "part", "episode"]


def test_word_error_rate_counts_substitutions_insertions_and_deletions() -> None:
    assert word_error_rate("the cat sat", "The cat sat.") == 0.0
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    assert word_error_rate("the cat sat", "the cat sat down") == pytest.approx(1 / 3)
    assert word_error_rate("the cat sat", "cat") == pytest.approx(2 / 3)


def test_word_error_rate_of_an_empty_reference() -> None:
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "hello") == 1.0


def test_voice_models_finds_exported_models_by_epoch(tmp_path: Path) -> None:
    for name in ("dave_20e_1900s.pth", "dave_100e_9500s.pth", "G_latest.pth", "dave_10e_950s.pth"):
        (tmp_path / name).touch()

    models = voice_models(tmp_path)

    assert list(models) == [10, 20, 100]
    assert models[100].name == "dave_100e_9500s.pth"


def test_choose_epochs_keeps_every_step_and_the_last() -> None:
    assert choose_epochs([10, 20, 30, 40, 50], step=20) == [20, 40, 50]
    assert choose_epochs([20, 40], step=20) == [20, 40]
    assert choose_epochs([], step=20) == []


def test_summarize_averages_per_voice_and_epoch() -> None:
    scores = [
        ClipScore("dave", 20, "a.wav", 0.8, 0.2, 0.1, "", ""),
        ClipScore("dave", 20, "b.wav", 0.6, 0.4, 0.3, "", ""),
        ClipScore("dave", 40, "a.wav", 0.9, 0.1, 0.0, "", ""),
    ]

    summaries = summarize(scores)

    assert [(s.epoch, s.target_similarity, s.word_error_rate) for s in summaries] == [
        (20, pytest.approx(0.7), pytest.approx(0.2)),
        (40, pytest.approx(0.9), pytest.approx(0.0)),
    ]


def test_scores_round_trip_through_csv(tmp_path: Path) -> None:
    scores = [
        ClipScore("dave", 20, "a, b.wav", 0.8, 0.2, 0.1, "The cat, sat.", "the cat sat"),
        ClipScore("dave", 40, "c.wav", 0.9, 0.1, 0.0, "", "", index_rate=0.9),
    ]
    path = scores_path(tmp_path, "dave")

    write_scores(path, scores)

    assert path.name == "scores_dave.csv"
    assert read_scores(path) == scores


def test_summarize_keeps_index_rates_apart() -> None:
    scores = [
        ClipScore("dave", 130, "a.wav", 0.8, 0.2, 0.1, "", "", index_rate=0.5),
        ClipScore("dave", 130, "a.wav", 0.9, 0.1, 0.2, "", "", index_rate=1.0),
    ]

    assert [(s.index_rate, s.target_similarity) for s in summarize(scores)] == [(0.5, 0.8), (1.0, 0.9)]


def test_scores_from_before_index_rates_read_as_the_default(tmp_path: Path) -> None:
    path = tmp_path / "scores_dave.csv"
    path.write_text(
        "voice,epoch,clip,target_similarity,source_similarity,word_error_rate,reference_text,converted_text\n"
        "dave,20,a.wav,0.8,0.2,0.1,,\n",
        encoding="utf-8",
    )

    assert read_scores(path) == [ClipScore("dave", 20, "a.wav", 0.8, 0.2, 0.1, "", "", index_rate=0.75)]
