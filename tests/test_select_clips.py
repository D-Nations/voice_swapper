from pathlib import Path

import numpy as np
import soundfile as sf

from diarization.select_clips import (
    MAX_CLIP_SECONDS,
    IndexedSegment,
    choose_clips,
    eligible_episodes,
    held_out_episodes,
    select_sets,
    trim_silence,
)
from diarization.speaker_maps import Segment


def clip(episode: str, speaker: str, similarity: float, duration: float = 12.0, start: float = 0.0) -> IndexedSegment:
    return IndexedSegment(episode, Segment(speaker, start, start + duration, similarity))


def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_held_out_episodes_is_every_nth_in_sorted_order() -> None:
    files = [f"ep{i:02d}.mp3" for i in range(20)]

    assert held_out_episodes(list(reversed(files)), every=10) == {"ep00.mp3", "ep10.mp3"}


def test_choose_clips_drops_short_clips_and_cuts_long_ones() -> None:
    candidates = [clip("a", "dave", 0.9, duration=5.0), clip("b", "dave", 0.9, duration=40.0, start=100.0)]

    chosen = choose_clips(candidates, rng(), min_similarity_quantile=0.0)

    assert [(c.audio_file, c.segment.start, c.segment.duration) for c in chosen] == [("b", 100.0, MAX_CLIP_SECONDS)]


def test_choose_clips_draws_only_from_the_better_matching_half() -> None:
    candidates = [clip(f"ep{i}", "dave", similarity=i / 10) for i in range(10)]

    chosen = choose_clips(candidates, rng(), min_similarity_quantile=0.5)

    assert {c.audio_file for c in chosen} == {f"ep{i}" for i in range(5, 10)}


def test_choose_clips_is_random_but_reproducible() -> None:
    candidates = [clip(f"ep{i}", "dave", 0.8) for i in range(30)]

    first = choose_clips(candidates, np.random.default_rng(1), max_clips=5)
    again = choose_clips(candidates, np.random.default_rng(1), max_clips=5)

    assert first == again
    assert [c.audio_file for c in first] != [f"ep{i}" for i in range(5)]


def test_choose_clips_respects_budget_and_per_episode_limits() -> None:
    candidates = [clip("a", "dave", 0.8, start=i * 20.0) for i in range(5)] + [clip("b", "dave", 0.8)]

    chosen = choose_clips(candidates, rng(), max_seconds_per_episode=24.0, min_similarity_quantile=0.0)

    assert sorted(c.audio_file for c in chosen) == ["a", "a", "b"]
    assert len(choose_clips(candidates, rng(), budget_seconds=20.0, min_similarity_quantile=0.0)) == 2


def test_select_sets_never_trains_on_held_out_episodes() -> None:
    index = [clip(f"ep{i:02d}.mp3", speaker, 0.8) for i in range(20) for speaker in ("dave", "tamler")]

    sets = select_sets(index, train_minutes=10, test_clips=5)

    held_out = {"ep00.mp3", "ep10.mp3"}
    for speaker in ("dave", "tamler"):
        assert not {c.audio_file for c in sets["train"][speaker]} & held_out
        assert {c.audio_file for c in sets["test"][speaker]} <= held_out


def test_select_sets_uses_only_eligible_episodes() -> None:
    index = [clip(f"ep{i:02d}.mp3", "dave", 0.8) for i in range(20)]
    eligible = {f"ep{i:02d}.mp3" for i in range(10, 20)}

    sets = select_sets(index, train_minutes=10, test_clips=5, eligible=eligible)

    assert {c.audio_file for c in sets["train"]["dave"] + sets["test"]["dave"]} <= eligible


def test_eligible_episodes_filters_by_bitrate(tmp_path: Path) -> None:
    # 16-bit mono PCM at 8 kHz is 128 kbps. At 4 kHz it is 64 kbps.
    sf.write(tmp_path / "high.wav", np.zeros(8000 * 2, dtype=np.float32), 8000, subtype="PCM_16")
    sf.write(tmp_path / "low.wav", np.zeros(4000 * 2, dtype=np.float32), 4000, subtype="PCM_16")

    assert eligible_episodes(tmp_path, {"high.wav", "low.wav"}, min_kbps=100) == {"high.wav"}


def test_trim_silence_cuts_quiet_ends_and_keeps_padding() -> None:
    sample_rate = 1000
    speech = 0.5 * np.sin(np.arange(2 * sample_rate) * 0.3)
    audio = np.concatenate([np.zeros(sample_rate), speech, np.zeros(3 * sample_rate)]).astype(np.float32)

    trimmed = trim_silence(audio, sample_rate)

    # Two seconds of speech plus 0.1 seconds of padding on each side.
    assert abs(len(trimmed) - 2.2 * sample_rate) <= 0.05 * sample_rate
