from pathlib import Path

import numpy as np
import soundfile as sf

from diarization.select_clips import (
    IndexedSegment,
    choose_clips,
    eligible_episodes,
    held_out_episodes,
    select_sets,
)
from diarization.speaker_maps import Segment


def clip(episode: str, speaker: str, similarity: float, duration: float = 5.0) -> IndexedSegment:
    return IndexedSegment(episode, Segment(speaker, 0.0, duration, similarity))


def test_held_out_episodes_is_every_nth_in_sorted_order() -> None:
    files = [f"ep{i:02d}.mp3" for i in range(20)]

    assert held_out_episodes(list(reversed(files)), every=10) == {"ep00.mp3", "ep10.mp3"}


def test_choose_clips_prefers_high_similarity_and_stops_at_budget() -> None:
    candidates = [clip("a", "dave", 0.6), clip("b", "dave", 0.9), clip("c", "dave", 0.7)]

    chosen = choose_clips(candidates, budget_seconds=10.0)

    assert [c.audio_file for c in chosen] == ["b", "c"]


def test_choose_clips_skips_clips_outside_the_length_limits() -> None:
    candidates = [clip("a", "dave", 0.9, duration=2.0), clip("b", "dave", 0.8, duration=30.0), clip("c", "dave", 0.7)]

    assert [c.audio_file for c in choose_clips(candidates)] == ["c"]


def test_choose_clips_caps_each_episode() -> None:
    candidates = [clip("a", "dave", 0.9), clip("a", "dave", 0.8), clip("b", "dave", 0.5)]

    chosen = choose_clips(candidates, max_seconds_per_episode=5.0)

    assert [c.audio_file for c in chosen] == ["a", "b"]


def test_select_sets_never_trains_on_held_out_episodes() -> None:
    index = [clip(f"ep{i:02d}.mp3", speaker, 0.5 + i / 100) for i in range(20) for speaker in ("dave", "tamler")]

    sets = select_sets(index, train_minutes=10, test_clips=5)

    held_out = {"ep00.mp3", "ep10.mp3"}
    for speaker in ("dave", "tamler"):
        assert not {c.audio_file for c in sets["train"][speaker]} & held_out
        assert {c.audio_file for c in sets["test"][speaker]} == held_out


def test_select_sets_uses_only_eligible_episodes() -> None:
    index = [clip(f"ep{i:02d}.mp3", "dave", 0.5 + i / 100) for i in range(20)]
    eligible = {f"ep{i:02d}.mp3" for i in range(10, 20)}

    sets = select_sets(index, train_minutes=10, test_clips=5, eligible=eligible)

    used = {c.audio_file for c in sets["train"]["dave"] + sets["test"]["dave"]}
    assert used <= eligible


def test_eligible_episodes_filters_by_bitrate(tmp_path: Path) -> None:
    # 16-bit mono PCM at 8 kHz is 128 kbps. At 4 kHz it is 64 kbps.
    sf.write(tmp_path / "high.wav", np.zeros(8000 * 2, dtype=np.float32), 8000, subtype="PCM_16")
    sf.write(tmp_path / "low.wav", np.zeros(4000 * 2, dtype=np.float32), 4000, subtype="PCM_16")

    assert eligible_episodes(tmp_path, {"high.wav", "low.wav"}, min_kbps=100) == {"high.wav"}
