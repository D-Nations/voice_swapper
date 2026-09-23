from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from diarization.embedding import HOP_SECONDS, WINDOW_SECONDS, frame_windows, loudness_db
from diarization.speaker_maps import (
    Segment,
    SpeakerMap,
    extract_segments,
    find_segments,
    label_cells,
    label_windows,
    map_file,
    robust_voiceprint,
)

WINDOWS_PER_CELL = round(WINDOW_SECONDS / HOP_SECONDS)


def make_map(similarities: list[list[float]], loud: bool = True) -> SpeakerMap:
    sims = np.array(similarities, dtype=np.float32)
    loudness = np.full(len(sims), -20.0 if loud else -90.0, dtype=np.float32)
    return SpeakerMap(speakers=["dave", "tamler"], similarities=sims, loudness_db=loudness)


def test_frame_windows_cuts_overlapping_windows() -> None:
    waveform = torch.arange(10.0)

    windows = frame_windows(waveform, sample_rate=2, window_seconds=2.0, hop_seconds=1.0)

    assert windows.tolist() == [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]]


def test_loudness_of_a_full_scale_square_wave_is_zero_db() -> None:
    windows = torch.tensor([[1.0, -1.0, 1.0, -1.0]])

    assert loudness_db(windows).item() == pytest.approx(0.0, abs=1e-5)


def test_robust_voiceprint_ignores_a_minority_of_wrong_embeddings() -> None:
    right = np.tile([1.0, 0.0], (8, 1))
    wrong = np.tile([0.0, 1.0], (2, 1))

    voiceprint = robust_voiceprint(np.concatenate([right, wrong]), keep_fraction=0.8)

    assert np.allclose(voiceprint, [1.0, 0.0])


def test_label_windows_requires_similarity_margin_and_sound() -> None:
    speaker_map = make_map([[0.8, 0.1], [0.4, 0.1], [0.7, 0.6], [0.1, 0.9]])

    assert label_windows(speaker_map, min_similarity=0.5, min_margin=0.2).tolist() == [0, -1, -1, 1]
    assert label_windows(make_map([[0.8, 0.1]], loud=False)).tolist() == [-1]


def test_label_cells_drops_cells_next_to_a_change_of_speaker() -> None:
    # Windows span two cells. Cell 2 is covered by windows 1 and 2, which disagree.
    cells = label_cells(np.array([0, 0, 1, 1]), windows_per_cell=2)

    assert cells.tolist() == [0, 0, -1, 1, 1]


def test_find_segments_trims_edges_and_drops_short_runs() -> None:
    dave, tamler = [0.9, 0.1], [0.1, 0.9]
    speaker_map = make_map([dave] * 10 + [tamler] * 2)

    segments = find_segments(speaker_map, edge_trim_seconds=0.25, min_segment_seconds=3.0)

    # Dave's windows 0-9 give cells 0-9 (cell 10 is shared with Tamler). Tamler's run is too short.
    assert segments == [Segment("dave", 0.25, 10 * HOP_SECONDS - 0.25)]


def test_speaker_map_round_trips_through_csv(tmp_path: Path) -> None:
    speaker_map = make_map([[0.5, 0.25], [0.125, 0.75]])

    speaker_map.save(tmp_path / "map.csv")
    loaded = SpeakerMap.load(tmp_path / "map.csv")

    assert loaded.speakers == ["dave", "tamler"]
    assert np.allclose(loaded.similarities, speaker_map.similarities)
    assert np.allclose(loaded.loudness_db, speaker_map.loudness_db)


def test_map_file_scores_windows_against_each_voiceprint(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(16000 * 3, dtype=np.float32), 16000)

    def fake_embedder(windows: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[1.0, 0.0]]).repeat(len(windows), 1)

    voiceprints = {"dave": np.array([1.0, 0.0]), "tamler": np.array([0.0, 1.0])}
    speaker_map = map_file(path, voiceprints, fake_embedder)

    assert speaker_map.speakers == ["dave", "tamler"]
    assert np.allclose(speaker_map.similarities, [[1.0, 0.0]] * len(speaker_map.similarities))


def test_extract_segments_writes_wavs_per_speaker(tmp_path: Path) -> None:
    source = tmp_path / "episode.wav"
    sf.write(source, np.random.uniform(-0.5, 0.5, 8000 * 10).astype(np.float32), 8000)

    extract_segments(source, [Segment("dave", 1.0, 4.0), Segment("tamler", 5.0, 9.0)], tmp_path / "out")

    dave_files = list((tmp_path / "out" / "dave").glob("*.wav"))
    tamler_files = list((tmp_path / "out" / "tamler").glob("*.wav"))
    assert len(dave_files) == len(tamler_files) == 1
    assert sf.info(dave_files[0]).duration == pytest.approx(3.0)
    assert sf.info(tamler_files[0]).duration == pytest.approx(4.0)
