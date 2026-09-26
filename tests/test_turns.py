import numpy as np
import torch

from diarization.compare_labelers import handoff_errors, true_handoffs
from diarization.labelers import detector_labels, mask_non_hosts
from diarization.mixtures import FRAME, FRAME_SECONDS, Clip, make_mixture, speech_frames
from diarization.turns import (
    SILENT,
    UNKNOWN,
    Turn,
    best_split,
    clip_turns,
    plan_turns,
    refine_boundaries,
    snap_boundaries,
    turns_to_frames,
)

RATE = 16000


def test_plan_turns_splits_silence_between_different_speakers() -> None:
    labels = np.array([0, 0, SILENT, SILENT, 1, 1])

    turns = plan_turns(labels, hop=1.0)

    assert turns == [Turn(0, 0.0, 3.0), Turn(1, 3.0, 6.0)]


def test_plan_turns_drops_a_flicker_inside_a_turn() -> None:
    labels = np.array([0, 0, 0, 1, 0, 0])

    assert plan_turns(labels, hop=0.5, min_turn_seconds=1.0) == [Turn(0, 0.0, 3.0)]


def test_plan_turns_keeps_long_unknown_audio() -> None:
    labels = np.array([0] + [UNKNOWN] * 10 + [1])

    turns = plan_turns(labels, hop=1.0, absorb_seconds=8.0)

    assert [t.speaker for t in turns] == [0, UNKNOWN, 1]


def test_plan_turns_absorbs_a_short_sound_between_silences() -> None:
    labels = np.array([0, 0, SILENT, UNKNOWN, SILENT, 0, 0])

    assert plan_turns(labels, hop=0.5, absorb_seconds=8.0) == [Turn(0, 0.0, 3.5)]


def test_best_split_finds_the_sign_change() -> None:
    scores = np.array([-1.0, -0.5, -0.8, 0.9, 0.7, 1.0])

    assert best_split(scores) == 3


def test_best_split_ignores_one_noisy_window() -> None:
    scores = np.array([-1.0, 0.4, -1.0, -1.0, 1.0, 1.0, 1.0])

    assert best_split(scores) == 4


class LoudnessEmbedder:
    """Embeds a window as speaker 0 when it's positive on average and speaker 1 when negative."""

    def __call__(self, windows: torch.Tensor) -> torch.Tensor:
        sign = torch.sign(windows.mean(dim=1))
        return torch.stack([sign.clamp(min=0), (-sign).clamp(min=0)], dim=1)


def test_refine_boundaries_moves_a_handoff_to_where_the_voice_changes() -> None:
    # Speaker 0 is a positive offset until 4.3 s, then speaker 1 is a negative offset.
    audio = np.full(8 * RATE, 0.1, dtype=np.float32)
    audio[round(4.3 * RATE) :] = -0.1
    turns = [Turn(0, 0.0, 4.0), Turn(1, 4.0, 8.0)]

    refined = refine_boundaries(turns, audio, np.eye(2, dtype=np.float32), LoudnessEmbedder())

    assert abs(refined[0].end - 4.3) <= 0.05
    assert refined[1].start == refined[0].end


def test_refine_boundaries_leaves_unknown_edges_alone() -> None:
    audio = np.full(8 * RATE, 0.1, dtype=np.float32)
    turns = [Turn(0, 0.0, 4.0), Turn(UNKNOWN, 4.0, 8.0)]

    assert refine_boundaries(turns, audio, np.eye(2, dtype=np.float32), LoudnessEmbedder()) == turns


def test_snap_boundaries_moves_to_the_quiet_point() -> None:
    audio = np.full(4 * RATE, 0.5, dtype=np.float32)
    audio[round(2.2 * RATE) : round(2.26 * RATE)] = 0.0
    turns = [Turn(0, 0.0, 2.0), Turn(1, 2.0, 4.0)]

    snapped = snap_boundaries(turns, audio, RATE, snap_seconds=0.5)

    assert 2.2 <= snapped[0].end <= 2.26


def test_clip_turns_cuts_and_shifts() -> None:
    turns = [Turn(0, 0.0, 5.0), Turn(1, 5.0, 10.0), Turn(0, 10.0, 15.0)]

    assert clip_turns(turns, 4.0, 11.0) == [Turn(0, 0.0, 1.0), Turn(1, 1.0, 6.0), Turn(0, 6.0, 7.0)]


def test_turns_to_frames_labels_by_frame_centre() -> None:
    turns = [Turn(0, 0.0, 0.05), Turn(1, 0.05, 0.1)]

    assert turns_to_frames(turns, 6, 0.02).tolist() == [0, 0, 1, 1, 1, UNKNOWN]


def test_speech_frames_fills_short_gaps() -> None:
    audio = np.full(50 * FRAME, 0.5, dtype=np.float32)
    audio[10 * FRAME : 13 * FRAME] = 0.0  # 60 ms gap, filled.
    audio[30 * FRAME : 45 * FRAME] = 0.0  # 300 ms gap, kept.

    speech = speech_frames(audio, fill_seconds=0.2)

    assert speech[10:13].all()
    assert not speech[31:44].any()


def test_make_mixture_labels_match_where_each_speaker_was_placed() -> None:
    # Speaker 0's clips are all positive, speaker 1's all negative, so the mix's sign shows who's there.
    loud = np.ones(10 * RATE, dtype=np.float32)
    speakers = [[Clip(0.5 * loud, speech_frames(loud))], [Clip(-0.5 * loud, speech_frames(loud))]]

    audio, active, energy = make_mixture(speakers, [], np.random.default_rng(0), seconds=8.0)

    frames = audio[: len(active) * FRAME].reshape(len(active), FRAME).mean(axis=1)
    only_0 = active[:, 0] & ~active[:, 1]
    only_1 = active[:, 1] & ~active[:, 0]
    assert only_0.any() and only_1.any()
    assert (frames[only_0] > 0).all()
    assert (frames[only_1] < 0).all()
    assert (energy[active[:, 0], 0] > 0).all()


def test_true_handoffs_span_the_pause_between_speakers() -> None:
    active = np.zeros((10, 2), dtype=bool)
    active[0:3, 0] = True
    active[6:10, 1] = True

    assert true_handoffs(active) == [(0, 1, 3 * FRAME_SECONDS, 6 * FRAME_SECONDS)]


def test_handoff_errors_count_a_boundary_inside_the_pause_as_exact() -> None:
    active = np.zeros((100, 2), dtype=bool)
    active[0:40, 0] = True
    active[60:100, 1] = True  # Pause from 0.8 s to 1.2 s.

    assert handoff_errors([Turn(0, 0.0, 1.0), Turn(1, 1.0, 2.0)], active) == [0.0]
    assert handoff_errors([Turn(0, 0.0, 1.5), Turn(1, 1.5, 2.0)], active) == [0.30000000000000004]
    assert handoff_errors([Turn(0, 0.0, 2.0)], active) == [None]


def test_detector_labels_separate_silence_from_other_sound() -> None:
    probs = np.array([[0.9, 0.1], [0.6, 0.8], [0.1, 0.1], [0.1, 0.1]], dtype=np.float32)
    audio = np.zeros(4 * FRAME, dtype=np.float32)
    audio[2 * FRAME : 3 * FRAME] = 0.3  # Loud but neither host.

    assert detector_labels(probs, audio).tolist() == [0, 1, UNKNOWN, SILENT]


def test_mask_non_hosts_hands_frames_the_map_rejects_back_to_the_original() -> None:
    labels = np.array([0, 0, 1, 1, SILENT, 0])  # Frames of 0.5 s.
    cells = np.array([0, UNKNOWN, 1])  # Cells of 1 s: the second matches neither host.

    masked = mask_non_hosts(labels, cells, frame_seconds=0.5, cell_seconds=1.0)

    assert masked.tolist() == [0, 0, UNKNOWN, UNKNOWN, SILENT, 0]


def test_plan_turns_keeps_a_long_clip_split_by_short_pauses() -> None:
    # 0.5 s frames: 6 s of a clip, then 1 s of pause, then 6 s more. Each piece alone is under 8 s.
    labels = np.array([0] * 4 + [UNKNOWN] * 12 + [SILENT] * 2 + [UNKNOWN] * 12 + [1] * 4)

    turns = plan_turns(labels, hop=0.5, absorb_seconds=8.0)

    assert [t.speaker for t in turns] == [0, UNKNOWN, 1]
    assert turns[1].start == 2.0 and turns[1].end == 15.0


def test_plan_turns_does_not_join_short_sounds_across_a_long_pause() -> None:
    labels = np.array([0] * 4 + [UNKNOWN] * 4 + [SILENT] * 12 + [UNKNOWN] * 4 + [0] * 4)

    assert [t.speaker for t in plan_turns(labels, hop=0.5, absorb_seconds=8.0)] == [0]
