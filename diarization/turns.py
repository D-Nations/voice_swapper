"""Turn per-frame speaker labels into speaker turns, and place the boundaries between them.

Labels come from either the ECAPA speaker map (one cell per HOP_SECONDS, see cell_labels) or the
frame detector (one frame per 20 ms). plan_turns merges them into turns, refine_boundaries re-scores
the audio around each handoff with short windows to place it more precisely, and snap_boundaries
moves each boundary to the quietest nearby point so no word is cut.
"""

from dataclasses import dataclass

import numpy as np
import torch

from diarization.embedding import EMBEDDING_SAMPLE_RATE, HOP_SECONDS, SILENCE_DB, Embedder, frame_windows, loudness_db
from diarization.speaker_maps import SpeakerMap

SILENT = -2
UNKNOWN = -1

# A cell belongs to a speaker when its best similarity is at least MIN_SIMILARITY. Music and quoted
# clips score near 0, while laughter and crosstalk score 0.2 to 0.45, so the threshold is low.
MIN_SIMILARITY = 0.2
# Unrecognized runs shorter than ABSORB_SECONDS join the neighbouring turns. Longer ones keep the
# original audio.
ABSORB_SECONDS = 8.0
# Speaker runs shorter than this, with the same label on both sides, take that label.
MIN_TURN_SECONDS = 1.5
# Pauses shorter than this don't end a stretch of unrecognized sound (see plan_turns).
MAX_PAUSE_SECONDS = 2.0

# refine_boundaries searches REFINE_SECONDS either side of each handoff, scoring windows of
# REFINE_WINDOW_SECONDS every REFINE_HOP_SECONDS. ECAPA is less reliable on short windows, but the
# split point is chosen from all the windows together, so single noisy windows barely move it.
REFINE_SECONDS = 1.5
REFINE_WINDOW_SECONDS = 0.75
REFINE_HOP_SECONDS = 0.05

SNAP_FRAME_SECONDS = 0.02


@dataclass(frozen=True)
class Turn:
    speaker: int  # Speaker index, or UNKNOWN to keep the original audio.
    start: float  # Seconds.
    end: float


def cell_labels(speaker_map: SpeakerMap, min_similarity: float = MIN_SIMILARITY) -> np.ndarray:
    """Label each hop-long cell of a speaker map with a speaker index, UNKNOWN, or SILENT.

    Cell i is covered by windows i - 1 and i, so it takes their mean similarity and loudest level.
    """
    similarities = speaker_map.similarities
    loudness = speaker_map.loudness_db
    previous_sims = np.vstack([similarities[:1], similarities[:-1]])
    previous_loud = np.concatenate([loudness[:1], loudness[:-1]])
    cell_sims = (similarities + previous_sims) / 2
    cell_loud = np.maximum(loudness, previous_loud)

    labels = cell_sims.argmax(axis=1)
    labels[cell_sims.max(axis=1) < min_similarity] = UNKNOWN
    labels[cell_loud < SILENCE_DB] = SILENT
    return labels


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """(label, first cell, cell after the last) for each run of equal labels."""
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(labels)]])
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def plan_turns(
    labels: np.ndarray,
    hop: float = HOP_SECONDS,
    absorb_seconds: float = ABSORB_SECONDS,
    min_turn_seconds: float = MIN_TURN_SECONDS,
) -> list[Turn]:
    """Merge per-cell labels, hop seconds apart, into speaker turns and stretches of original audio."""
    labels = labels.copy()
    # Drop brief flickers to another speaker in the middle of someone's turn or of music.
    runs = _runs(labels)
    for i in range(1, len(runs) - 1):
        label, start, end = runs[i]
        if label >= 0 and (end - start) * hop < min_turn_seconds and runs[i - 1][0] == runs[i + 1][0] != SILENT:
            labels[start:end] = runs[i - 1][0]

    # Unrecognized sound too short to be music or a clip, like a laugh or a breath, counts as silence.
    # A stretch is measured from its first unrecognized run to its last, across pauses shorter than
    # MAX_PAUSE_SECONDS, since at fine resolution a clip's own pauses split it into short runs.
    runs = _runs(labels)
    stretch: list[tuple[int, int, int]] = []
    for i, run in enumerate([*runs, (0, len(labels), len(labels))]):
        label, start, end = run
        pause_ends_stretch = label == SILENT and (end - start) * hop >= MAX_PAUSE_SECONDS
        if i == len(runs) or label >= 0 or pause_ends_stretch:
            unknown = [r for r in stretch if r[0] == UNKNOWN]
            if unknown and (unknown[-1][2] - unknown[0][1]) * hop >= absorb_seconds:
                labels[unknown[0][1] : unknown[-1][2]] = UNKNOWN
            else:
                for _, a, b in unknown:
                    labels[a:b] = SILENT
            stretch = []
        else:
            stretch.append(run)

    # Silence goes to the neighbouring turns, split at its midpoint when the speakers on either side differ.
    runs = _runs(labels)
    for i, (label, start, end) in enumerate(runs):
        if label != SILENT:
            continue
        before = runs[i - 1][0] if i > 0 else None
        after = runs[i + 1][0] if i + 1 < len(runs) else None
        before = before if before is not None and before >= 0 else None
        after = after if after is not None and after >= 0 else None
        if before is None and after is None:
            labels[start:end] = UNKNOWN
        elif before is None or after is None:
            labels[start:end] = before if before is not None else after
        else:
            middle = (start + end) // 2
            labels[start:middle] = before
            labels[middle:end] = after

    return [Turn(label, start * hop, end * hop) for label, start, end in _runs(labels)]


def best_split(scores: np.ndarray) -> int:
    """The index k that best splits scores into a first part scoring low and a second scoring high.

    scores[i] > 0 means window i sounds more like the speaker after the handoff. Returns the k that
    maximizes sum(scores[k:]) - sum(scores[:k]).
    """
    before = np.concatenate([[0.0], np.cumsum(scores)])
    total = before[-1]
    return int(np.argmax(total - 2 * before))


def refine_boundaries(
    turns: list[Turn],
    audio: np.ndarray,
    voiceprints: np.ndarray,
    embedder: Embedder,
    radius: float = REFINE_SECONDS,
    window_seconds: float = REFINE_WINDOW_SECONDS,
    hop_seconds: float = REFINE_HOP_SECONDS,
) -> list[Turn]:
    """Re-place each handoff between two speakers by scoring short windows around it.

    audio is 16 kHz mono, and voiceprints is (dims, speakers) in the turns' speaker order. Boundaries
    next to UNKNOWN stretches are left alone. A refined boundary never passes its neighbours.
    """
    rate = EMBEDDING_SAMPLE_RATE
    edges = [turn.start for turn in turns] + [turns[-1].end]
    for i in range(1, len(turns)):
        before, after = turns[i - 1].speaker, turns[i].speaker
        if before < 0 or after < 0 or before == after:
            continue
        lo = max(edges[i] - radius, (edges[i - 1] + edges[i]) / 2)
        hi = min(edges[i] + radius, (edges[i] + edges[i + 1]) / 2)
        first = max(0, round((lo - window_seconds / 2) * rate))
        last = min(len(audio), round((hi + window_seconds / 2) * rate))
        windows = frame_windows(
            torch.from_numpy(np.ascontiguousarray(audio[first:last], dtype=np.float32)),
            rate,
            window_seconds,
            hop_seconds,
        )
        if windows.shape[0] < 2:
            continue
        similarities = embedder(windows).numpy() @ voiceprints
        scores = similarities[:, after] - similarities[:, before]
        scores[loudness_db(windows).numpy() < SILENCE_DB] = 0.0
        centres = first / rate + window_seconds / 2 + np.arange(len(scores)) * hop_seconds
        k = best_split(scores)
        if 0 < k < len(scores):
            edges[i] = float((centres[k - 1] + centres[k]) / 2)
    return [Turn(turn.speaker, edges[i], edges[i + 1]) for i, turn in enumerate(turns)]


def snap_boundaries(turns: list[Turn], audio: np.ndarray, sample_rate: int, snap_seconds: float) -> list[Turn]:
    """Move each boundary between turns to the quietest short frame within snap_seconds of it."""
    frame = round(SNAP_FRAME_SECONDS * sample_rate)
    duration = len(audio) / sample_rate
    snapped = []
    for boundary in (turn.end for turn in turns[:-1]):
        lo = max(0, round((boundary - snap_seconds) * sample_rate))
        hi = min(len(audio), round((boundary + snap_seconds) * sample_rate))
        count = (hi - lo) // frame
        if count == 0:
            snapped.append(boundary)
            continue
        energy = np.mean(audio[lo : lo + count * frame].reshape(count, frame) ** 2, axis=1)
        snapped.append((lo + (int(energy.argmin()) + 0.5) * frame) / sample_rate)
    edges = [0.0, *snapped, duration]
    # Keep edges in order if two snaps crossed.
    edges = list(np.maximum.accumulate(edges))
    return [Turn(turn.speaker, edges[i], edges[i + 1]) for i, turn in enumerate(turns) if edges[i + 1] > edges[i]]


def clip_turns(turns: list[Turn], start: float, end: float) -> list[Turn]:
    """The turns overlapping start to end, cut to it and shifted so start is 0."""
    return [
        Turn(t.speaker, max(t.start, start) - start, min(t.end, end) - start)
        for t in turns
        if t.end > start and t.start < end
    ]


def turns_to_frames(turns: list[Turn], frames: int, frame_seconds: float) -> np.ndarray:
    """The speaker of each frame, by its centre, or UNKNOWN outside every turn."""
    labels = np.full(frames, UNKNOWN)
    centres = (np.arange(frames) + 0.5) * frame_seconds
    for turn in turns:
        labels[(centres >= turn.start) & (centres < turn.end)] = turn.speaker
    return labels
