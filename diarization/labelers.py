"""Three ways to split a recording into speaker turns, for swapping voices turn by turn.

- map: the ECAPA speaker map, 1.5 second windows every 0.75 seconds. Handoffs are only known to
  within about 0.75 seconds, so each one snaps to the quietest point within SNAP_SECONDS["map"].
- refined: the map's turns, with each handoff re-placed by scoring short windows around it
  (turns.refine_boundaries), then snapped within a smaller distance.
- detector: the trained frame detector's per-20 ms probabilities. Where both hosts talk at once,
  the likelier one wins, since the audio can't be separated. The detector was trained almost only on
  the hosts, so it calls any voice one of them. Wherever the speaker map says a stretch matches
  neither host's voiceprint, its frames are labeled UNKNOWN, so guests and quoted clips keep their
  original audio just as they do with the map. The map decides who, and the detector decides when.

All work on 16 kHz mono audio and return turns in seconds.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from diarization.embedding import EMBEDDING_SAMPLE_RATE, HOP_SECONDS, SILENCE_DB, SpeakerEmbedder, embed_waveform
from diarization.speaker_maps import SpeakerMap, load_voiceprints
from diarization.turns import SILENT, UNKNOWN, Turn, cell_labels, plan_turns, refine_boundaries, snap_boundaries

if TYPE_CHECKING:
    # Imported only for its type: loading it pulls in transformers, which only the detector needs.
    from diarization.frame_detector import FrameDetector

METHODS = ("map", "refined", "detector")
VOICEPRINTS = Path("data/speaker_maps/voiceprints.npz")

SNAP_SECONDS = {"map": 0.75, "refined": 0.15, "detector": 0.1}
# The detector can place short turns precisely, so a backchannel only needs to be this long to count
# as its own turn rather than a flicker.
DETECTOR_MIN_TURN_SECONDS = 0.3


class Labeler:
    """Holds the models the methods need, loading each on first use."""

    def __init__(self, speakers: list[str], voiceprints_path: Path = VOICEPRINTS, device: str | None = None) -> None:
        self.speakers = speakers
        self.device = device
        voiceprints = load_voiceprints(voiceprints_path)
        self.voiceprints = np.stack([voiceprints[name] for name in speakers], axis=1)
        self._embedder: SpeakerEmbedder | None = None
        self._detector: FrameDetector | None = None

    @property
    def embedder(self) -> SpeakerEmbedder:
        if self._embedder is None:
            self._embedder = SpeakerEmbedder(self.device)
        return self._embedder

    @property
    def detector(self) -> FrameDetector:
        if self._detector is None:
            from diarization.frame_detector import SPEAKERS, load_detector

            if SPEAKERS != self.speakers:
                raise ValueError(f"The detector's speakers are {SPEAKERS}, not {self.speakers}.")
            self._detector = load_detector(device=self.device)
        return self._detector

    def speaker_map(self, audio: np.ndarray) -> SpeakerMap:
        windows = embed_waveform(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)), self.embedder)
        return SpeakerMap(self.speakers, windows.embeddings @ self.voiceprints, windows.loudness_db)

    def turns(self, method: str, audio: np.ndarray, speaker_map: SpeakerMap | None = None) -> list[Turn]:
        """Speaker turns for 16 kHz audio. speaker_map, if given, must be this audio's map."""
        rate = EMBEDDING_SAMPLE_RATE
        if method in ("map", "refined"):
            turns = plan_turns(cell_labels(speaker_map or self.speaker_map(audio)))
            if method == "refined":
                turns = refine_boundaries(turns, audio, self.voiceprints, self.embedder)
        elif method == "detector":
            from diarization.frame_detector import detect

            probs = detect(self.detector, audio)
            labels = detector_labels(probs, audio)
            labels = mask_non_hosts(labels, cell_labels(speaker_map or self.speaker_map(audio)))
            turns = plan_turns(labels, hop=0.02, min_turn_seconds=DETECTOR_MIN_TURN_SECONDS)
        else:
            raise ValueError(f"Unknown method {method!r}. Choose from {METHODS}.")
        # Turns end on whole cells, so the last can run past the audio.
        end = len(audio) / rate
        turns = [Turn(t.speaker, t.start, min(t.end, end)) for t in turns if t.start < end]
        return snap_boundaries(turns, audio, rate, SNAP_SECONDS[method])


def detector_labels(probs: np.ndarray, audio: np.ndarray, frame_seconds: float = 0.02) -> np.ndarray:
    """Per-frame labels from detector probabilities: the likelier host when either is talking,
    otherwise UNKNOWN for loud frames (music, a guest) and SILENT for quiet ones."""
    from diarization.frame_detector import ACTIVE

    frame = round(frame_seconds * EMBEDDING_SAMPLE_RATE)
    frames = min(len(probs), len(audio) // frame)
    probs = probs[:frames]
    level = 20 * np.log10(np.sqrt(np.mean(audio[: frames * frame].reshape(frames, frame) ** 2, axis=1)) + 1e-10)
    labels = probs.argmax(axis=1)
    quiet = probs.max(axis=1) < ACTIVE
    labels[quiet] = np.where(level[quiet] < SILENCE_DB, SILENT, UNKNOWN)
    return labels


def mask_non_hosts(
    labels: np.ndarray, cells: np.ndarray, frame_seconds: float = 0.02, cell_seconds: float = HOP_SECONDS
) -> np.ndarray:
    """Label as UNKNOWN every frame whose speaker-map cell matches neither host (cell label UNKNOWN)."""
    cell_of_frame = np.minimum((np.arange(len(labels)) * frame_seconds / cell_seconds).astype(int), len(cells) - 1)
    masked = labels.copy()
    masked[(cells[cell_of_frame] == UNKNOWN) & (labels != SILENT)] = UNKNOWN
    return masked
