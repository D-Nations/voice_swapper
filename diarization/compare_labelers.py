"""Compare the ways of splitting audio into speaker turns, on synthetic conversations with exact labels.

The conversations are built from held-out test clips (diarization.mixtures), so no method has seen
them. Each method's turns are scored three ways:

- single-speaker accuracy: of the frames where exactly one host is talking, the share assigned to them.
- overlap accuracy: of the frames where both are talking, the share assigned to the louder one.
- handoff error: how far each handoff lands from the stretch where it belongs, which runs from the
  end of one host's speech to the start of the other's. Anywhere in that stretch counts as 0.

Run with: python -m diarization.compare_labelers [--mixtures 100] [--methods map refined detector]
"""

import argparse
from collections import defaultdict
from itertools import pairwise

import numpy as np
from tqdm import tqdm

from diarization.frame_detector import SPEAKERS, TEST_CLIPS
from diarization.labelers import METHODS, Labeler
from diarization.mixtures import FRAME_SECONDS, load_clips, make_mixture
from diarization.turns import Turn, turns_to_frames

MIXTURE_SECONDS = 30.0
MIXTURES = 100
SEED = 2
MAX_HANDOFF_ERROR = 2.0  # Seconds. A handoff with no matching boundary this close counts as missed.


def true_handoffs(active: np.ndarray) -> list[tuple[int, int, float, float]]:
    """(from speaker, to speaker, earliest, latest) seconds for each change of the one active speaker."""
    handoffs = []
    last_speaker, last_frame = None, None
    for frame, row in enumerate(active):
        if row.sum() != 1:
            continue
        speaker = int(row.argmax())
        if last_speaker is not None and last_frame is not None and speaker != last_speaker:
            handoffs.append((last_speaker, speaker, (last_frame + 1) * FRAME_SECONDS, frame * FRAME_SECONDS))
        last_speaker, last_frame = speaker, frame
    return handoffs


def handoff_errors(turns: list[Turn], active: np.ndarray) -> list[float | None]:
    """Distance from each true handoff to the nearest matching predicted boundary, or None if missed."""
    boundaries = [(a.speaker, b.speaker, a.end) for a, b in pairwise(turns) if a.speaker >= 0 and b.speaker >= 0]
    errors: list[float | None] = []
    for before, after, earliest, latest in true_handoffs(active):
        distances = [max(earliest - t, t - latest, 0.0) for b, a, t in boundaries if (b, a) == (before, after)]
        best = min(distances, default=None)
        errors.append(best if best is not None and best <= MAX_HANDOFF_ERROR else None)
    return errors


def score(turns: list[Turn], active: np.ndarray, energy: np.ndarray) -> dict[str, float]:
    predicted = turns_to_frames(turns, len(active), FRAME_SECONDS)
    single = active.sum(axis=1) == 1
    both = active.all(axis=1)
    return {
        "single_frames": float(single.sum()),
        "single_right": float((single & (predicted == active.argmax(axis=1))).sum()),
        "overlap_frames": float(both.sum()),
        "overlap_right": float((both & (predicted == energy.argmax(axis=1))).sum()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mixtures", type=int, default=MIXTURES)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", help="Such as cuda:0 or cpu. Defaults to the GPU if there is one.")
    args = parser.parse_args(argv)

    speakers = [load_clips(TEST_CLIPS / name) for name in SPEAKERS]
    labeler = Labeler(SPEAKERS, device=args.device)
    rng = np.random.default_rng(SEED)
    totals: dict[str, dict[str, float]] = {m: defaultdict(float) for m in args.methods}
    errors: dict[str, list[float | None]] = {m: [] for m in args.methods}
    for _ in tqdm(range(args.mixtures), desc="Scoring mixtures", unit="mixture"):
        audio, active, energy = make_mixture(speakers, [], rng, MIXTURE_SECONDS)
        speaker_map = labeler.speaker_map(audio) if {"map", "refined"} & set(args.methods) else None
        for method in args.methods:
            turns = labeler.turns(method, audio, speaker_map)
            for key, value in score(turns, active, energy).items():
                totals[method][key] += value
            errors[method].extend(handoff_errors(turns, active))

    print(f"\n{args.mixtures} mixtures of {MIXTURE_SECONDS:g} s from held-out clips.")
    print(
        f"{'method':<10} {'single':>7} {'overlap':>8} {'handoffs':>9} {'missed':>7} {'median':>7} {'<=0.1 s':>8} {'<=0.25 s':>9}"
    )
    for method in args.methods:
        t = totals[method]
        found = [e for e in errors[method] if e is not None]
        count = len(errors[method])
        print(
            f"{method:<10} {t['single_right'] / t['single_frames']:>7.1%} {t['overlap_right'] / max(t['overlap_frames'], 1):>8.1%}"
            f" {count:>9} {1 - len(found) / max(count, 1):>7.1%} {np.median(found) if found else float('nan'):>6.2f}s"
            f" {sum(e <= 0.1 for e in found) / max(count, 1):>8.1%} {sum(e <= 0.25 for e in found) / max(count, 1):>9.1%}"
        )


if __name__ == "__main__":
    main()
