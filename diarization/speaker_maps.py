"""Map who is speaking when in long recordings, and extract clean single-speaker audio.

Workflow, run as `python -m diarization.speaker_maps <command>`:

1. enroll   Build a voiceprint for each speaker from folders of clips that mostly contain them.
2. map      Score every window of every audio file in a folder against the voiceprints and save
            one CSV map per file.
3. extract  Turn maps into confident single-speaker segments and write them out as WAV files.

The enrollment clips do not need to be perfect. Voiceprints are built from the windows that
agree with each other most, so a minority of wrong windows barely affects them.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from diarization.embedding import (
    HOP_SECONDS,
    SILENCE_DB,
    WINDOW_SECONDS,
    Embedder,
    SpeakerEmbedder,
    embed_file,
    load_mono,
)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")

# Share of enrollment windows kept when building a voiceprint. The rest, the windows least
# like the speaker's average, are treated as mislabeled or noisy and dropped.
ENROLL_KEEP_FRACTION = 0.8

# A window is assigned to a speaker only when its cosine similarity to that speaker's voiceprint
# is at least MIN_SIMILARITY and beats every other speaker by at least MIN_MARGIN.
MIN_SIMILARITY = 0.5
MIN_MARGIN = 0.2

# Extra audio trimmed from both ends of every segment, on top of the overlap rule in label_cells.
EDGE_TRIM_SECONDS = 0.25
MIN_SEGMENT_SECONDS = 3.0


# ---------------------------------------------------------------- voiceprints


def robust_voiceprint(embeddings: np.ndarray, keep_fraction: float = ENROLL_KEEP_FRACTION) -> np.ndarray:
    """Average unit-length embeddings, then re-average only the keep_fraction closest to that average."""
    if len(embeddings) == 0:
        raise ValueError("Cannot build a voiceprint from zero embeddings.")
    centroid = _unit(embeddings.mean(axis=0))
    similarities = embeddings @ centroid
    keep = max(1, round(len(embeddings) * keep_fraction))
    closest = np.argsort(similarities)[-keep:]
    return _unit(embeddings[closest].mean(axis=0))


def save_voiceprints(path: str | Path, voiceprints: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(voiceprints)
    np.savez(path, names=np.array(names), voiceprints=np.stack([voiceprints[n] for n in names]))


def load_voiceprints(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {str(name): vector for name, vector in zip(data["names"], data["voiceprints"])}


def enroll(folders: dict[str, Path], embedder: Embedder) -> dict[str, np.ndarray]:
    """Build one voiceprint per speaker from the non-silent windows of every audio file in their folder."""
    voiceprints = {}
    for name, folder in folders.items():
        files = audio_files(folder)
        if not files:
            raise ValueError(f"No audio files found in {folder}.")
        embeddings = []
        for file in tqdm(files, desc=f"Enrolling {name}", unit="file"):
            windows = embed_file(file, embedder)
            embeddings.append(windows.embeddings[windows.loudness_db >= SILENCE_DB])
        voiceprints[name] = robust_voiceprint(np.concatenate(embeddings))
    return voiceprints


# ---------------------------------------------------------------- maps


@dataclass
class SpeakerMap:
    """Per-window similarity to each speaker for one audio file. Window i starts at i * HOP_SECONDS."""

    speakers: list[str]
    similarities: np.ndarray  # (windows, speakers)
    loudness_db: np.ndarray  # (windows,)

    def save(self, path: str | Path) -> None:
        """Write the map as a CSV with one row per window."""
        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["start", "end", "loudness_db", *self.speakers])
            for i, (loudness, row) in enumerate(zip(self.loudness_db, self.similarities)):
                start = i * HOP_SECONDS
                writer.writerow(
                    [f"{start:.2f}", f"{start + WINDOW_SECONDS:.2f}", f"{loudness:.1f}", *(f"{s:.4f}" for s in row)]
                )

    @classmethod
    def load(cls, path: str | Path) -> SpeakerMap:
        with open(path, newline="") as file:
            reader = csv.reader(file)
            header = next(reader)
            rows = np.array([[float(value) for value in row] for row in reader], dtype=np.float32)
        rows = rows.reshape(-1, len(header))
        return cls(speakers=header[3:], similarities=rows[:, 3:], loudness_db=rows[:, 2])


def map_file(path: str | Path, voiceprints: dict[str, np.ndarray], embedder: Embedder) -> SpeakerMap:
    """Score every window of an audio file against every voiceprint."""
    speakers = list(voiceprints)
    windows = embed_file(path, embedder)
    if len(windows.embeddings) == 0:
        return SpeakerMap(speakers, np.zeros((0, len(speakers)), dtype=np.float32), windows.loudness_db)
    matrix = np.stack([voiceprints[name] for name in speakers], axis=1)
    return SpeakerMap(speakers, windows.embeddings @ matrix, windows.loudness_db)


# ---------------------------------------------------------------- segments


@dataclass(frozen=True)
class Segment:
    speaker: str
    start: float  # seconds
    end: float  # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


def label_windows(
    speaker_map: SpeakerMap,
    min_similarity: float = MIN_SIMILARITY,
    min_margin: float = MIN_MARGIN,
    silence_db: float = SILENCE_DB,
) -> np.ndarray:
    """Return the speaker index each window confidently belongs to, or -1 for none."""
    similarities = speaker_map.similarities
    labels = np.full(len(similarities), -1)
    if len(similarities) == 0:
        return labels
    best = similarities.argmax(axis=1)
    best_score = similarities[np.arange(len(similarities)), best]
    if similarities.shape[1] > 1:
        runner_up = np.sort(similarities, axis=1)[:, -2]
    else:
        runner_up = np.full(len(similarities), -np.inf)
    confident = (
        (best_score >= min_similarity)
        & (best_score - runner_up >= min_margin)
        & (speaker_map.loudness_db >= silence_db)
    )
    labels[confident] = best[confident]
    return labels


def label_cells(window_labels: np.ndarray, windows_per_cell: int) -> np.ndarray:
    """Label each hop-length cell of time, keeping a label only if every window covering it agrees.

    Windows are windows_per_cell hops long, so cell j is covered by windows j - windows_per_cell + 1
    through j. Requiring agreement means a cell next to a different or unknown window is dropped,
    which keeps audio near speaker changes out of the segments.
    """
    num_windows = len(window_labels)
    if num_windows == 0:
        return np.full(0, -1)
    num_cells = num_windows + windows_per_cell - 1
    cells = np.full(num_cells, -1)
    for j in range(num_cells):
        covering = window_labels[max(0, j - windows_per_cell + 1) : min(num_windows, j + 1)]
        if covering[0] >= 0 and np.all(covering == covering[0]):
            cells[j] = covering[0]
    return cells


def find_segments(
    speaker_map: SpeakerMap,
    min_similarity: float = MIN_SIMILARITY,
    min_margin: float = MIN_MARGIN,
    edge_trim_seconds: float = EDGE_TRIM_SECONDS,
    min_segment_seconds: float = MIN_SEGMENT_SECONDS,
) -> list[Segment]:
    """Turn a map into confident single-speaker segments."""
    windows_per_cell = round(WINDOW_SECONDS / HOP_SECONDS)
    cells = label_cells(label_windows(speaker_map, min_similarity, min_margin), windows_per_cell)
    segments = []
    run_start = 0
    for j in range(1, len(cells) + 1):
        if j < len(cells) and cells[j] == cells[run_start]:
            continue
        if cells[run_start] >= 0:
            segment = Segment(
                speaker=speaker_map.speakers[cells[run_start]],
                start=run_start * HOP_SECONDS + edge_trim_seconds,
                end=j * HOP_SECONDS - edge_trim_seconds,
            )
            if segment.duration >= min_segment_seconds:
                segments.append(segment)
        run_start = j
    return segments


def extract_segments(audio_path: str | Path, segments: list[Segment], output_dir: str | Path) -> None:
    """Write each segment as a mono WAV at the source sample rate, in output_dir/<speaker>/."""
    if not segments:
        return
    waveform, sample_rate = load_mono(audio_path)
    stem = Path(audio_path).stem
    for segment in segments:
        speaker_dir = Path(output_dir) / segment.speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        audio = waveform[int(segment.start * sample_rate) : int(segment.end * sample_rate)]
        sf.write(speaker_dir / f"{stem} - {segment.start:08.2f}.wav", audio.numpy(), sample_rate)


# ---------------------------------------------------------------- command line


def audio_files(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _parse_speaker(value: str) -> tuple[str, Path]:
    name, _, folder = value.partition("=")
    if not name or not folder:
        raise argparse.ArgumentTypeError("Use the form name=folder, for example pizarro=data/audio/pizarro.")
    return name, Path(folder)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    enroll_parser = commands.add_parser("enroll", help="Build voiceprints from folders of clips")
    enroll_parser.add_argument("--speaker", type=_parse_speaker, action="append", required=True, help="name=folder")
    enroll_parser.add_argument("--voiceprints", type=Path, required=True, help="Output .npz file")

    map_parser = commands.add_parser("map", help="Write a CSV speaker map for every audio file in a folder")
    map_parser.add_argument("audio_dir", type=Path)
    map_parser.add_argument("--voiceprints", type=Path, required=True)
    map_parser.add_argument("--maps", type=Path, required=True, help="Folder for the CSV maps")
    map_parser.add_argument("--overwrite", action="store_true", help="Redo files that already have a map")

    extract_parser = commands.add_parser("extract", help="Write confident single-speaker segments as WAV files")
    extract_parser.add_argument("audio_dir", type=Path)
    extract_parser.add_argument("--maps", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True, help="Folder for per-speaker WAV files")
    extract_parser.add_argument("--min-similarity", type=float, default=MIN_SIMILARITY)
    extract_parser.add_argument("--min-margin", type=float, default=MIN_MARGIN)

    args = parser.parse_args(argv)

    if args.command == "enroll":
        voiceprints = enroll(dict(args.speaker), SpeakerEmbedder())
        save_voiceprints(args.voiceprints, voiceprints)
        print(f"Saved voiceprints for {', '.join(voiceprints)} to {args.voiceprints}")

    elif args.command == "map":
        voiceprints = load_voiceprints(args.voiceprints)
        args.maps.mkdir(parents=True, exist_ok=True)
        todo = [f for f in audio_files(args.audio_dir) if args.overwrite or not (args.maps / f"{f.stem}.csv").exists()]
        embedder = SpeakerEmbedder()
        for file in tqdm(todo, desc="Mapping", unit="file"):
            map_file(file, voiceprints, embedder).save(args.maps / f"{file.stem}.csv")

    elif args.command == "extract":
        totals: dict[str, float] = {}
        for file in tqdm(audio_files(args.audio_dir), desc="Extracting", unit="file"):
            map_path = args.maps / f"{file.stem}.csv"
            if not map_path.exists():
                continue
            segments = find_segments(SpeakerMap.load(map_path), args.min_similarity, args.min_margin)
            extract_segments(file, segments, args.output)
            for segment in segments:
                totals[segment.speaker] = totals.get(segment.speaker, 0.0) + segment.duration
        for speaker, seconds in sorted(totals.items()):
            print(f"{speaker}: {seconds / 3600:.2f} hours extracted")


if __name__ == "__main__":
    main()
