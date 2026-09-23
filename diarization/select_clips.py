"""Choose RVC training and test clips for each speaker from the segment index, and write them out.

Only episodes published at MIN_EPISODE_KBPS or higher are used. The early episodes were published
at 64 kbps, and the compression artifacts at that bitrate would be learned as part of each voice.

Every HELD_OUT_EVERY-th eligible episode is reserved for testing, so test clips always come from episodes the
models never trained on. Within each set, clips are taken in order of how strongly they match the
speaker's voiceprint, with a per-episode cap so the set spans many episodes and years.

Run with:
    python -m diarization.select_clips --index data/speaker_maps/segments.csv
        --audio-dir data/audio/very_bad_wizards --output data/rvc
which writes data/rvc/train/<speaker>/*.wav and data/rvc/test/<speaker>/*.wav.
"""

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from diarization.speaker_maps import Segment, extract_segments

TRAIN_MINUTES = 60.0  # RVC does well with 30 to 60 minutes of clean speech per voice.
TEST_CLIPS = 100  # Per speaker, from held-out episodes only.
HELD_OUT_EVERY = 10  # Every 10th eligible episode, in sorted order, is held out for testing.
MIN_EPISODE_KBPS = 128  # Most episodes from 2018 on. Earlier ones are 64 kbps.

# Clip length limits. Very short clips carry little voice, and very long ones are more likely to
# hide a brief interruption.
MIN_CLIP_SECONDS = 3.0
MAX_CLIP_SECONDS = 15.0

MAX_TRAIN_SECONDS_PER_EPISODE = 60.0
MAX_TEST_CLIPS_PER_EPISODE = 3


@dataclass(frozen=True)
class IndexedSegment:
    audio_file: str
    segment: Segment


def load_index(path: str | Path) -> list[IndexedSegment]:
    with open(path, newline="", encoding="utf-8") as file:
        return [
            IndexedSegment(
                audio_file=row["audio_file"],
                segment=Segment(row["speaker"], float(row["start"]), float(row["end"]), float(row["similarity"])),
            )
            for row in csv.DictReader(file)
        ]


def episode_kbps(path: str | Path) -> float:
    """Average bitrate of an audio file, from its size and duration."""
    return os.path.getsize(path) * 8 / sf.info(path).duration / 1000


def eligible_episodes(audio_dir: Path, audio_files: set[str], min_kbps: float = MIN_EPISODE_KBPS) -> set[str]:
    return {name for name in audio_files if episode_kbps(audio_dir / name) >= min_kbps}


def held_out_episodes(audio_files: list[str], every: int = HELD_OUT_EVERY) -> set[str]:
    """Pick every `every`-th episode in sorted order, so the choice is stable across runs."""
    return set(sorted(set(audio_files))[::every])


def choose_clips(
    candidates: list[IndexedSegment],
    budget_seconds: float = float("inf"),
    max_clips: int | None = None,
    max_seconds_per_episode: float = float("inf"),
    max_clips_per_episode: int | None = None,
) -> list[IndexedSegment]:
    """Take the best-matching clips first until a budget runs out, respecting per-episode caps."""
    chosen = []
    total = 0.0
    seconds_by_episode: dict[str, float] = defaultdict(float)
    clips_by_episode: dict[str, int] = defaultdict(int)
    for item in sorted(candidates, key=lambda c: c.segment.similarity, reverse=True):
        duration = item.segment.duration
        if not MIN_CLIP_SECONDS <= duration <= MAX_CLIP_SECONDS:
            continue
        if seconds_by_episode[item.audio_file] + duration > max_seconds_per_episode:
            continue
        if max_clips_per_episode is not None and clips_by_episode[item.audio_file] >= max_clips_per_episode:
            continue
        chosen.append(item)
        total += duration
        seconds_by_episode[item.audio_file] += duration
        clips_by_episode[item.audio_file] += 1
        if total >= budget_seconds or (max_clips is not None and len(chosen) >= max_clips):
            break
    return chosen


def select_sets(
    index: list[IndexedSegment],
    train_minutes: float = TRAIN_MINUTES,
    test_clips: int = TEST_CLIPS,
    eligible: set[str] | None = None,
) -> dict[str, dict[str, list[IndexedSegment]]]:
    """Return {"train": {speaker: clips}, "test": {speaker: clips}}, using only eligible episodes if given."""
    if eligible is not None:
        index = [item for item in index if item.audio_file in eligible]
    held_out = held_out_episodes([item.audio_file for item in index])
    speakers = sorted({item.segment.speaker for item in index})
    sets: dict[str, dict[str, list[IndexedSegment]]] = {"train": {}, "test": {}}
    for speaker in speakers:
        mine = [item for item in index if item.segment.speaker == speaker]
        sets["train"][speaker] = choose_clips(
            [item for item in mine if item.audio_file not in held_out],
            budget_seconds=train_minutes * 60,
            max_seconds_per_episode=MAX_TRAIN_SECONDS_PER_EPISODE,
        )
        sets["test"][speaker] = choose_clips(
            [item for item in mine if item.audio_file in held_out],
            max_clips=test_clips,
            max_clips_per_episode=MAX_TEST_CLIPS_PER_EPISODE,
        )
    return sets


def write_sets(sets: dict[str, dict[str, list[IndexedSegment]]], audio_dir: Path, output_dir: Path) -> None:
    """Cut each chosen clip from its episode into output_dir/<set>/<speaker>/."""
    for set_name, by_speaker in sets.items():
        by_file: dict[str, list[Segment]] = defaultdict(list)
        for clips in by_speaker.values():
            for item in clips:
                by_file[item.audio_file].append(item.segment)
        for audio_file, segments in tqdm(by_file.items(), desc=f"Writing {set_name}", unit="episode"):
            extract_segments(audio_dir / audio_file, segments, output_dir / set_name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True, help="Segment index CSV from speaker_maps index")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Folder of episode audio files")
    parser.add_argument("--output", type=Path, required=True, help="Folder for train/ and test/ clips")
    parser.add_argument("--train-minutes", type=float, default=TRAIN_MINUTES)
    parser.add_argument("--test-clips", type=int, default=TEST_CLIPS)
    parser.add_argument("--min-kbps", type=float, default=MIN_EPISODE_KBPS, help="Skip lower-bitrate episodes")
    args = parser.parse_args(argv)

    for set_name in ("train", "test"):
        if (args.output / set_name).exists() and any((args.output / set_name).rglob("*.wav")):
            parser.error(f"{args.output / set_name} already has clips. Remove it first to choose a new set.")

    index = load_index(args.index)
    all_episodes = {item.audio_file for item in index}
    eligible = eligible_episodes(args.audio_dir, all_episodes, args.min_kbps)
    print(f"{len(eligible)} of {len(all_episodes)} episodes are at least {args.min_kbps:g} kbps.")
    sets = select_sets(index, args.train_minutes, args.test_clips, eligible)
    write_sets(sets, args.audio_dir, args.output)
    for set_name, by_speaker in sets.items():
        for speaker, clips in by_speaker.items():
            minutes = sum(item.segment.duration for item in clips) / 60
            episodes = len({item.audio_file for item in clips})
            print(f"{set_name} {speaker}: {len(clips)} clips, {minutes:.1f} minutes, from {episodes} episodes")


if __name__ == "__main__":
    main()
