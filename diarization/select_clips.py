"""Choose RVC training and test clips for each speaker from the segment index, and write them out.

The rules follow Applio's dataset guide (docs.applio.org/guides/how-create-datasets):

- 10 to 30 minutes of speech per voice, since more gives diminishing returns. TRAIN_MINUTES is 30.
- Clips of 10 to 15 seconds. Longer segments are cut to their first 15 seconds.
- A wide range of tones and moods. Clips are drawn at random from those matching the speaker's
  voiceprint at least as well as the speaker's median clip, rather than always taking the top
  scorers, and each episode contributes at most MAX_TRAIN_SECONDS_PER_EPISODE.
- Silence trimmed from the start and end of every clip.

Only episodes published at MIN_EPISODE_KBPS or higher are used. The early episodes were published
at 64 kbps, and the compression artifacts at that bitrate would be learned as part of each voice.

Every HELD_OUT_EVERY-th eligible episode is reserved for testing, so test clips always come from
episodes the models never trained on. Random draws use a fixed seed, so reruns pick the same clips.

Run with:
    python -m diarization.select_clips --index data/speaker_maps/segments.csv
        --audio-dir data/audio/very_bad_wizards --output data/rvc
which writes data/rvc/train/<speaker>/*.wav and data/rvc/test/<speaker>/*.wav.
"""

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from diarization.embedding import load_mono
from diarization.speaker_maps import Segment

TRAIN_MINUTES = 30.0
TEST_CLIPS = 100  # Per speaker, from held-out episodes only.
HELD_OUT_EVERY = 10  # Every 10th eligible episode, in sorted order, is held out for testing.
MIN_EPISODE_KBPS = 128  # Most episodes from 2018 on. Earlier ones are 64 kbps.
SEED = 0

MIN_CLIP_SECONDS = 10.0
MAX_CLIP_SECONDS = 15.0

# Clips are drawn only from segments at or above this quantile of the speaker's match scores.
MIN_SIMILARITY_QUANTILE = 0.5

MAX_TRAIN_SECONDS_PER_EPISODE = 30.0
MAX_TEST_CLIPS_PER_EPISODE = 6

# Silence trimming: frames more than SILENCE_BELOW_PEAK_DB quieter than the clip's loudest frame
# count as silence. SILENCE_PAD_SECONDS of audio is kept around the speech so words aren't clipped.
SILENCE_FRAME_SECONDS = 0.02
SILENCE_BELOW_PEAK_DB = 35.0
SILENCE_PAD_SECONDS = 0.1


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
    rng: np.random.Generator,
    budget_seconds: float = float("inf"),
    max_clips: int | None = None,
    max_seconds_per_episode: float = float("inf"),
    max_clips_per_episode: int | None = None,
    min_similarity_quantile: float = MIN_SIMILARITY_QUANTILE,
) -> list[IndexedSegment]:
    """Draw clips at random from the long, well-matching candidates until a budget runs out.

    Candidates shorter than MIN_CLIP_SECONDS are dropped and longer ones are cut to MAX_CLIP_SECONDS.
    Of the rest, only those at or above min_similarity_quantile of the match scores are eligible.
    """
    long_enough = [
        replace(item, segment=replace(item.segment, end=min(item.segment.end, item.segment.start + MAX_CLIP_SECONDS)))
        for item in candidates
        if item.segment.duration >= MIN_CLIP_SECONDS
    ]
    if not long_enough:
        return []
    threshold = np.quantile([item.segment.similarity for item in long_enough], min_similarity_quantile)
    pool = [item for item in long_enough if item.segment.similarity >= threshold]

    chosen = []
    total = 0.0
    seconds_by_episode: dict[str, float] = defaultdict(float)
    clips_by_episode: dict[str, int] = defaultdict(int)
    for index in rng.permutation(len(pool)):
        item = pool[index]
        duration = item.segment.duration
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
    seed: int = SEED,
) -> dict[str, dict[str, list[IndexedSegment]]]:
    """Return {"train": {speaker: clips}, "test": {speaker: clips}}, using only eligible episodes if given."""
    if eligible is not None:
        index = [item for item in index if item.audio_file in eligible]
    held_out = held_out_episodes([item.audio_file for item in index])
    rng = np.random.default_rng(seed)
    speakers = sorted({item.segment.speaker for item in index})
    sets: dict[str, dict[str, list[IndexedSegment]]] = {"train": {}, "test": {}}
    for speaker in speakers:
        mine = [item for item in index if item.segment.speaker == speaker]
        sets["train"][speaker] = choose_clips(
            [item for item in mine if item.audio_file not in held_out],
            rng,
            budget_seconds=train_minutes * 60,
            max_seconds_per_episode=MAX_TRAIN_SECONDS_PER_EPISODE,
        )
        sets["test"][speaker] = choose_clips(
            [item for item in mine if item.audio_file in held_out],
            rng,
            max_clips=test_clips,
            max_clips_per_episode=MAX_TEST_CLIPS_PER_EPISODE,
        )
    return sets


def trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Cut silence from both ends, keeping SILENCE_PAD_SECONDS around the first and last speech."""
    frame = max(1, round(SILENCE_FRAME_SECONDS * sample_rate))
    num_frames = len(audio) // frame
    if num_frames == 0:
        return audio
    frames = audio[: num_frames * frame].reshape(num_frames, frame)
    level_db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-10)
    loud = np.nonzero(level_db >= level_db.max() - SILENCE_BELOW_PEAK_DB)[0]
    pad = round(SILENCE_PAD_SECONDS * sample_rate)
    start = max(0, loud[0] * frame - pad)
    end = min(len(audio), (loud[-1] + 1) * frame + pad)
    return audio[start:end]


def write_sets(sets: dict[str, dict[str, list[IndexedSegment]]], audio_dir: Path, output_dir: Path) -> None:
    """Cut each chosen clip from its episode, trim its silence, and write output_dir/<set>/<speaker>/*.wav."""
    for set_name, by_speaker in sets.items():
        by_file: dict[str, list[Segment]] = defaultdict(list)
        for clips in by_speaker.values():
            for item in clips:
                by_file[item.audio_file].append(item.segment)
        for audio_file, segments in tqdm(by_file.items(), desc=f"Writing {set_name}", unit="episode"):
            waveform, sample_rate = load_mono(audio_dir / audio_file)
            audio = waveform.numpy()
            for segment in segments:
                clip = trim_silence(
                    audio[int(segment.start * sample_rate) : int(segment.end * sample_rate)], sample_rate
                )
                speaker_dir = output_dir / set_name / segment.speaker
                speaker_dir.mkdir(parents=True, exist_ok=True)
                sf.write(speaker_dir / f"{Path(audio_file).stem} - {segment.start:08.2f}.wav", clip, sample_rate)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True, help="Segment index CSV from speaker_maps index")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Folder of episode audio files")
    parser.add_argument("--output", type=Path, required=True, help="Folder for train/ and test/ clips")
    parser.add_argument("--train-minutes", type=float, default=TRAIN_MINUTES)
    parser.add_argument("--test-clips", type=int, default=TEST_CLIPS)
    parser.add_argument("--min-kbps", type=float, default=MIN_EPISODE_KBPS, help="Skip lower-bitrate episodes")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    for set_name in ("train", "test"):
        if (args.output / set_name).exists() and any((args.output / set_name).rglob("*.wav")):
            parser.error(f"{args.output / set_name} already has clips. Remove it first to choose a new set.")

    index = load_index(args.index)
    all_episodes = {item.audio_file for item in index}
    eligible = eligible_episodes(args.audio_dir, all_episodes, args.min_kbps)
    print(f"{len(eligible)} of {len(all_episodes)} episodes are at least {args.min_kbps:g} kbps.")
    sets = select_sets(index, args.train_minutes, args.test_clips, eligible, args.seed)
    write_sets(sets, args.audio_dir, args.output)
    for set_name, by_speaker in sets.items():
        for speaker, clips in by_speaker.items():
            minutes = sum(item.segment.duration for item in clips) / 60
            episodes = len({item.audio_file for item in clips})
            print(
                f"{set_name} {speaker}: {len(clips)} clips, {minutes:.1f} minutes before trimming, from {episodes} episodes"
            )


if __name__ == "__main__":
    main()
