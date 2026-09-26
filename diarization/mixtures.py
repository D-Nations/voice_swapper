"""Synthetic conversations with exact labels, for training and testing speaker detection.

Real episodes have no frame-accurate labels, but there are hours of clean single-speaker clips.
make_mixture splices clips of the two hosts into a short conversation with handoffs, pauses,
backchannels and overlapping speech, and records which host is talking in every 20 ms frame.
Some mixtures also include "other" audio (music, quoted clips, guests), which is labeled as neither host.

Speaker clips come from the RVC clip folders (train clips for training, held-out test clips for
testing). Other audio is cut from stretches of the episodes that match neither host's voiceprint.

Build the other-audio folders with:
    python -m diarization.mixtures --index data/speaker_maps/segments.csv
        --audio-dir data/audio/very_bad_wizards --maps data/speaker_maps/maps --output data/detector/other
"""

import argparse
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from diarization.embedding import EMBEDDING_SAMPLE_RATE, HOP_SECONDS, SILENCE_DB, WINDOW_SECONDS, load_mono
from diarization.select_clips import MIN_EPISODE_KBPS, eligible_episodes, held_out_episodes, load_index
from diarization.speaker_maps import SpeakerMap, audio_files

SAMPLE_RATE = EMBEDDING_SAMPLE_RATE
FRAME_SECONDS = 0.02  # One label per 20 ms, WavLM's frame rate.
FRAME = round(FRAME_SECONDS * SAMPLE_RATE)

# A frame of a clip counts as speech when it's no more than SPEECH_BELOW_PEAK_DB quieter than the
# clip's loudest frame. Gaps shorter than FILL_SECONDS, like stop consonants, count as speech too.
SPEECH_BELOW_PEAK_DB = 30.0
FILL_SECONDS = 0.2

MIXTURE_SECONDS = 8.0
OTHER_CHANCE = 0.1  # Chance each chunk is other audio instead of a host.
SHORT_CHANCE = 0.2  # Chance a host chunk is a short backchannel, like "yeah" or a laugh.
SWITCH_CHANCE = 0.8  # Chance the next chunk is the other host.
OVERLAP_CHANCE = 0.3  # Chance the next chunk starts before this one ends.
LONG_PAUSE_CHANCE = 0.15

# Other audio: stretches at least OTHER_MIN_SECONDS long whose windows all match neither host better
# than OTHER_MAX_SIMILARITY, at most OTHER_SECONDS_PER_EPISODE from each episode.
OTHER_MAX_SIMILARITY = 0.15
OTHER_MIN_SECONDS = 3.0
OTHER_SECONDS_PER_EPISODE = 30.0


@dataclass
class Clip:
    audio: np.ndarray  # 16 kHz mono float32.
    speech: np.ndarray  # One bool per FRAME samples.


def speech_frames(
    audio: np.ndarray, below_peak_db: float = SPEECH_BELOW_PEAK_DB, fill_seconds: float = FILL_SECONDS
) -> np.ndarray:
    """Which FRAME-long frames of a clip are speech, from their level relative to the loudest frame."""
    count = len(audio) // FRAME
    if count == 0:
        return np.zeros(0, dtype=bool)
    frames = audio[: count * FRAME].reshape(count, FRAME)
    level = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-10)
    speech = level >= level.max() - below_peak_db
    # Fill short gaps between speech frames.
    fill = round(fill_seconds / FRAME_SECONDS)
    indices = np.flatnonzero(speech)
    for a, b in pairwise(indices):
        if 1 < b - a <= fill + 1:
            speech[a:b] = True
    return speech


def load_clips(folder: Path) -> list[Clip]:
    clips = []
    for path in audio_files(folder):
        audio = load_mono(path, SAMPLE_RATE)[0].numpy()
        clips.append(Clip(audio, speech_frames(audio)))
    return clips


def load_other(folder: Path) -> list[np.ndarray]:
    return [load_mono(path, SAMPLE_RATE)[0].numpy() for path in audio_files(folder)]


def make_mixture(
    speakers: list[list[Clip]],
    other: list[np.ndarray],
    rng: np.random.Generator,
    seconds: float = MIXTURE_SECONDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A synthetic conversation between the speakers.

    Returns the audio, a (frames, speakers) bool array of who is talking in each frame, and a
    (frames, speakers) array of each speaker's level in each frame, for telling who dominates overlaps.
    """
    samples = round(seconds * SAMPLE_RATE)
    frames = samples // FRAME
    mix = np.zeros(samples, dtype=np.float32)
    active = np.zeros((frames, len(speakers)), dtype=bool)
    energy = np.zeros((frames, len(speakers)), dtype=np.float32)

    t = rng.uniform(-1.0, 0.5)
    speaker = int(rng.integers(len(speakers)))
    while t < seconds:
        if other and rng.random() < OTHER_CHANCE:
            audio = other[rng.integers(len(other))]
            length = min(len(audio), round(rng.uniform(1.0, 4.0) * SAMPLE_RATE))
            offset = int(rng.integers(len(audio) - length + 1))
            chunk, speech, who = audio[offset : offset + length], None, None
        else:
            clip = speakers[speaker][rng.integers(len(speakers[speaker]))]
            wanted = rng.uniform(0.3, 1.2) if rng.random() < SHORT_CHANCE else rng.uniform(1.5, 6.0)
            # Start and end on frame boundaries so the speech labels line up.
            length = min(len(clip.speech), round(wanted / FRAME_SECONDS)) * FRAME
            offset = int(rng.integers(len(clip.speech) - length // FRAME + 1)) * FRAME
            chunk = clip.audio[offset : offset + length]
            speech, who = clip.speech[offset // FRAME : (offset + length) // FRAME], speaker
        chunk = chunk * np.float32(10 ** (rng.uniform(-6.0, 3.0) / 20))

        start = round(t / FRAME_SECONDS) * FRAME
        a, b = max(start, 0), min(start + len(chunk), samples)
        if b > a:
            mix[a:b] += chunk[a - start : b - start]
            if who is not None and speech is not None:
                fa, fb = a // FRAME, min(b // FRAME, frames)
                piece = chunk[a - start : a - start + (fb - fa) * FRAME].reshape(fb - fa, FRAME)
                energy[fa:fb, who] += np.mean(piece**2, axis=1)
                active[fa:fb, who] |= speech[(a - start) // FRAME : (a - start) // FRAME + fb - fa]

        duration = len(chunk) / SAMPLE_RATE
        roll = rng.random()
        if roll < OVERLAP_CHANCE:
            gap = -rng.uniform(0.1, min(1.5, 0.8 * duration))
        elif roll < OVERLAP_CHANCE + LONG_PAUSE_CHANCE:
            gap = rng.uniform(0.6, 2.0)
        else:
            gap = rng.uniform(0.0, 0.6)
        t += duration + gap
        if rng.random() < SWITCH_CHANCE:
            speaker = (speaker + 1) % len(speakers)

    # Faint room noise, so pauses aren't digital silence, then a random overall level.
    mix += rng.normal(0, 10 ** (rng.uniform(-75, -55) / 20), samples).astype(np.float32)
    peak = np.abs(mix).max()
    if peak > 0:
        mix *= np.float32(rng.uniform(0.3, 0.9) / peak)
    return mix, active, energy


def other_stretches(speaker_map: SpeakerMap) -> list[tuple[float, float]]:
    """Loud stretches of an episode where every window matches neither host."""
    other = (speaker_map.similarities.max(axis=1) < OTHER_MAX_SIMILARITY) & (speaker_map.loudness_db >= SILENCE_DB)
    stretches = []
    start = None
    for i, flag in enumerate([*other, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            begin, end = start * HOP_SECONDS, (i - 1) * HOP_SECONDS + WINDOW_SECONDS
            if end - begin >= OTHER_MIN_SECONDS:
                stretches.append((begin, end))
            start = None
    return stretches


def write_other_audio(index: Path, audio_dir: Path, maps_dir: Path, output: Path) -> None:
    """Cut other audio from training episodes into output/train and held-out episodes into output/test."""
    episodes = {item.audio_file for item in load_index(index)}
    eligible = eligible_episodes(audio_dir, episodes, MIN_EPISODE_KBPS)
    held_out = held_out_episodes(sorted(eligible))
    for name in tqdm(sorted(eligible), desc="Cutting other audio", unit="episode"):
        stretches = other_stretches(SpeakerMap.load(maps_dir / f"{Path(name).stem}.csv"))
        if not stretches:
            continue
        audio = load_mono(audio_dir / name, SAMPLE_RATE)[0].numpy()
        folder = output / ("test" if name in held_out else "train")
        folder.mkdir(parents=True, exist_ok=True)
        left = OTHER_SECONDS_PER_EPISODE
        for begin, end in stretches:
            end = min(end, begin + left)
            sf.write(
                folder / f"{Path(name).stem} - {begin:08.2f}.wav",
                audio[round(begin * SAMPLE_RATE) : round(end * SAMPLE_RATE)],
                SAMPLE_RATE,
            )
            left -= end - begin
            if left <= OTHER_MIN_SECONDS:
                break


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True, help="Segment index CSV from speaker_maps index")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True, help="Folder of speaker map CSVs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_other_audio(args.index, args.audio_dir, args.maps, args.output)
    for split in ("train", "test"):
        files = list((args.output / split).glob("*.wav"))
        seconds = sum(sf.info(f).duration for f in files)
        print(f"{split}: {len(files)} stretches, {seconds / 60:.1f} minutes")


if __name__ == "__main__":
    main()
