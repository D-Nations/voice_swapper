"""Swap the two hosts' voices across a whole episode.

The episode is split into speaker turns (diarization.labelers), and every turn is converted to the
other host's voice. Silence and short unrecognized sounds (laughs, crosstalk) join the neighbouring
turn. Long stretches that match neither host, like music or quoted clips, keep the original audio.

--method picks how turns are found: "map" uses the episode's ECAPA speaker map as is, "refined"
re-places each handoff with short windows, and "detector" uses the trained frame detector.

Each turn is converted with CONTEXT_SECONDS of the audio around it, which is then cropped off, so
short turns still give the model enough to work with. Each direction uses a fixed pitch shift, from
the source speaker's median training pitch to the target's, so a speaker's intonation is kept from
turn to turn.

Run with:
    python -m voice_service.swap_episode --episode "data/audio/very_bad_wizards/<episode>.mp3"
        [--method refined] [--start 0 --duration 300] [--output data/rvc/swaps/<episode>.wav]
        [--voice pizarro=pizarro_60min:400 --voice sommers=sommers_60min:400]

--voice converts a speaker's turns with a model other than voices.py's default: the training folder
under data/rvc/models and the epoch to use.
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from diarization.embedding import EMBEDDING_SAMPLE_RATE
from diarization.labelers import METHODS, Labeler
from diarization.speaker_maps import SpeakerMap
from diarization.turns import Turn, clip_turns
from rvc.infer.infer import VoiceConverter
from rvc.infer.pipeline import SAMPLE_RATE, ConversionSettings, semitones_between
from rvc.lib.audio import load_audio
from voice_service.evaluate import voice_median_pitch
from voice_service.voices import MODELS_DIR, VOICES, VoiceChoice

MAPS_DIR = Path("data/speaker_maps/maps")
OUTPUT_DIR = Path("data/rvc/swaps")
SWAP = {"pizarro": "sommers", "sommers": "pizarro"}
DEFAULT_METHOD = "map"

CONTEXT_SECONDS = 0.5
CROSSFADE_SECONDS = 0.03
MIN_TURN_SECONDS = 0.1  # Shorter turns keep the original audio.


def crossfade_into(output: np.ndarray, piece: np.ndarray, start: int, fade: int) -> None:
    """Write piece into output at start, fading in over the first fade samples on top of what's there."""
    end = start + len(piece)
    fade = min(fade, len(piece), start)
    if fade > 0:
        ramp = np.linspace(0, 1, fade, dtype=output.dtype)
        output[start : start + fade] = output[start : start + fade] * (1 - ramp) + piece[:fade] * ramp
    output[start + fade : end] = piece[fade:]


def swap_episode(
    episode: Path,
    output: Path,
    start: float = 0.0,
    duration: float | None = None,
    method: str = DEFAULT_METHOD,
    voices: dict[str, VoiceChoice] | None = None,
    maps_dir: Path = MAPS_DIR,
    models_dir: Path = MODELS_DIR,
    device: str | None = None,
) -> list[Turn]:
    """Write the episode, or the part from start for duration seconds, with the hosts' voices swapped.

    voices maps a speaker to the voice their turns are converted *to*, overriding voices.py.
    """
    speaker_map = SpeakerMap.load(maps_dir / f"{episode.stem}.csv")
    labeler = Labeler(speaker_map.speakers, device=device)
    audio_16k = load_audio(episode, EMBEDDING_SAMPLE_RATE).astype(np.float32)
    turns = labeler.turns(method, audio_16k, speaker_map)
    del audio_16k

    converter = VoiceConverter(device)
    choices = {name: VOICES[name] for name in SWAP} | (voices or {})
    for choice in choices.values():
        print(f"{choice.label}: {choice.key}, epoch {choice.epoch}, index rate {choice.index_rate}")
    loaded = {
        name: converter.voice(choice.model_path(models_dir), choice.index_path(models_dir))
        for name, choice in choices.items()
    }
    rates = {voice.sample_rate for voice in loaded.values()}
    if len(rates) != 1:
        raise ValueError(f"The voices use different sample rates: {sorted(rates)}.")
    rate = rates.pop()
    pitch = {name: voice_median_pitch(models_dir / choices[name].key) for name in SWAP}
    settings = {
        source: ConversionSettings(
            pitch_shift=semitones_between(pitch[source], pitch[target]), index_rate=choices[target].index_rate
        )
        for source, target in SWAP.items()
    }

    audio = load_audio(episode, rate).astype(np.float32)
    first = round(start * rate)
    last = len(audio) if duration is None else min(len(audio), round((start + duration) * rate))
    audio = audio[first:last]
    turns = clip_turns(turns, start, start + len(audio) / rate)

    result = audio.copy()
    fade = round(CROSSFADE_SECONDS * rate)
    context = round(CONTEXT_SECONDS * rate)
    for turn in tqdm(turns, desc="Converting turns", unit="turn"):
        a, b = round(turn.start * rate), round(turn.end * rate)
        if turn.speaker < 0 or b - a < MIN_TURN_SECONDS * rate:
            piece = audio[a:b]
        else:
            source = speaker_map.speakers[turn.speaker]
            lo, hi = max(0, a - context), min(len(audio), b + context)
            clip = librosa.resample(audio[lo:hi], orig_sr=rate, target_sr=SAMPLE_RATE, res_type="soxr_hq")
            converted = converter.convert(clip, loaded[SWAP[source]], settings[source]).astype(np.float32)
            converted = np.pad(converted, (0, max(0, (hi - lo) - len(converted))))
            piece = converted[a - lo : b - lo]
        crossfade_into(result, piece, a, fade)

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, result, rate)
    return turns


def parse_voices(values: list[str]) -> dict[str, VoiceChoice]:
    """--voice values like pizarro=pizarro_60min:400, as voice choices keeping voices.py's other settings."""
    voices = {}
    for value in values:
        speaker, _, model = value.partition("=")
        folder, _, epoch = model.partition(":")
        if speaker not in VOICES or not folder or not epoch.isdigit():
            raise ValueError(f"--voice {value!r} should look like pizarro=pizarro_60min:400.")
        voices[speaker] = replace(VOICES[speaker], key=folder, epoch=int(epoch))
    return voices


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=Path, required=True, help="The episode's audio file")
    parser.add_argument("--output", type=Path, help=f"WAV file to write. Defaults to {OUTPUT_DIR}/<episode>.wav")
    parser.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD, help="How to find speaker turns")
    parser.add_argument(
        "--voice",
        action="append",
        default=[],
        metavar="SPEAKER=FOLDER:EPOCH",
        help="Use another model for a voice, like pizarro=pizarro_60min:400. Repeat for each voice.",
    )
    parser.add_argument("--start", type=float, default=0.0, help="Seconds into the episode to start")
    parser.add_argument("--duration", type=float, help="Seconds to convert. Defaults to the rest of the episode")
    parser.add_argument("--device", help="Such as cuda:0 or cpu. Defaults to the GPU if there is one.")
    args = parser.parse_args(argv)

    output = args.output or OUTPUT_DIR / f"{args.episode.stem}.wav"
    started = time.time()
    turns = swap_episode(
        args.episode, output, args.start, args.duration, args.method, parse_voices(args.voice), device=args.device
    )
    kept = sum(t.end - t.start for t in turns if t.speaker < 0)
    total = sum(t.end - t.start for t in turns)
    print(f"Wrote {output} in {time.time() - started:.0f} seconds.")
    print(f"{sum(t.speaker >= 0 for t in turns)} turns swapped. {kept:.0f} of {total:.0f} seconds kept as original.")


if __name__ == "__main__":
    main()
