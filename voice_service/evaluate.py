"""Compare a voice's saved epochs by converting the other speaker's held-out clips.

Each converted clip is scored two ways:
- Speaker similarity: cosine similarity of its ECAPA embedding to the target's voiceprint (higher is
  better) and to the source speaker's (lower is better).
- Word error rate: Whisper's transcript of the conversion against its transcript of the original,
  which catches epochs that sound like the target but garble the words.

Writes one row per clip and epoch to scores_<voice>.csv and prints the mean scores per epoch.

Run with: python -m voice_service.evaluate [--clips 20] [--step 20]
Print the summary of earlier runs with: python -m voice_service.evaluate --report
"""

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import cast

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from diarization.embedding import SILENCE_DB, SpeakerEmbedder, embed_waveform
from diarization.speaker_maps import load_voiceprints
from rvc.infer.infer import VoiceConverter, load_voice
from rvc.infer.pipeline import SAMPLE_RATE, ConversionSettings, median_pitch
from rvc.lib.audio import load_audio

MODELS_DIR = Path("data/rvc/models")
TEST_DIR = Path("data/rvc/test")
VOICEPRINTS = Path("data/speaker_maps/voiceprints.npz")
OUTPUT_DIR = Path("data/rvc/evaluation")

# Each voice is evaluated on the other speaker's clips, since that's how it will be used.
SOURCE_SPEAKER = {"pizarro": "sommers", "sommers": "pizarro"}
CLIPS_PER_VOICE = 20
EPOCH_STEP = 20
SEED = 0
ASR_MODEL = "openai/whisper-small.en"

MODEL_FILE = re.compile(r"_(\d+)e_\d+s\.pth$")


@dataclass(frozen=True)
class ClipScore:
    voice: str
    epoch: int
    clip: str
    target_similarity: float
    source_similarity: float
    word_error_rate: float
    reference_text: str  # Transcript of the original clip.
    converted_text: str  # Transcript of the conversion.


@dataclass(frozen=True)
class EpochSummary:
    voice: str
    epoch: int
    target_similarity: float
    source_similarity: float
    word_error_rate: float


def normalize_words(text: str) -> list[str]:
    """Lowercase words without punctuation, so transcripts compare on wording alone."""
    return re.findall(r"[a-z0-9']+", text.lower().replace("-", " "))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance divided by the reference's length. 0 is a perfect match."""
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1] / len(ref)


def voice_models(experiment_dir: Path) -> dict[int, Path]:
    """The exported voice models in a training folder, by epoch."""
    models = {}
    for path in experiment_dir.glob("*.pth"):
        match = MODEL_FILE.search(path.name)
        if match is not None:
            models[int(match.group(1))] = path
    return dict(sorted(models.items()))


def choose_epochs(available: list[int], step: int) -> list[int]:
    """Every epoch divisible by step, plus the last one."""
    chosen = [epoch for epoch in available if epoch % step == 0]
    if available and available[-1] not in chosen:
        chosen.append(available[-1])
    return chosen


def voice_median_pitch(experiment_dir: Path) -> float:
    """Median pitch in Hz of a voice's training data, from the pitch tracks saved during extraction."""
    f0 = np.concatenate([np.load(path) for path in (experiment_dir / "f0_voiced").glob("*.npy")])
    pitch = median_pitch(f0)
    if pitch is None:
        raise ValueError(f"No voiced frames in {experiment_dir / 'f0_voiced'}.")
    return pitch


def sample_clips(folder: Path, count: int, seed: int = SEED) -> list[Path]:
    clips = sorted(folder.glob("*.wav"))
    chosen = np.random.default_rng(seed).choice(len(clips), size=min(count, len(clips)), replace=False)
    return [clips[i] for i in sorted(chosen)]


class Transcriber:
    """English speech to text with Whisper."""

    def __init__(self, device: torch.device, model_name: str = ASR_MODEL) -> None:
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).eval()
        # Module.to moves the weights in place. Calling it through Module avoids transformers' wrapped
        # signature, which type checkers misread.
        cast(torch.nn.Module, self.model).to(device)

    def __call__(self, audio: np.ndarray) -> str:
        """Transcribe up to 30 seconds of mono 16 kHz audio."""
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            # Without timestamps, the English-only models sometimes stop after the first word.
            tokens = self.model.generate(
                inputs.input_features.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                return_timestamps=True,
            )
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


class Scorer:
    """Speaker similarity and word error rate for converted clips."""

    def __init__(self, voiceprints: dict[str, np.ndarray], device: torch.device) -> None:
        self.voiceprints = voiceprints
        self.embedder = SpeakerEmbedder(str(device))
        self.transcribe = Transcriber(device)

    def similarity(self, audio: np.ndarray, speaker: str) -> float:
        """Cosine similarity of 16 kHz audio's mean speaker embedding, over its non-silent windows, to a voiceprint."""
        windows = embed_waveform(torch.from_numpy(audio).float(), self.embedder)
        loud = windows.embeddings[windows.loudness_db > SILENCE_DB]
        embedding = (loud if len(loud) else windows.embeddings).mean(axis=0)
        return float(embedding @ self.voiceprints[speaker] / np.linalg.norm(embedding))


def evaluate_voice(
    voice: str,
    converter: VoiceConverter,
    scorer: Scorer,
    clips: list[Path],
    epochs: list[int],
    models_dir: Path = MODELS_DIR,
) -> list[ClipScore]:
    experiment_dir = models_dir / voice
    models = voice_models(experiment_dir)
    index_path = experiment_dir / f"{voice}.index"
    settings = ConversionSettings(target_pitch_hz=voice_median_pitch(experiment_dir))
    source = SOURCE_SPEAKER[voice]

    originals = {clip: load_audio(clip, SAMPLE_RATE) for clip in clips}
    references = {clip: scorer.transcribe(audio) for clip, audio in originals.items()}

    scores = []
    for epoch in tqdm(epochs, desc=voice):
        # Loaded without the converter's cache, which would keep every epoch in GPU memory.
        loaded = load_voice(models[epoch], index_path if index_path.exists() else None, converter.device)
        for clip, audio in originals.items():
            converted = converter.convert(audio, loaded, settings)
            converted = librosa.resample(converted, orig_sr=loaded.sample_rate, target_sr=SAMPLE_RATE)
            converted_text = scorer.transcribe(converted)
            scores.append(
                ClipScore(
                    voice=voice,
                    epoch=epoch,
                    clip=clip.name,
                    target_similarity=scorer.similarity(converted, voice),
                    source_similarity=scorer.similarity(converted, source),
                    word_error_rate=word_error_rate(references[clip], converted_text),
                    reference_text=references[clip],
                    converted_text=converted_text,
                )
            )
    return scores


def summarize(scores: list[ClipScore]) -> list[EpochSummary]:
    """Mean scores per voice and epoch."""
    groups: dict[tuple[str, int], list[ClipScore]] = defaultdict(list)
    for score in scores:
        groups[(score.voice, score.epoch)].append(score)
    return [
        EpochSummary(
            voice=voice,
            epoch=epoch,
            target_similarity=float(np.mean([s.target_similarity for s in group])),
            source_similarity=float(np.mean([s.source_similarity for s in group])),
            word_error_rate=float(np.mean([s.word_error_rate for s in group])),
        )
        for (voice, epoch), group in sorted(groups.items())
    ]


def write_scores(path: Path, scores: list[ClipScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[field.name for field in fields(ClipScore)])
        writer.writeheader()
        writer.writerows(asdict(score) for score in scores)


def read_scores(path: Path) -> list[ClipScore]:
    with open(path, newline="", encoding="utf-8") as file:
        return [
            ClipScore(
                voice=row["voice"],
                epoch=int(row["epoch"]),
                clip=row["clip"],
                target_similarity=float(row["target_similarity"]),
                source_similarity=float(row["source_similarity"]),
                word_error_rate=float(row["word_error_rate"]),
                reference_text=row["reference_text"],
                converted_text=row["converted_text"],
            )
            for row in csv.DictReader(file)
        ]


def scores_path(output_dir: Path, voice: str) -> Path:
    return output_dir / f"scores_{voice}.csv"


def print_summary(summaries: list[EpochSummary]) -> None:
    print(f"{'voice':8} {'epoch':>5} {'target sim':>10} {'source sim':>10} {'WER':>6}")
    for s in summaries:
        print(
            f"{s.voice:8} {s.epoch:5d} {s.target_similarity:10.3f} {s.source_similarity:10.3f} {s.word_error_rate:6.1%}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voices", nargs="+", choices=sorted(SOURCE_SPEAKER), default=sorted(SOURCE_SPEAKER))
    parser.add_argument("--clips", type=int, default=CLIPS_PER_VOICE, help="Held-out clips per voice")
    parser.add_argument("--step", type=int, default=EPOCH_STEP, help="Evaluate every this many epochs")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Folder for scores_<voice>.csv")
    parser.add_argument("--report", action="store_true", help="Summarize earlier runs instead of evaluating")
    args = parser.parse_args(argv)

    if not args.report:
        converter = VoiceConverter()
        scorer = Scorer(load_voiceprints(VOICEPRINTS), converter.device)
        for voice in args.voices:
            clips = sample_clips(TEST_DIR / SOURCE_SPEAKER[voice], args.clips)
            epochs = choose_epochs(list(voice_models(MODELS_DIR / voice)), args.step)
            write_scores(scores_path(args.output_dir, voice), evaluate_voice(voice, converter, scorer, clips, epochs))

    paths = [scores_path(args.output_dir, voice) for voice in args.voices]
    print_summary(summarize([score for path in paths if path.exists() for score in read_scores(path)]))
    print(f"Per-clip scores are in {args.output_dir}")


if __name__ == "__main__":
    main()
