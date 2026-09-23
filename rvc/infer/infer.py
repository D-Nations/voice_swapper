"""Convert speech to a trained voice.

Run with:
    python -m rvc.infer.infer --model data/rvc/models/pizarro/pizarro_300e_28500s.pth
        --index data/rvc/models/pizarro/pizarro.index --input in.wav --output out.wav [--pitch 3]
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import soundfile as sf
import torch

from rvc.infer.pipeline import SAMPLE_RATE, ConversionSettings, Pipeline, RetrievalIndex
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.audio import load_audio
from rvc.lib.embedders import load_contentvec
from rvc.lib.predictors.rmvpe import RMVPE
from rvc.train.process.extract_model import VoiceModel, read_saved_config
from rvc.train.utils import from_legacy_names

SEGMENT_FRAMES = 32  # Only used in training, so any value works.
INPUT_PEAK = 0.95  # Louder inputs are scaled down to this peak before conversion.


@dataclass(frozen=True)
class Voice:
    """A loaded voice model and its retrieval index."""

    name: str
    net_g: Synthesizer
    sample_rate: int
    index: RetrievalIndex | None


def load_voice(model_path: Path, index_path: Path | None, device: torch.device) -> Voice:
    """Load an exported voice model, and its index if given, ready for conversion."""
    model: VoiceModel = torch.load(model_path, map_location="cpu", weights_only=True)
    # The saved speaker count can be the config default. The embedding's size is the real one.
    speakers = model["weight"]["emb_g.weight"].shape[0]
    saved = read_saved_config(model["config"], speakers)
    net_g = Synthesizer.from_config(saved.model, saved.spec_channels, SEGMENT_FRAMES, saved.sample_rate)
    del net_g.enc_q  # Only training uses the posterior encoder.
    net_g.load_state_dict(from_legacy_names(model["weight"]), strict=False)
    net_g.remove_weight_norm()
    net_g = net_g.to(device).float().eval()
    index = RetrievalIndex.load(str(index_path)) if index_path is not None else None
    return Voice(model["model_name"], net_g, model["sr"], index)


class VoiceConverter:
    """Holds the shared pitch tracker and content encoder, and converts audio to any loaded voice."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        embedder = cast(torch.nn.Module, load_contentvec()).to(self.device).float().eval()
        self.pipeline = Pipeline(embedder, RMVPE(self.device), self.device)
        self._voices: dict[tuple[Path, Path | None], Voice] = {}

    def voice(self, model_path: Path, index_path: Path | None = None) -> Voice:
        """Load a voice, or return it if already loaded."""
        key = (model_path.resolve(), index_path.resolve() if index_path is not None else None)
        if key not in self._voices:
            self._voices[key] = load_voice(model_path, index_path, self.device)
        return self._voices[key]

    def convert(self, audio: np.ndarray, voice: Voice, settings: ConversionSettings) -> np.ndarray:
        """Convert mono 16 kHz audio to voice. Returns audio at voice.sample_rate."""
        audio = audio.astype(np.float64)
        peak = np.abs(audio).max() / INPUT_PEAK
        if peak > 1:
            audio = audio / peak
        return self.pipeline.convert(audio, voice.net_g, voice.sample_rate, settings, voice.index)

    def convert_file(self, input_path: Path, output_path: Path, voice: Voice, settings: ConversionSettings) -> Path:
        """Convert an audio file of any common format and sample rate, and write a WAV file."""
        converted = self.convert(load_audio(input_path, SAMPLE_RATE), voice, settings)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, converted, voice.sample_rate)
        return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="Exported voice model, <name>_<epoch>e_<step>s.pth")
    parser.add_argument("--index", type=Path, help="The voice's .index file")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="WAV file to write")
    parser.add_argument("--pitch", type=float, default=0.0, help="Semitones to shift the pitch by")
    parser.add_argument("--target-pitch", type=float, help="Shift the input's median pitch to this many Hz instead")
    parser.add_argument("--index-rate", type=float, default=ConversionSettings.index_rate)
    parser.add_argument("--protect", type=float, default=ConversionSettings.protect)
    parser.add_argument("--device", help="Such as cuda:0 or cpu. Defaults to the GPU if there is one.")
    args = parser.parse_args(argv)

    settings = ConversionSettings(
        pitch_shift=args.pitch, target_pitch_hz=args.target_pitch, index_rate=args.index_rate, protect=args.protect
    )
    converter = VoiceConverter(args.device)
    voice = converter.voice(args.model, args.index)
    started = time.time()
    converter.convert_file(args.input, args.output, voice, settings)
    print(f"Wrote {args.output} in {time.time() - started:.1f} seconds.")


if __name__ == "__main__":
    main()
