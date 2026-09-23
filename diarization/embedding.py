"""Speaker embeddings for short overlapping windows of audio, from a pretrained ECAPA-TDNN model."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "ecapa"
EMBEDDING_SAMPLE_RATE = 16000  # The model was trained on 16 kHz audio.

# Each window is embedded on its own. WINDOW_SECONDS must be a whole multiple of HOP_SECONDS.
WINDOW_SECONDS = 1.5
HOP_SECONDS = 0.75
BATCH_SIZE = 256  # Windows per GPU batch. Lower it if the GPU runs out of memory.

# A window whose RMS level is below this, in dB relative to full scale, counts as silence.
SILENCE_DB = -45.0

# Maps a (windows, samples) batch of 16 kHz audio to (windows, dims) unit-length embeddings.
Embedder = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class WindowEmbeddings:
    """Embeddings for the windows of one audio file. Window i starts at i * HOP_SECONDS."""

    embeddings: np.ndarray  # (windows, dims), unit length
    loudness_db: np.ndarray  # (windows,)


class SpeakerEmbedder:
    """Wraps the pretrained SpeechBrain ECAPA-TDNN speaker model. Downloads it on first use."""

    def __init__(self, device: str | None = None) -> None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = EncoderClassifier.from_hparams(
            source=MODEL_SOURCE,
            savedir=str(MODEL_DIR),
            # Copy rather than symlink, since Windows needs extra privileges for symlinks.
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": self.device},
        )
        if model is None:
            raise RuntimeError(f"Could not load the speaker model {MODEL_SOURCE}.")
        model.eval()
        self.model = model

    @torch.inference_mode()
    def __call__(self, windows: torch.Tensor) -> torch.Tensor:
        batches = []
        for batch in windows.split(BATCH_SIZE):
            embeddings = self.model.encode_batch(batch.to(self.device)).squeeze(1)
            batches.append(torch.nn.functional.normalize(embeddings, dim=1).cpu())
        return torch.cat(batches)


def load_mono(path: str | Path, sample_rate: int | None = None) -> tuple[torch.Tensor, int]:
    """Load an audio file as a 1D mono waveform, optionally resampled. Returns it with its sample rate."""
    data, file_sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.mean(axis=1))
    if sample_rate is not None and sample_rate != file_sample_rate:
        return resample(waveform, file_sample_rate, sample_rate), sample_rate
    return waveform, file_sample_rate


def frame_windows(
    waveform: torch.Tensor,
    sample_rate: int,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
) -> torch.Tensor:
    """Cut a 1D waveform into overlapping (windows, samples) frames. A partial final window is dropped."""
    window = round(window_seconds * sample_rate)
    hop = round(hop_seconds * sample_rate)
    if waveform.shape[0] < window:
        return waveform.new_zeros((0, window))
    return waveform.unfold(0, window, hop)


def loudness_db(windows: torch.Tensor) -> torch.Tensor:
    """RMS level of each window in dB relative to full scale."""
    rms = windows.pow(2).mean(dim=1).sqrt()
    return 20 * torch.log10(rms.clamp(min=1e-10))


def embed_waveform(waveform: torch.Tensor, embedder: Embedder) -> WindowEmbeddings:
    """Embed every window of a 16 kHz mono waveform."""
    windows = frame_windows(waveform, EMBEDDING_SAMPLE_RATE)
    if windows.shape[0] == 0:
        return WindowEmbeddings(np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.float32))
    return WindowEmbeddings(
        embeddings=embedder(windows).numpy(),
        loudness_db=loudness_db(windows).numpy(),
    )


def embed_file(path: str | Path, embedder: Embedder) -> WindowEmbeddings:
    """Embed every window of an audio file in any format soundfile reads, including WAV and MP3."""
    waveform, _ = load_mono(path, EMBEDDING_SAMPLE_RATE)
    return embed_waveform(waveform, embedder)
