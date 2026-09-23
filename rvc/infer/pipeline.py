"""The signal path of a conversion: pitch, content features, index blending, and synthesis."""

from dataclasses import dataclass

import faiss
import librosa
import numpy as np
import torch
from scipy import signal
from torch.nn import functional as F

from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.predictors.rmvpe import RMVPE
from rvc.train.extract.extract import coarse_f0

SAMPLE_RATE = 16000  # Pitch and content features work at 16 kHz.
WINDOW = 160  # Samples per pitch frame, 10 ms.

# Long inputs are converted in pieces cut at quiet points, with padding either side so the pieces join
# cleanly. All in seconds, as in Applio's settings for GPUs with 6 GB or more.
PAD_SECONDS = 1
QUERY_SECONDS = 6  # Search this far either side of each cut for the quietest point.
CENTER_SECONDS = 38  # Aim for pieces this long.
MAX_SECONDS = 41  # Inputs up to this long are converted in one piece.

INDEX_NEIGHBOURS = 8


def _high_pass_filter() -> tuple[np.ndarray, np.ndarray]:
    """Coefficients of the filter that removes rumble below 48 Hz before conversion."""
    coefficients = signal.butter(N=5, Wn=48, btype="high", fs=SAMPLE_RATE, output="ba")
    # SciPy's stubs allow other shapes than the (numerator, denominator) pair "ba" returns.
    if not isinstance(coefficients, tuple) or len(coefficients) != 2:
        raise TypeError(f"Expected filter coefficients (b, a), got {coefficients!r}.")
    numerator, denominator = coefficients
    return np.asarray(numerator), np.asarray(denominator)


_HIGH_PASS_B, _HIGH_PASS_A = _high_pass_filter()


@dataclass(frozen=True)
class ConversionSettings:
    """How to convert.

    pitch_shift: Semitones to shift the input's pitch by. Ignored when target_pitch_hz is set.
    target_pitch_hz: Shift the input so its median pitch lands here, whoever is speaking.
    index_rate: How far to pull the content features toward their nearest neighbours in the target
        voice's training data, from 0 (not at all) to 1. Higher sounds more like the target but can
        slur words.
    protect: Below 0.5, keep this share of the original features in unvoiced frames, which protects
        consonants and breaths from artifacts. 0.5 turns protection off.
    volume_envelope: 1 keeps the converted audio's own loudness contour. Lower values move it toward
        the input's.
    """

    pitch_shift: float = 0.0
    target_pitch_hz: float | None = None
    index_rate: float = 0.75
    protect: float = 0.33
    volume_envelope: float = 1.0


@dataclass(frozen=True)
class RetrievalIndex:
    """A voice's faiss index and the training features it holds."""

    index: faiss.Index
    features: np.ndarray  # [frames, 768]

    @classmethod
    def load(cls, path: str) -> RetrievalIndex:
        index = faiss.read_index(path)
        return cls(index, index.reconstruct_n(0, index.ntotal))


def median_pitch(f0: np.ndarray) -> float | None:
    """Median pitch in Hz of the voiced frames, or None if too few frames are voiced."""
    voiced = f0[f0 > 0]
    return float(np.median(voiced)) if voiced.size >= 2 else None


def semitones_between(from_hz: float, to_hz: float) -> float:
    return 12 * float(np.log2(to_hz / from_hz))


def match_loudness(
    source: np.ndarray, source_rate: int, target: np.ndarray, target_rate: int, rate: float
) -> np.ndarray:
    """Blend target's loudness contour toward source's. rate 1 leaves target unchanged, 0 copies source's."""

    def contour(audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        rms = librosa.feature.rms(y=audio, frame_length=sample_rate // 2 * 2, hop_length=sample_rate // 2)
        return F.interpolate(torch.from_numpy(rms).float().unsqueeze(0), size=target.shape[0], mode="linear").squeeze()

    source_rms = contour(source, source_rate)
    target_rms = torch.clamp(contour(target, target_rate), min=1e-6)
    return target * (torch.pow(source_rms, 1 - rate) * torch.pow(target_rms, rate - 1)).numpy()


class Pipeline:
    """Converts 16 kHz audio to a voice's sample rate with its model and, optionally, its index."""

    def __init__(
        self,
        embedder: torch.nn.Module,
        pitch_tracker: RMVPE,
        device: torch.device,
        rmvpe_threshold: float = 0.03,
    ) -> None:
        self.embedder = embedder
        self.pitch_tracker = pitch_tracker
        self.device = device
        self.rmvpe_threshold = rmvpe_threshold

    def convert(
        self,
        audio: np.ndarray,
        net_g: Synthesizer,
        target_rate: int,
        settings: ConversionSettings,
        index: RetrievalIndex | None = None,
        speaker_id: int = 0,
    ) -> np.ndarray:
        """Convert mono 16 kHz audio and return it at target_rate, peak-limited to 0.99."""
        audio = signal.filtfilt(_HIGH_PASS_B, _HIGH_PASS_A, audio)
        cuts = self._cut_points(audio)
        pad = SAMPLE_RATE * PAD_SECONDS
        target_pad = target_rate * PAD_SECONDS
        audio_pad = np.pad(audio, (pad, pad), mode="reflect")
        n_frames = audio_pad.shape[0] // WINDOW

        f0 = self.pitch_tracker.get_f0(audio_pad, threshold=self.rmvpe_threshold)
        f0 = f0 * 2 ** (self._shift(f0, settings) / 12)
        pitch = torch.tensor(coarse_f0(f0)[:n_frames], device=self.device).unsqueeze(0).long()
        pitchf = torch.tensor(f0[:n_frames], device=self.device).unsqueeze(0).float()
        sid = torch.tensor([speaker_id], device=self.device).long()
        use_index = index if settings.index_rate > 0 else None

        pieces = []
        start = 0
        for cut in cuts:
            cut = cut // WINDOW * WINDOW
            end = cut + 2 * pad
            converted = self._convert_piece(
                audio_pad[start : end + WINDOW],
                pitch[:, start // WINDOW : end // WINDOW],
                pitchf[:, start // WINDOW : end // WINDOW],
                net_g,
                sid,
                settings,
                use_index,
            )
            pieces.append(converted[target_pad:-target_pad])
            start = cut
        converted = self._convert_piece(
            audio_pad[start:],
            pitch[:, start // WINDOW :],
            pitchf[:, start // WINDOW :],
            net_g,
            sid,
            settings,
            use_index,
        )
        pieces.append(converted[target_pad:-target_pad])

        output = np.concatenate(pieces)
        if settings.volume_envelope != 1:
            output = match_loudness(audio, SAMPLE_RATE, output, target_rate, settings.volume_envelope)
        peak = np.abs(output).max() / 0.99
        if peak > 1:
            output /= peak
        return output

    @staticmethod
    def _shift(f0: np.ndarray, settings: ConversionSettings) -> float:
        if settings.target_pitch_hz is None:
            return settings.pitch_shift
        source = median_pitch(f0)
        return semitones_between(source, settings.target_pitch_hz) if source is not None else 0.0

    @staticmethod
    def _cut_points(audio: np.ndarray) -> list[int]:
        """Where to split audio longer than MAX_SECONDS: the quietest point near each multiple of CENTER_SECONDS."""
        if audio.shape[0] + WINDOW <= SAMPLE_RATE * MAX_SECONDS:
            return []
        padded = np.pad(audio, (WINDOW // 2, WINDOW // 2), mode="reflect")
        # Sum over a sliding window, so a cut lands in a quiet stretch, not just a zero crossing.
        level = np.zeros_like(audio)
        for i in range(WINDOW):
            level += padded[i : i - WINDOW]
        query = SAMPLE_RATE * QUERY_SECONDS
        return [
            t - query + int(np.argmin(np.abs(level[t - query : t + query])))
            for t in range(SAMPLE_RATE * CENTER_SECONDS, audio.shape[0], SAMPLE_RATE * CENTER_SECONDS)
        ]

    def _convert_piece(
        self,
        audio: np.ndarray,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        net_g: Synthesizer,
        sid: torch.Tensor,
        settings: ConversionSettings,
        index: RetrievalIndex | None,
    ) -> np.ndarray:
        with torch.inference_mode():
            waveform = torch.from_numpy(audio).float().view(1, -1).to(self.device)
            feats = self.embedder(waveform)["last_hidden_state"]
            original = feats.clone()
            if index is not None:
                feats = self._blend_with_index(feats, index, settings.index_rate)

            # Content features come every 20 ms and pitch every 10 ms, so repeat each feature frame.
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            original = F.interpolate(original.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            n_frames = min(audio.shape[0] // WINDOW, feats.shape[1])
            pitch, pitchf = pitch[:, :n_frames], pitchf[:, :n_frames]

            if settings.protect < 0.5:
                # 1 in voiced frames, protect in unvoiced ones.
                weight = torch.where(pitchf > 0, 1.0, settings.protect).unsqueeze(-1)
                feats = (feats * weight + original * (1 - weight)).to(original.dtype)

            lengths = torch.tensor([n_frames], device=self.device).long()
            output, _, _ = net_g.infer(feats.float(), lengths, pitch, pitchf, sid)
        return output[0, 0].float().cpu().numpy()

    def _blend_with_index(self, feats: torch.Tensor, index: RetrievalIndex, rate: float) -> torch.Tensor:
        """Mix each frame with the inverse-square-distance weighted mean of its nearest training frames."""
        query = feats[0].cpu().numpy()
        distances, neighbours = index.index.search(query, k=INDEX_NEIGHBOURS)
        weight = np.square(1 / distances)
        weight /= weight.sum(axis=1, keepdims=True)
        retrieved = np.sum(index.features[neighbours] * np.expand_dims(weight, axis=2), axis=1)
        return torch.from_numpy(retrieved).unsqueeze(0).to(self.device) * rate + (1 - rate) * feats
