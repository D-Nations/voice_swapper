from itertools import pairwise

import numpy as np


class Slicer:
    """Split audio at silences into chunks of speech.

    Args:
        sr: Sample rate of the audio.
        threshold: Frames quieter than this RMS level, in dB, count as silence.
        min_length: Shortest chunk to cut off, in milliseconds.
        min_interval: Shortest silence to cut at, in milliseconds.
        hop_size: Step between RMS frames, in milliseconds.
        max_sil_kept: Most silence to keep at either end of a chunk, in milliseconds.
    """

    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ) -> None:
        if not min_length >= min_interval >= hop_size:
            raise ValueError("min_length >= min_interval >= hop_size is required")
        if not max_sil_kept >= hop_size:
            raise ValueError("max_sil_kept >= hop_size is required")

        # From milliseconds to samples (win_size) and to RMS frames (the rest).
        min_interval_samples = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval_samples), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval_samples / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform: np.ndarray, begin: int, end: int) -> np.ndarray:
        """The samples from RMS frame begin to RMS frame end."""
        start_idx = begin * self.hop_size
        end_idx = min(waveform.shape[-1], end * self.hop_size)
        return waveform[..., start_idx:end_idx]

    def slice(self, waveform: np.ndarray) -> list[np.ndarray]:
        """Split waveform, of shape [samples] or [channels, samples], into chunks at its silences."""
        samples = waveform.mean(axis=0) if waveform.ndim > 1 else waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]

        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)

        # Each silence tag is a (start, end) range of RMS frames to cut out.
        sil_tags: list[tuple[int, int]] = []
        silence_start: int | None = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue

            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            if i - silence_start <= self.max_sil_kept:
                # Short silence: cut at its quietest frame.
                pos = int(rms_list[silence_start : i + 1].argmin()) + silence_start
                sil_tags.append((0, pos) if silence_start == 0 else (pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                # Medium silence: cut out its middle, keeping up to max_sil_kept at each side.
                pos = int(rms_list[i - self.max_sil_kept : silence_start + self.max_sil_kept + 1].argmin())
                pos += i - self.max_sil_kept
                pos_l = int(rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin()) + silence_start
                pos_r = int(rms_list[i - self.max_sil_kept : i + 1].argmin()) + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                # Long silence: cut out all but max_sil_kept at each side.
                pos_l = int(rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin()) + silence_start
                pos_r = int(rms_list[i - self.max_sil_kept : i + 1].argmin()) + i - self.max_sil_kept
                sil_tags.append((0, pos_r) if silence_start == 0 else (pos_l, pos_r))
                clip_start = pos_r
            silence_start = None

        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = int(rms_list[silence_start : silence_end + 1].argmin()) + silence_start
            sil_tags.append((pos, total_frames + 1))

        if not sil_tags:
            return [waveform]
        chunks = []
        if sil_tags[0][0] > 0:
            chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
        for (_, end), (next_start, _) in pairwise(sil_tags):
            chunks.append(self._apply_slice(waveform, end, next_start))
        if sil_tags[-1][1] < total_frames:
            chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
        return chunks


def get_rms(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """RMS level of each frame of a 1-D signal, as a [1, frames] array. Matches librosa.feature.rms."""
    padding = (frame_length // 2, frame_length // 2)
    y = np.pad(y, padding, mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length]
    power = np.mean(np.abs(frames) ** 2, axis=-1)
    return np.sqrt(power)[None, :]
