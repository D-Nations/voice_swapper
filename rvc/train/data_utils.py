import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.utils.data

from rvc.configs.config import DataConfig
from rvc.train.mel_processing import spectrogram_torch
from rvc.train.utils import load_filelist, load_wav_to_torch

MAX_FRAMES = 900  # Longest slice used, in pitch frames (10 ms each at 40 kHz).

type Sample = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class TrainingBatch(NamedTuple):
    phone: torch.Tensor  # Content features, [batch, frames, 768]
    phone_lengths: torch.Tensor
    pitch: torch.Tensor  # Coarse pitch bins, [batch, frames]
    pitchf: torch.Tensor  # Pitch in Hz, [batch, frames]
    spec: torch.Tensor  # Linear spectrogram, [batch, bins, frames]
    spec_lengths: torch.Tensor
    wave: torch.Tensor  # Audio, [batch, 1, samples]
    wave_lengths: torch.Tensor
    sid: torch.Tensor  # Speaker ids, [batch]

    def to(self, device: torch.device | str, non_blocking: bool = False) -> TrainingBatch:
        return TrainingBatch(*(tensor.to(device, non_blocking=non_blocking) for tensor in self))


class TextAudioLoaderMultiNSFsid(torch.utils.data.Dataset[Sample]):
    """Training slices from filelist.txt, each as (spec, wave, phone, pitch, pitchf, speaker id).

    Content features come at half the spectrogram's frame rate, so they are repeated to match,
    and every item is trimmed to its shortest stream.
    """

    def __init__(self, filelist: Path, data: DataConfig) -> None:
        self.entries = load_filelist(filelist)
        self.sample_rate = data.sample_rate
        self.filter_length = data.filter_length
        self.hop_length = data.hop_length
        self.win_length = data.win_length
        # Approximate lengths for bucketing, from the file size: 4-byte samples, so size // (3 * hop) is a
        # slight overestimate of the frame count.
        self.lengths = [os.path.getsize(entry[0]) // (3 * self.hop_length) for entry in self.entries]

    def __getitem__(self, index: int) -> Sample:
        wav_path, phone_path, pitch_path, pitchf_path, speaker = self.entries[index]
        phone, pitch, pitchf = self.get_labels(phone_path, pitch_path, pitchf_path)
        spec, wav = self.get_audio(wav_path)
        sid = torch.LongTensor([int(speaker)])

        len_min = min(phone.size(0), spec.size(-1))
        spec = spec[:, :len_min]
        wav = wav[:, : len_min * self.hop_length]
        phone = phone[:len_min, :]
        pitch = pitch[:len_min]
        pitchf = pitchf[:len_min]
        return spec, wav, phone, pitch, pitchf, sid

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def get_labels(
        phone_path: str, pitch_path: str, pitchf_path: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phone = np.repeat(np.load(phone_path), 2, axis=0)
        pitch = np.load(pitch_path)
        pitchf = np.load(pitchf_path)
        n_num = min(phone.shape[0], MAX_FRAMES)
        return (
            torch.FloatTensor(phone[:n_num, :]),
            torch.LongTensor(pitch[:n_num]),
            torch.FloatTensor(pitchf[:n_num]),
        )

    def get_audio(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        audio, sample_rate = load_wav_to_torch(path)
        if sample_rate != self.sample_rate:
            raise ValueError(f"{path} is at {sample_rate} Hz, not the model's {self.sample_rate} Hz.")
        audio = audio.unsqueeze(0)
        spec = spectrogram_torch(audio, self.filter_length, self.hop_length, self.win_length, center=False)
        return spec.squeeze(0), audio


def collate(batch: Sequence[Sample]) -> TrainingBatch:
    """Zero-pad a batch to its longest item, ordered from longest to shortest spectrogram."""
    _, order = torch.sort(torch.LongTensor([item[0].size(1) for item in batch]), dim=0, descending=True)
    size = len(batch)
    max_spec_len = max(item[0].size(1) for item in batch)
    max_wave_len = max(item[1].size(1) for item in batch)
    max_phone_len = max(item[2].size(0) for item in batch)

    spec_padded = torch.zeros(size, batch[0][0].size(0), max_spec_len)
    wave_padded = torch.zeros(size, 1, max_wave_len)
    phone_padded = torch.zeros(size, max_phone_len, batch[0][2].shape[1])
    pitch_padded = torch.zeros(size, max_phone_len, dtype=torch.long)
    pitchf_padded = torch.zeros(size, max_phone_len)
    spec_lengths = torch.zeros(size, dtype=torch.long)
    wave_lengths = torch.zeros(size, dtype=torch.long)
    phone_lengths = torch.zeros(size, dtype=torch.long)
    sid = torch.zeros(size, dtype=torch.long)

    for i, index in enumerate(order.tolist()):
        spec, wave, phone, pitch, pitchf, speaker = batch[index]
        spec_padded[i, :, : spec.size(1)] = spec
        spec_lengths[i] = spec.size(1)
        wave_padded[i, :, : wave.size(1)] = wave
        wave_lengths[i] = wave.size(1)
        phone_padded[i, : phone.size(0), :] = phone
        phone_lengths[i] = phone.size(0)
        pitch_padded[i, : pitch.size(0)] = pitch
        pitchf_padded[i, : pitchf.size(0)] = pitchf
        sid[i] = speaker

    return TrainingBatch(
        phone_padded,
        phone_lengths,
        pitch_padded,
        pitchf_padded,
        spec_padded,
        spec_lengths,
        wave_padded,
        wave_lengths,
        sid,
    )


class BucketBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Batches of similar-length items, so little of each batch is padding.

    Items are grouped into buckets by length between consecutive boundaries, and items outside
    all buckets are dropped. Each bucket is padded with repeats to a whole number of batches.
    Call set_epoch before each epoch so the shuffle changes between epochs but not between runs.

    Args:
        lengths: Length of each item.
        batch_size: Items per batch.
        boundaries: Increasing bucket edges. Bucket i holds lengths in (boundaries[i], boundaries[i + 1]].
        shuffle: Shuffle within buckets and the order of batches.
    """

    def __init__(
        self, lengths: Sequence[int], batch_size: int, boundaries: Sequence[int], shuffle: bool = True
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0
        self.buckets = self._create_buckets(lengths, list(boundaries))
        self.num_samples_per_bucket = [
            len(bucket) + (batch_size - len(bucket) % batch_size) % batch_size for bucket in self.buckets
        ]
        self.num_samples = sum(self.num_samples_per_bucket)

    @staticmethod
    def _create_buckets(lengths: Sequence[int], boundaries: list[int]) -> list[list[int]]:
        buckets: list[list[int]] = [[] for _ in range(len(boundaries) - 1)]
        for index, length in enumerate(lengths):
            for b in range(len(boundaries) - 1):
                if boundaries[b] < length <= boundaries[b + 1]:
                    buckets[b].append(index)
                    break
        return [bucket for bucket in buckets if bucket]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        batches = []
        for bucket, num_samples in zip(self.buckets, self.num_samples_per_bucket, strict=True):
            ids = torch.randperm(len(bucket), generator=g).tolist() if self.shuffle else list(range(len(bucket)))
            rem = num_samples - len(bucket)
            ids = ids + ids * (rem // len(bucket)) + ids[: rem % len(bucket)]
            for j in range(len(ids) // self.batch_size):
                batches.append([bucket[i] for i in ids[j * self.batch_size : (j + 1) * self.batch_size]])

        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]
        return iter(batches)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
