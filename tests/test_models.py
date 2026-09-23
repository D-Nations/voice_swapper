from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cycle_gan.training.models import Discriminator, Generator, VoiceDataset


def test_dataset_items_have_a_channel_dimension(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.random.rand(128, 64).astype(np.float32))

    item = VoiceDataset(str(tmp_path), segment_frames=64)[0]

    assert tuple(item.shape) == (1, 128, 64)


def test_models_accept_dataloader_batches(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.random.rand(128, 64).astype(np.float32))
    batch = next(iter(DataLoader(VoiceDataset(str(tmp_path), segment_frames=64), batch_size=1)))

    generated = Generator()(batch)
    prediction = Discriminator()(generated)

    assert generated.shape == batch.shape
    assert prediction.shape[:2] == (1, 1)


def test_dataset_splits_files_into_fixed_length_segments(tmp_path: Path) -> None:
    # 100 frames gives 3 segments of 32 plus a dropped 4-frame tail. 20 frames is too short to use.
    long = np.arange(128 * 100, dtype=np.float32).reshape(128, 100)
    np.save(tmp_path / "long.npy", long)
    np.save(tmp_path / "short.npy", np.zeros((128, 20), dtype=np.float32))

    dataset = VoiceDataset(str(tmp_path), segment_frames=32)

    assert len(dataset) == 3
    assert all(tuple(dataset[i].shape) == (1, 128, 32) for i in range(len(dataset)))
    assert np.array_equal(dataset[1].squeeze(0).numpy(), long[:, 32:64])


def test_generator_has_no_batch_norm() -> None:
    assert not any(isinstance(module, nn.BatchNorm2d) for module in Generator().modules())


def test_generator_output_for_a_sample_does_not_depend_on_the_rest_of_the_batch() -> None:
    torch.manual_seed(0)
    generator = Generator(num_res_blocks=2).train()
    sample = torch.randn(1, 1, 32, 32)
    other = torch.randn(1, 1, 32, 32) * 10

    alone = generator(sample)
    batched = generator(torch.cat([sample, other]))[:1]

    assert torch.allclose(alone, batched, atol=1e-5)
