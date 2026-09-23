from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from cycle_gan.training.models import Discriminator, Generator, VoiceDataset


def test_dataset_items_have_a_channel_dimension(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.random.rand(128, 64).astype(np.float32))

    item = VoiceDataset(str(tmp_path))[0]

    assert tuple(item.shape) == (1, 128, 64)


def test_models_accept_dataloader_batches(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.random.rand(128, 64).astype(np.float32))
    batch = next(iter(DataLoader(VoiceDataset(str(tmp_path)), batch_size=1)))

    generated = Generator()(batch)
    prediction = Discriminator()(generated)

    assert generated.shape == batch.shape
    assert prediction.shape[:2] == (1, 1)
