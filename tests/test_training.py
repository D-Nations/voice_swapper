import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cycle_gan.training.models import Discriminator, Generator
from cycle_gan.training.training import discriminator_loss, paired_batches


def test_discriminator_loss_trains_the_discriminator_on_fakes() -> None:
    torch.manual_seed(0)
    discriminator = Discriminator()
    generator = Generator()
    real = torch.randn(1, 1, 64, 64)
    fake = generator(torch.randn(1, 1, 64, 64))

    loss = discriminator_loss(discriminator, real, fake, nn.MSELoss())
    loss.backward()

    # The discriminator must receive gradients, including from the fake term.
    assert all(p.grad is not None for p in discriminator.parameters())
    # The generator must not be updated by the discriminator's loss.
    assert all(p.grad is None for p in generator.parameters())


def test_discriminator_loss_fake_term_has_gradient() -> None:
    torch.manual_seed(0)
    discriminator = Discriminator()
    real = torch.randn(1, 1, 64, 64)
    fake = torch.randn(1, 1, 64, 64)
    criterion = nn.MSELoss()

    # Zero out the real term so any gradient must come from the fake term.
    def fake_term_only(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.eq(1).all():
            return (pred * 0).sum()
        return criterion(pred, target)

    loss = discriminator_loss(discriminator, real, fake, fake_term_only)
    loss.backward()

    first_layer = next(discriminator.parameters())
    assert first_layer.grad is not None
    assert first_layer.grad.abs().sum() > 0


def test_paired_batches_covers_every_batch_of_the_longer_loader() -> None:
    loader_a = DataLoader(TensorDataset(torch.arange(5)), batch_size=1)
    loader_b = DataLoader(TensorDataset(torch.arange(2)), batch_size=1)

    pairs = [(int(a[0]), int(b[0])) for a, b in paired_batches(loader_a, loader_b)]

    assert [a for a, _ in pairs] == [0, 1, 2, 3, 4]
    assert [b for _, b in pairs] == [0, 1, 0, 1, 0]


def test_paired_batches_rejects_an_empty_loader() -> None:
    with pytest.raises(ValueError):
        paired_batches(DataLoader(TensorDataset(torch.arange(0))), DataLoader(TensorDataset(torch.arange(1))))
