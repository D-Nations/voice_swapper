import torch
from torch import nn

from cycle_gan.training.models import Discriminator, Generator
from cycle_gan.training.training import discriminator_loss


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
