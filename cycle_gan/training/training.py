from collections.abc import Callable, Iterator
from itertools import islice

import torch
from torch import device as torch_device
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .models import Discriminator, Generator, VoiceDataset


def _repeat(loader: DataLoader) -> Iterator:
    """Iterate over a loader forever, starting a fresh (reshuffled) pass each time it runs out."""
    while True:
        yield from loader


def paired_batches(loader_a: DataLoader, loader_b: DataLoader) -> Iterator[tuple]:
    """Pair batches from two loaders for as many steps as the longer one has.

    The shorter loader restarts when it runs out, so no data from the larger
    speaker is skipped in an epoch.
    """
    if len(loader_a) == 0 or len(loader_b) == 0:
        raise ValueError("Both data loaders need at least one batch.")
    steps = max(len(loader_a), len(loader_b))
    return zip(islice(_repeat(loader_a), steps), islice(_repeat(loader_b), steps))


def discriminator_loss(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Least-squares GAN loss that pushes real predictions to 1 and fake predictions to 0.

    The generated batch is detached before it reaches the discriminator, so gradients
    flow into the discriminator for both terms but never back into the generator.
    """
    pred_real = discriminator(real)
    pred_fake = discriminator(fake.detach())
    loss_real = criterion(pred_real, torch.ones_like(pred_real))
    loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
    return (loss_real + loss_fake) * 0.5


def train_cycle_gan(
    voice_data_A: str,
    voice_data_B: str,
    num_epochs: int,
    device: torch_device,
    lambda_cycle: float = 10.0,
    lambda_identity: float = 5.0,
) -> None:
    # Create the dataset and data loader
    dataset_A = VoiceDataset(voice_data_A)
    dataset_B = VoiceDataset(voice_data_B)
    data_loader_A = DataLoader(dataset_A, batch_size=1, shuffle=True)
    data_loader_B = DataLoader(dataset_B, batch_size=1, shuffle=True)

    # Initialize the generator and discriminator models
    gen_A2B = Generator().to(device)
    gen_B2A = Generator().to(device)
    disc_A = Discriminator().to(device)
    disc_B = Discriminator().to(device)

    # Initialize the optimizers
    optimizer_gen = Adam(
        list(gen_A2B.parameters()) + list(gen_B2A.parameters()),
        lr=0.0002,
        betas=(0.5, 0.999),
    )
    optimizer_disc_A = Adam(disc_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_disc_B = Adam(disc_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Initialize the loss functions
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_adv = nn.MSELoss()

    for epoch in range(num_epochs):
        for data_A, data_B in paired_batches(data_loader_A, data_loader_B):
            real_A = data_A.to(device)
            real_B = data_B.to(device)

            # Train the generators
            optimizer_gen.zero_grad()

            # Identity loss
            same_A = gen_B2A(real_A)
            same_B = gen_A2B(real_B)
            loss_identity_A = criterion_identity(same_A, real_A) * lambda_identity
            loss_identity_B = criterion_identity(same_B, real_B) * lambda_identity

            # Adversarial loss
            fake_A = gen_B2A(real_B)
            fake_B = gen_A2B(real_A)
            pred_fake_A = disc_A(fake_A)
            pred_fake_B = disc_B(fake_B)
            loss_adv_A = criterion_adv(pred_fake_A, torch.ones_like(pred_fake_A))
            loss_adv_B = criterion_adv(pred_fake_B, torch.ones_like(pred_fake_B))

            # Cycle-consistency loss
            reconstructed_A = gen_B2A(fake_B)
            reconstructed_B = gen_A2B(fake_A)
            loss_cycle_A = criterion_cycle(reconstructed_A, real_A) * lambda_cycle
            loss_cycle_B = criterion_cycle(reconstructed_B, real_B) * lambda_cycle

            # Total generator loss
            loss_gen = loss_identity_A + loss_identity_B + loss_adv_A + loss_adv_B + loss_cycle_A + loss_cycle_B
            loss_gen.backward()
            optimizer_gen.step()

            # Train the discriminators
            optimizer_disc_A.zero_grad()
            loss_disc_A = discriminator_loss(disc_A, real_A, fake_A, criterion_adv)
            loss_disc_A.backward()
            optimizer_disc_A.step()

            optimizer_disc_B.zero_grad()
            loss_disc_B = discriminator_loss(disc_B, real_B, fake_B, criterion_adv)
            loss_disc_B.backward()
            optimizer_disc_B.step()
