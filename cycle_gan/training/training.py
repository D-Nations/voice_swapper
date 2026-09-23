import argparse
from collections.abc import Callable, Iterator
from itertools import islice
from pathlib import Path
from typing import NamedTuple

import torch
from torch import device as torch_device
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from cycle_gan.training.models import Discriminator, Generator, VoiceDataset


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


class CycleGAN(NamedTuple):
    gen_A2B: Generator
    gen_B2A: Generator
    disc_A: Discriminator
    disc_B: Discriminator


def train_cycle_gan(
    voice_data_A: str,
    voice_data_B: str,
    num_epochs: int,
    device: torch_device,
    checkpoint_dir: str | Path,
    lambda_cycle: float = 10.0,
    lambda_identity: float = 5.0,
    batch_size: int = 1,
    segment_frames: int = 128,
) -> CycleGAN:
    """Train a CycleGAN between two speakers and save a checkpoint after every epoch.

    Checkpoints are written to checkpoint_dir as epoch_001.pt, epoch_002.pt, and so on.
    Each one holds the state dicts of all four models and all three optimizers.
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Create the dataset and data loader
    dataset_A = VoiceDataset(voice_data_A, segment_frames=segment_frames)
    dataset_B = VoiceDataset(voice_data_B, segment_frames=segment_frames)
    data_loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    data_loader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)

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

    for epoch in range(1, num_epochs + 1):
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        num_steps = 0

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

            total_gen_loss += loss_gen.item()
            total_disc_loss += loss_disc_A.item() + loss_disc_B.item()
            num_steps += 1

        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"generator loss {total_gen_loss / num_steps:.4f}, "
            f"discriminator loss {total_disc_loss / num_steps:.4f}"
        )
        torch.save(
            {
                "epoch": epoch,
                "gen_A2B": gen_A2B.state_dict(),
                "gen_B2A": gen_B2A.state_dict(),
                "disc_A": disc_A.state_dict(),
                "disc_B": disc_B.state_dict(),
                "optimizer_gen": optimizer_gen.state_dict(),
                "optimizer_disc_A": optimizer_disc_A.state_dict(),
                "optimizer_disc_B": optimizer_disc_B.state_dict(),
            },
            checkpoint_path / f"epoch_{epoch:03d}.pt",
        )

    return CycleGAN(gen_A2B, gen_B2A, disc_A, disc_B)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CycleGAN voice converter between two speakers.")
    parser.add_argument("voice_data_A", help="Folder of preprocessed .npy spectrograms for speaker A")
    parser.add_argument("voice_data_B", help="Folder of preprocessed .npy spectrograms for speaker B")
    parser.add_argument("checkpoint_dir", help="Folder to write epoch checkpoints to")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--segment-frames", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_cycle_gan(
        args.voice_data_A,
        args.voice_data_B,
        num_epochs=args.epochs,
        device=torch_device(args.device),
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        segment_frames=args.segment_frames,
    )
