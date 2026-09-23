from collections.abc import Iterable, Sequence

import torch


def _sum(values: Iterable[torch.Tensor]) -> torch.Tensor:
    """Add tensors in order, keeping their dtype, unlike torch.stack(...).sum()."""
    iterator = iter(values)
    total = next(iterator)
    for value in iterator:
        total = total + value
    return total


def feature_loss(fmap_r: Sequence[Sequence[torch.Tensor]], fmap_g: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
    """Feature matching loss: L1 distance between the discriminators' feature maps of real and generated audio."""
    return 2 * _sum(
        torch.mean(torch.abs(rl - gl))
        for dr, dg in zip(fmap_r, fmap_g, strict=True)
        for rl, gl in zip(dr, dg, strict=True)
    )


def discriminator_loss(
    disc_real_outputs: Sequence[torch.Tensor], disc_generated_outputs: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Least-squares GAN loss for the discriminators: real audio should score 1, generated audio 0."""
    return _sum(
        torch.mean((1 - dr.float()) ** 2) + torch.mean(dg.float() ** 2)
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs, strict=True)
    )


def generator_loss(disc_outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Least-squares GAN loss for the generator: generated audio should score 1."""
    return _sum(torch.mean((1 - dg.float()) ** 2) for dg in disc_outputs)


def kl_loss(
    z_p: torch.Tensor, logs_q: torch.Tensor, m_p: torch.Tensor, logs_p: torch.Tensor, z_mask: torch.Tensor
) -> torch.Tensor:
    """KL divergence between the posterior, sampled as z_p, and the prior N(m_p, exp(logs_p)), averaged over frames.

    All inputs are [batch, channels, frames].
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    return (kl * z_mask).sum() / z_mask.sum()
