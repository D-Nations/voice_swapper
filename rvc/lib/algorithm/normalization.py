import torch


class LayerNorm(torch.nn.Module):
    """Layer normalization over the channel dimension of a [batch, channels, time] tensor."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(x, (x.size(-1),), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
