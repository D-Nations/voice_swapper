from collections.abc import Iterable

import torch
from torch.nn.utils import parametrize


def init_weights(module: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Draw the weights of convolution layers from a normal distribution. Pass to Module.apply."""
    weight = getattr(module, "weight", None)
    if "Conv" in type(module).__name__ and isinstance(weight, torch.Tensor):
        weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Padding that keeps the length of a stride-1 convolution unchanged."""
    return (kernel_size * dilation - dilation) // 2


def convert_pad_shape(pad_shape: list[list[int]]) -> list[int]:
    """Turn per-dimension [before, after] pairs, first dimension first, into F.pad's flat last-first order."""
    return [item for pair in reversed(pad_shape) for item in pair]


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4, dim: int = 2) -> torch.Tensor:
    """Cut a segment_size slice from each batch item, starting at that item's index in ids_str.

    dim is the number of dimensions of x: 2 for [batch, time] and 3 for [batch, channels, time].
    """
    if dim == 2:
        ret = torch.zeros_like(x[:, :segment_size])
    elif dim == 3:
        ret = torch.zeros_like(x[:, :, :segment_size])
    else:
        raise ValueError(f"slice_segments supports 2 or 3 dimensions, not {dim}.")

    for i in range(x.size(0)):
        idx_str = int(ids_str[i].item())
        idx_end = idx_str + segment_size
        if dim == 2:
            ret[i] = x[i, idx_str:idx_end]
        else:
            ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor, x_lengths: torch.Tensor | int | None = None, segment_size: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut a random segment_size slice from each item of a [batch, channels, time] tensor."""
    b, _, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    return slice_segments(x, ids_str, segment_size, dim=3), ids_str


def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor | float, n_channels: int
) -> torch.Tensor:
    """WaveNet's gated activation: tanh of the first n_channels times sigmoid of the rest."""
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    return t_act * s_act


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """Boolean [batch, max_length] mask that is True inside each sequence."""
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def grad_norm(parameters: Iterable[torch.Tensor], norm_type: float = 2.0) -> float:
    """Total norm of the parameters' gradients, for logging."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return torch.linalg.vector_norm(torch.stack([g.norm(norm_type) for g in grads]), ord=norm_type).item()


def remove_weight_norm(module: torch.nn.Module) -> None:
    """Fold weight normalization into plain weights in module and all its children, for faster inference."""
    for child in module.modules():
        if parametrize.is_parametrized(child, "weight"):
            parametrize.remove_parametrizations(child, "weight")
