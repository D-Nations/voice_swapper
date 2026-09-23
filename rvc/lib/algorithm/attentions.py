import math
from collections.abc import Callable

import torch
from torch.nn import functional as F

from rvc.lib.algorithm.commons import convert_pad_shape


class MultiHeadAttention(torch.nn.Module):
    """Multi-head attention with optional windowed relative position encoding.

    Args:
        channels: Input channels.
        out_channels: Output channels.
        n_heads: Attention heads. Must divide channels.
        p_dropout: Dropout on the attention weights.
        window_size: Window for relative position encoding, or None for none.
        heads_share: Whether the heads share relative position embeddings.
        block_length: Limit attention to this many positions either side, or None for no limit.
        proximal_bias: Bias self-attention toward nearby positions.
        proximal_init: Start the key projection equal to the query projection.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int | None = None,
        heads_share: bool = True,
        block_length: int | None = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ) -> None:
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError("Channels must be divisible by the number of heads.")

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.block_length = block_length
        self.proximal_bias = proximal_bias

        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        if window_size:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = torch.nn.Parameter(
                torch.randn(n_heads_rel, 2 * window_size + 1, self.k_channels) * rel_stddev
            )
            self.emb_rel_v = torch.nn.Parameter(
                torch.randn(n_heads_rel, 2 * window_size + 1, self.k_channels) * rel_stddev
            )

        for conv in (self.conv_q, self.conv_k, self.conv_v, self.conv_o):
            torch.nn.init.xavier_uniform_(conv.weight)

        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                if self.conv_k.bias is not None and self.conv_q.bias is not None:
                    self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    def attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, d, t_s = key.size()
        t_t = query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))

        if self.window_size:
            if t_s != t_t:
                raise ValueError("Relative attention only supports self-attention.")
            scores += self._compute_relative_scores(query, t_s)

        if self.proximal_bias:
            if t_s != t_t:
                raise ValueError("Proximal bias only supports self-attention.")
            scores += self._attention_bias_proximal(t_s).to(scores.device, scores.dtype)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length:
                block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                scores = scores.masked_fill(block_mask == 0, -1e4)

        p_attn = self.drop(F.softmax(scores, dim=-1))
        output = torch.matmul(p_attn, value)

        if self.window_size:
            output += self._apply_relative_values(p_attn, t_s)

        return output.transpose(2, 3).contiguous().view(b, d, t_t), p_attn

    def _compute_relative_scores(self, query: torch.Tensor, length: int) -> torch.Tensor:
        rel_emb = self._get_relative_embeddings(self.emb_rel_k, length)
        rel_logits = torch.matmul(query / math.sqrt(self.k_channels), rel_emb.unsqueeze(0).transpose(-2, -1))
        return self._relative_position_to_absolute_position(rel_logits)

    def _apply_relative_values(self, p_attn: torch.Tensor, length: int) -> torch.Tensor:
        rel_weights = self._absolute_position_to_relative_position(p_attn)
        rel_emb = self._get_relative_embeddings(self.emb_rel_v, length)
        return torch.matmul(rel_weights, rel_emb.unsqueeze(0))

    def _get_relative_embeddings(self, embeddings: torch.Tensor, length: int) -> torch.Tensor:
        if self.window_size is None:
            raise ValueError("Relative position embeddings need a window_size.")
        pad_length = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        end = start + 2 * length - 1
        if pad_length > 0:
            embeddings = F.pad(embeddings, convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        return embeddings[:, start:end]

    @staticmethod
    def _relative_position_to_absolute_position(x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view(batch, heads, length * 2 * length)
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        return x_flat.view(batch, heads, length + 1, 2 * length - 1)[:, :, :length, length - 1 :]

    @staticmethod
    def _absolute_position_to_relative_position(x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view(batch, heads, length**2 + length * (length - 1))
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        return x_flat.view(batch, heads, length, 2 * length)[:, :, :, 1:]

    @staticmethod
    def _attention_bias_proximal(length: int) -> torch.Tensor:
        r = torch.arange(length, dtype=torch.float32)
        diff = r.unsqueeze(0) - r.unsqueeze(1)
        return -torch.log1p(torch.abs(diff)).unsqueeze(0).unsqueeze(0)


class FFN(torch.nn.Module):
    """Two-layer convolutional feed-forward network.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        filter_channels: Channels between the two layers.
        kernel_size: Kernel size of both convolutions.
        p_dropout: Dropout between the layers.
        activation: "gelu" for a GELU approximation, otherwise ReLU.
        causal: Pad on the left only, so no output depends on later inputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: str | None = None,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.padding_fn: Callable[[torch.Tensor], torch.Tensor] = self._causal_padding if causal else self._same_padding
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = torch.nn.Dropout(p_dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(self.padding_fn(x * x_mask))
        x = x * torch.sigmoid(1.702 * x) if self.activation == "gelu" else torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding_fn(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x: torch.Tensor) -> torch.Tensor:
        pad_l = self.conv_1.kernel_size[0] - 1
        return F.pad(x, convert_pad_shape([[0, 0], [0, 0], [pad_l, 0]]))

    def _same_padding(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.conv_1.kernel_size[0] - 1) // 2
        return F.pad(x, convert_pad_shape([[0, 0], [0, 0], [pad, pad]]))
