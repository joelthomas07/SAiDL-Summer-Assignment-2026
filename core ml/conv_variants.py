"""
conv_variants.py  –  Convolution + Attention hybrids for Core ML Part 4.

──────────────────────────────────────────────────────────────────────────────
Part 4(a)  –  A 1-D convolutional component that captures local n-gram context
              cheaply:  `CausalConv1d`, a *depthwise-separable* causal conv over
              the time axis.  Depthwise-separable = a depthwise conv (one filter
              per channel, mixing a window of `kernel_size` neighbouring tokens)
              followed by a 1x1 pointwise conv (mixing across channels).  Cost is
              O(T · k · d) instead of O(T² · d) for attention, so a kernel of a
              few tokens efficiently models local n-gram structure.

Part 4(b)  –  Two hybrid block layouts, registered in ARCHITECTURE_REGISTRY so
              they compose with ANY attention variant (Part 2) and ANY positional
              encoding (Part 3) — exactly what Part 4(c) needs.

    conv_prefix   A Conv1D sub-layer is inserted BEFORE self-attention inside
                  EVERY decoder block:
                      x → +conv → +attention → +ffn
                  Each token first absorbs its local neighbourhood, then
                  attention handles long-range dependencies.

    interleaved   Whole blocks alternate.  Even layers are pure Conv blocks
                  (a causal conv REPLACES attention as the token-mixer); odd
                  layers are standard attention blocks:
                      L0:  +conv  → +ffn
                      L1:  +attn  → +ffn
                      L2:  +conv  → +ffn
                      ...
                  Roughly halves the number of O(T²) attention layers.

How it plugs in
---------------
`build_transformer` (model.py) hands every registered architecture builder two
factories — `make_attention()` and `make_ffn()` — that already bake in the
selected attention variant and positional encoding.  The builders below only
decide *how blocks are arranged*; they never re-implement attention or PE, so:

    python train.py --conv_arch conv_prefix --attention_type gqa  --pos_encoding_type rope
    python train.py --conv_arch interleaved --attention_type mqa  --pos_encoding_type alibi

all work without further changes.

Import this module BEFORE calling build_transformer so the registry is
populated.  train.py and eval_extrap.py both `import conv_variants`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    register_architecture,
    DecoderBlock,
    ResidualConnection,
)


# ─────────────────────────────────────────────────────────────────────────────
# Part 4(a)  –  Causal 1-D convolutional component
# ─────────────────────────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    """
    Causal 1-D convolution over the time axis.

    Input / output shape:  (B, T, d_model).

    Causality:  we left-pad the sequence by (kernel_size - 1) and use no right
    padding, so the output at position t depends only on inputs ≤ t.  This is
    the convolutional equivalent of the lower-triangular attention mask — no
    token can peek at the future.

    Depthwise-separable (default, `depthwise=True`):
        1. depthwise conv  – groups=d_model, one k-tap filter per channel.
           Mixes a local window of tokens *within* each feature.   (params: d·k)
        2. pointwise conv  – 1x1 conv, mixes information *across* channels.
                                                                    (params: d·d)
        Total ≈ d·k + d²  vs  a full conv's  d²·k  — much cheaper, and the
        standard choice for "efficient local context" (Part 4 wording).

    Set `depthwise=False` for an ordinary (dense) causal Conv1d.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dropout: float,
        depthwise: bool = True,
    ) -> None:
        super().__init__()
        assert kernel_size >= 1, "kernel_size must be ≥ 1"
        self.d_model     = d_model
        self.kernel_size = kernel_size
        self.depthwise   = depthwise

        if depthwise:
            # one filter per channel (groups == d_model) + 1x1 channel mixer
            self.conv      = nn.Conv1d(
                d_model, d_model, kernel_size, groups=d_model, bias=False
            )
            self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        else:
            self.conv      = nn.Conv1d(d_model, d_model, kernel_size)
            self.pointwise = nn.Identity()

        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) → (B, C, T) for nn.Conv1d
        x = x.transpose(1, 2)
        # causal left-pad: output[t] sees only inputs in (t-k, t]
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        x = self.pointwise(x)
        x = self.act(x)
        x = x.transpose(1, 2)                       # back to (B, T, C)
        return self.dropout(x)


def _make_conv(cfg) -> CausalConv1d:
    """Build a CausalConv1d from config fields (with safe defaults)."""
    return CausalConv1d(
        d_model=cfg.d_model,
        kernel_size=getattr(cfg, "conv_kernel_size", 5),
        dropout=cfg.dropout,
        depthwise=getattr(cfg, "conv_depthwise", True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Part 4(b)  –  Hybrid blocks
# ─────────────────────────────────────────────────────────────────────────────
class ConvPrefixDecoderBlock(nn.Module):
    """
    Design 1  –  Conv1D BEFORE attention, in every block.

    Pre-norm block with three residual sub-layers:
        x = x + conv(LN x)
        x = x + attention(LN x)
        x = x + ffn(LN x)

    The conv supplies local n-gram features; attention then mixes globally.
    """

    def __init__(
        self,
        features: int,
        conv: nn.Module,
        self_attention_block: nn.Module,
        feed_forward_block: nn.Module,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv                 = conv
        self.self_attention_block = self_attention_block
        self.feed_forward_block   = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # 1) local conv mixing (causal via left-padding, ignores mask)
        x = self.residual_connections[0](x, self.conv)
        # 2) global self-attention (uses the causal mask)
        x = self.residual_connections[1](
            x, lambda y: self.self_attention_block(y, y, y, causal_mask)
        )
        # 3) position-wise feed-forward
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class ConvBlock(nn.Module):
    """
    Design 2 building block  –  a causal Conv1D REPLACES attention.

    Pre-norm block with two residual sub-layers:
        x = x + conv(LN x)
        x = x + ffn(LN x)

    Used on alternate layers in the `interleaved` architecture.  `causal_mask`
    is accepted (so the block is drop-in for the Decoder loop) but unused —
    causality is enforced by the conv's left-padding.
    """

    def __init__(
        self,
        features: int,
        conv: nn.Module,
        feed_forward_block: nn.Module,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv               = conv
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](x, self.conv)
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Architecture builders  (registered → selectable via cfg.conv_arch)
# Each receives `cfg` plus the two factories from build_transformer and returns
# a plain list of blocks (the caller wraps it in nn.ModuleList / Decoder).
# ─────────────────────────────────────────────────────────────────────────────
@register_architecture("conv_prefix")
def build_conv_prefix(cfg, make_attention, make_ffn) -> list:
    """Conv1D before attention in EVERY decoder block (Design 1)."""
    return [
        ConvPrefixDecoderBlock(
            cfg.d_model,
            _make_conv(cfg),
            make_attention(),
            make_ffn(),
            cfg.dropout,
        )
        for _ in range(cfg.n_layers)
    ]


@register_architecture("interleaved")
def build_interleaved(cfg, make_attention, make_ffn) -> list:
    """
    Alternate Conv blocks and Attention blocks (Design 2).

    With cfg.conv_first=True (default): even layers (0,2,4,…) are Conv blocks,
    odd layers (1,3,5,…) are Attention blocks.  Set conv_first=False to flip.
    """
    conv_first = getattr(cfg, "conv_first", True)
    blocks = []
    for i in range(cfg.n_layers):
        is_conv_layer = (i % 2 == 0) if conv_first else (i % 2 == 1)
        if is_conv_layer:
            blocks.append(
                ConvBlock(cfg.d_model, _make_conv(cfg), make_ffn(), cfg.dropout)
            )
        else:
            blocks.append(
                DecoderBlock(cfg.d_model, make_attention(), make_ffn(), cfg.dropout)
            )
    return blocks
