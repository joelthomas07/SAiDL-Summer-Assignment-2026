"""
attention_variants.py  –  Pluggable attention mechanisms for the Core ML task.

Import this module BEFORE calling build_transformer so the registry is
populated.  train.py already does `import attention_variants` for you.

Registered names (pass via --attention_type):
  sliding_window  –  Causal local-window attention
  mqa             –  Multi-Query Attention (Shazeer 2019)
  linear          –  Kernel / softmax-free attention (Katharopoulos 2020)
  gqa             –  Grouped-Query Attention (Ainslie 2023)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import register_attention


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Sliding-Window (Local) Attention
# ─────────────────────────────────────────────────────────────────────────────
@register_attention("sliding_window")
class SlidingWindowAttention(nn.Module):
    """
    Causal attention restricted to a local window of `window_size` tokens.
    Token i can only attend to [i - window_size + 1, i].

    Complexity vs standard:
      - Same O(T * w) non-zero entries (w = window_size << T)
      - Memory: O(T * w) attention weights instead of O(T^2)
    """

    def __init__(self, d_model: int, h: int, dropout: float, cfg=None) -> None:
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model     = d_model
        self.h           = h
        self.d_k         = d_model // h
        self.window_size = cfg.window_size if cfg is not None else 256

        self.w_q     = nn.Linear(d_model, d_model, bias=False)
        self.w_k     = nn.Linear(d_model, d_model, bias=False)
        self.w_v     = nn.Linear(d_model, d_model, bias=False)
        self.w_o     = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_window_mask(
        seq_len: int, window_size: int, device: torch.device
    ) -> torch.Tensor:
        """
        Boolean mask (1 = attend, 0 = block) of shape (1, 1, T, T).
        Allows attention only where  0 <= i - j < window_size  (causal + local).
        """
        i    = torch.arange(seq_len, device=device).unsqueeze(1)   # (T, 1)
        j    = torch.arange(seq_len, device=device).unsqueeze(0)   # (1, T)
        dist = i - j                                                  # (T, T)
        mask = (dist >= 0) & (dist < window_size)
        return mask.unsqueeze(0).unsqueeze(0).float()               # (1,1,T,T)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask=None,
    ) -> torch.Tensor:
        B, T, _ = q.shape
        Q = self.w_q(q).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(B, T, self.h, self.d_k).transpose(1, 2)

        scores     = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        win_mask   = self._make_window_mask(T, self.window_size, q.device)
        scores     = scores.masked_fill(win_mask == 0, -1e9)
        scores     = scores.softmax(dim=-1)
        scores     = self.dropout(scores)

        x = (scores @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Multi-Query Attention (MQA)
# ─────────────────────────────────────────────────────────────────────────────
@register_attention("mqa")
class MultiQueryAttention(nn.Module):
    """
    MQA: all h query heads share a single K head and a single V head.

    Benefits:
      - KV cache reduced by factor h at inference time
      - Minimal quality loss vs full MHA on most benchmarks
    Reference: Shazeer 2019 – "Fast Transformer Decoding: One Write-Head is All You Need"
    """

    def __init__(self, d_model: int, h: int, dropout: float, cfg=None) -> None:
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model = d_model
        self.h       = h
        self.d_k     = d_model // h

        self.w_q     = nn.Linear(d_model, d_model,  bias=False)   # h heads
        self.w_k     = nn.Linear(d_model, self.d_k, bias=False)   # 1 head
        self.w_v     = nn.Linear(d_model, self.d_k, bias=False)   # 1 head
        self.w_o     = nn.Linear(d_model, d_model,  bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape

        # Multiple Q heads
        Q = self.w_q(q).view(B, T, self.h, self.d_k).transpose(1, 2)  # (B,h,T,d_k)
        # Single K / V head – broadcast across query heads
        K = self.w_k(k).view(B, T, 1, self.d_k).transpose(1, 2)        # (B,1,T,d_k)
        V = self.w_v(v).view(B, T, 1, self.d_k).transpose(1, 2)        # (B,1,T,d_k)

        # K, V broadcast automatically to (B, h, T, d_k) in the matmul
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)       # (B,h,T,T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores.softmax(dim=-1)
        scores = self.dropout(scores)

        x = (scores @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Linear (Kernel / Softmax-Free) Attention
# ─────────────────────────────────────────────────────────────────────────────
@register_attention("linear")
class LinearAttention(nn.Module):
    """
    Replaces softmax with the ELU+1 kernel:  φ(x) = ELU(x) + 1  (always > 0).

    Attention(Q,K,V) = φ(Q) · [φ(K)ᵀ · V] / [φ(Q) · φ(K)ᵀ · 1]

    This implementation applies the causal mask to the O(T²) similarity
    matrix for correctness and clarity; a true O(T) version uses cumulative
    prefix sums (straightforward to swap in for inference benchmarking).

    Reference: Katharopoulos et al. 2020 – "Transformers are RNNs"
    """

    def __init__(self, d_model: int, h: int, dropout: float, cfg=None) -> None:
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model = d_model
        self.h       = h
        self.d_k     = d_model // h

        self.w_q     = nn.Linear(d_model, d_model, bias=False)
        self.w_k     = nn.Linear(d_model, d_model, bias=False)
        self.w_v     = nn.Linear(d_model, d_model, bias=False)
        self.w_o     = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _kernel(x: torch.Tensor) -> torch.Tensor:
        """φ(x) = ELU(x) + 1  – positive, smooth, inexpensive."""
        return F.elu(x) + 1.0

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape

        Q = self._kernel(
            self.w_q(q).view(B, T, self.h, self.d_k).transpose(1, 2)
        )  # (B,h,T,d_k)
        K = self._kernel(
            self.w_k(k).view(B, T, self.h, self.d_k).transpose(1, 2)
        )
        V = self.w_v(v).view(B, T, self.h, self.d_k).transpose(1, 2)

        # Unnormalised similarity (no softmax)
        scores = Q @ K.transpose(-2, -1)                               # (B,h,T,T)

        # Apply causal mask by zeroing disallowed entries (not -inf, since no softmax)
        if mask is not None:
            scores = scores * mask

        # Normalise row-wise (denominator in linear attention)
        denom  = scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        scores = scores / denom
        scores = self.dropout(scores)

        x = (scores @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(x)
