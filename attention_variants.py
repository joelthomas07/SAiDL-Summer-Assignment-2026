"""
attention_variants.py
Register new attention mechanisms into model.py's ATTENTION_REGISTRY.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import register_attention  # type: ignore


# -------------------------------------------------------
# 1. Sliding Window (Local) Attention
# -------------------------------------------------------
@register_attention("sliding_window")
class SlidingWindowAttention(nn.Module):
    """Each token attends only to a local window of `window` past tokens."""

    def __init__(self, d_model: int, h: int, dropout: float, window: int = 64):
        super().__init__()
        assert d_model % h == 0
        self.h, self.d_k, self.window = h, d_model // h, window
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape
        Q = self.w_q(q).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(B, T, self.h, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        i = torch.arange(T, device=q.device).unsqueeze(1)
        j = torch.arange(T, device=q.device).unsqueeze(0)
        local_mask = (j <= i) & (i - j < self.window)
        scores = scores.masked_fill(~local_mask, -1e9)

        x = self.dropout(scores.softmax(dim=-1)) @ V
        return self.w_o(x.transpose(1, 2).contiguous().view(B, T, self.h * self.d_k))


# -------------------------------------------------------
# 2. Multi-Query Attention (MQA)
# -------------------------------------------------------
@register_attention("mqa")
class MultiQueryAttention(nn.Module):
    """Single K and V shared across all heads — reduces KV cache at inference."""

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0
        self.h, self.d_k = h, d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.d_k, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape
        Q = self.w_q(q).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.w_k(k).unsqueeze(1).expand(B, self.h, T, self.d_k)
        V = self.w_v(v).unsqueeze(1).expand(B, self.h, T, self.d_k)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        x = self.dropout(scores.softmax(dim=-1)) @ V
        return self.w_o(x.transpose(1, 2).contiguous().view(B, T, self.h * self.d_k))


# -------------------------------------------------------
# 3. Linear Attention (O(n) via kernel trick)
# -------------------------------------------------------
@register_attention("linear")
class LinearAttention(nn.Module):
    """
    Softmax-free linear attention using feature map φ(x) = elu(x) + 1.
    Computes KV context matrix first → O(n·d²) vs O(n²·d).
    """

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0
        self.h, self.d_k = h, d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _phi(x):
        return F.elu(x) + 1

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape
        Q = self._phi(self.w_q(q).view(B, T, self.h, self.d_k).transpose(1, 2))
        K = self._phi(self.w_k(k).view(B, T, self.h, self.d_k).transpose(1, 2))
        V = self.w_v(v).view(B, T, self.h, self.d_k).transpose(1, 2)

        KV = torch.einsum("bhtn,bhtm->bhtnm", K, V).cumsum(dim=2)
        K_sum = K.cumsum(dim=2)

        num = torch.einsum("bhtn,bhtnm->bhtm", Q, KV)
        den = (Q * K_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)

        x = (num / den).transpose(1, 2).contiguous().view(B, T, self.h * self.d_k)
        return self.w_o(x)
