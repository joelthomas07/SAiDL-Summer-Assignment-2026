"""
model.py  –  Decoder-only Transformer for causal language modelling.

All major components (attention, positional encoding, architectural blocks)
are registered in module-level registries so they can be swapped by name
via TransformerConfig without touching this file.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Registries  (add new variants here; reference them in config.py)
# ---------------------------------------------------------------------------
ATTENTION_REGISTRY: dict = {}
POS_ENCODING_REGISTRY: dict = {}
ARCHITECTURE_REGISTRY: dict = {}   # Part 4: conv/attention block layouts


def register_attention(name: str):
    def decorator(cls):
        ATTENTION_REGISTRY[name] = cls
        return cls
    return decorator


def register_pos_encoding(name: str):
    def decorator(cls):
        POS_ENCODING_REGISTRY[name] = cls
        return cls
    return decorator


def register_architecture(name: str):
    """
    Register a block-layout builder (Part 4).

    A builder has signature  build(cfg, make_attention, make_ffn) -> list[nn.Module]
    where make_attention()/make_ffn() are factories (provided by
    build_transformer) that already encapsulate the chosen attention variant
    and positional encoding.  This lets conv hybrids compose with ANY Part 2
    attention and ANY Part 3 PE.
    """
    def decorator(fn):
        ARCHITECTURE_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Layer Normalization
# ---------------------------------------------------------------------------
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps   = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias  = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# ---------------------------------------------------------------------------
# Feed-Forward Block
# ---------------------------------------------------------------------------
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))


# ---------------------------------------------------------------------------
# Input Embeddings
# ---------------------------------------------------------------------------
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model   = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


# ---------------------------------------------------------------------------
# Positional Encodings
# ---------------------------------------------------------------------------
@register_pos_encoding("sinusoidal")
class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding from 'Attention Is All You Need'."""

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


@register_pos_encoding("learned")
class LearnedPositionalEncoding(nn.Module):
    """Learnable absolute position embeddings (GPT-style)."""

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe      = nn.Embedding(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T         = x.shape[1]
        positions = torch.arange(T, device=x.device)
        return self.dropout(x + self.pe(positions))


# ---------------------------------------------------------------------------
# Attention Mechanisms
# ---------------------------------------------------------------------------
@register_attention("standard")
class MultiHeadAttentionBlock(nn.Module):
    """Standard scaled dot-product multi-head attention."""

    # PATCH: added cfg=None so build_transformer can pass cfg uniformly
    # to all attention classes without special-casing.
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
    def _attention(query, key, value, mask, dropout):
        d_k    = query.shape[-1]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores.softmax(dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return scores @ value, scores

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape
        query = self.w_q(q).view(B, T, self.h, self.d_k).transpose(1, 2)
        key   = self.w_k(k).view(B, T, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(B, T, self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = self._attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(x)


# ---------------------------------------------------------------------------
# Residual Connection  (pre-norm variant)
# ---------------------------------------------------------------------------
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm    = LayerNormalization(features)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


# ---------------------------------------------------------------------------
# Decoder Block  (self-attention only – no cross-attention for LM)
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, features, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block   = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, causal_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


# ---------------------------------------------------------------------------
# Full Decoder Stack
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm   = LayerNormalization(features)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, causal_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Projection Layer
# ---------------------------------------------------------------------------
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Top-level Decoder-Only Transformer
# ---------------------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, embeddings, pos_encoding, decoder, projection):
        super().__init__()
        self.embeddings   = embeddings
        self.pos_encoding = pos_encoding
        self.decoder      = decoder
        self.projection   = projection

    @staticmethod
    def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular mask; shape (1, 1, seq_len, seq_len)."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, idx, targets=None):
        B, T        = idx.shape
        causal_mask = self.make_causal_mask(T, idx.device)

        x      = self.embeddings(idx)
        x      = self.pos_encoding(x)
        x      = self.decoder(x, causal_mask)
        logits = self.projection(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx[:, -self.pos_encoding.pe.shape[1]:]
                if hasattr(self.pos_encoding, "pe")
                else idx
            )
            logits, _ = self(idx_cond)
            logits     = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, idx_next], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_transformer(cfg) -> DecoderOnlyTransformer:
    """
    Build a DecoderOnlyTransformer from a TransformerConfig.
    cfg is passed to every attention constructor so variants can
    read extra fields (window_size, n_kv_heads, etc.) without
    changing the call site.
    """
    if cfg.pos_encoding_type not in POS_ENCODING_REGISTRY:
        raise ValueError(
            f"Unknown pos_encoding_type '{cfg.pos_encoding_type}'. "
            f"Available: {list(POS_ENCODING_REGISTRY)}"
        )
    if cfg.attention_type not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention_type '{cfg.attention_type}'. "
            f"Available: {list(ATTENTION_REGISTRY)}"
        )

    embeddings   = InputEmbeddings(cfg.d_model, cfg.vocab_size)
    pos_encoding = POS_ENCODING_REGISTRY[cfg.pos_encoding_type](
        cfg.d_model, cfg.seq_len, cfg.dropout
    )

    # For PE types that act inside attention (rope/alibi/relative), store the
    # PE instance on cfg so attention wrappers can retrieve it, and auto-compose
    # the attention registry key  e.g. "rope__standard", "alibi__gqa", etc.
    _EMBEDDING_PE = {"sinusoidal", "learned"}
    effective_attn_type = cfg.attention_type
    if cfg.pos_encoding_type not in _EMBEDDING_PE:
        cfg._pe_instance = pos_encoding
        composed_key = f"{cfg.pos_encoding_type}__{cfg.attention_type}"
        if composed_key in ATTENTION_REGISTRY:
            effective_attn_type = composed_key
        # else: fall through to base attention (PE wrappers not registered yet –
        #        shouldn't happen if pos_encoding_variants is imported first)

    # Factories so Part 4 architecture builders can request fresh sub-modules
    # that already respect the chosen attention variant (Part 2) and the
    # positional-encoding wrapper (Part 3).
    def make_attention():
        return ATTENTION_REGISTRY[effective_attn_type](
            cfg.d_model, cfg.n_heads, cfg.dropout, cfg=cfg
        )

    def make_ffn():
        return FeedForwardBlock(cfg.d_model, cfg.d_ff, cfg.dropout)

    conv_arch = getattr(cfg, "conv_arch", "none")
    if conv_arch == "none":
        # Standard stack: every layer is attention + FFN.
        blocks = [
            DecoderBlock(cfg.d_model, make_attention(), make_ffn(), cfg.dropout)
            for _ in range(cfg.n_layers)
        ]
    elif conv_arch in ARCHITECTURE_REGISTRY:
        # Part 4 conv/attention hybrid (conv_prefix, interleaved, ...).
        blocks = ARCHITECTURE_REGISTRY[conv_arch](cfg, make_attention, make_ffn)
    else:
        raise ValueError(
            f"Unknown conv_arch '{conv_arch}'. "
            f"Available: {['none'] + list(ARCHITECTURE_REGISTRY)} "
            "(did you forget to `import conv_variants`?)"
        )

    decoder    = Decoder(cfg.d_model, nn.ModuleList(blocks))
    projection = ProjectionLayer(cfg.d_model, cfg.vocab_size)
    model      = DecoderOnlyTransformer(embeddings, pos_encoding, decoder, projection)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv_info = (
        f" | conv_arch={conv_arch}(k={getattr(cfg, 'conv_kernel_size', '?')})"
        if conv_arch != "none" else ""
    )
    print(
        f"[model] {n_params / 1e6:.2f}M params | "
        f"attn={cfg.attention_type} | pos={cfg.pos_encoding_type}"
        f"{conv_info}"
    )
    return model
