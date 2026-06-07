"""
Part 3: Positional Embedding Variants
======================================
Implements:
  - RoPE  (Rotary Positional Embedding)
  - ALiBi (Attention with Linear Biases)
  - Relative Positional Encoding (Shaw et al.)
  - Optional: Positional Interpolation / Scaling for context extension

Each variant is a drop-in module. Swap by passing `pos_enc_type` to your
TransformerModel.  The extrapolation test at the bottom trains on L=512 and
evaluates on L=512, 1024, 2048.

Usage
-----
    from positional_encodings import build_pos_enc
    pe = build_pos_enc("rope", d_model=256, max_seq_len=4096)

    # in your attention forward():
    #   RoPE  -> q, k = pe(q, k, seq_len)
    #   ALiBi -> attn_bias = pe(seq_len, device)  # add to attn logits
    #   Relative -> attn_logits += pe(seq_len)     # add rel bias to logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1.  RoPE  –  Rotary Positional Embedding
# ─────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    """
    RoPE from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (Su et al., 2021).

    Applies a rotation in 2-D subspaces of the head dimension so that the
    dot-product q·k naturally encodes *relative* position.

    Supports Positional Interpolation (Press et al.) via `scale` parameter:
        scale = train_len / target_len   e.g. 512/2048 = 0.25
    This linearly shrinks positions so the model stays within its trained range.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: int = 10000, scale: float = 1.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.scale = scale  # set <1.0 to enable positional interpolation

        # Precompute inverse frequencies  [head_dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache cos/sin tables up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        t = t / self.scale  # positional interpolation: compress positions
        freqs = torch.outer(t, self.inv_freq)           # [seq_len, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)         # [seq_len, head_dim]
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension into the first half."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        """
        Args:
            q, k: [batch, heads, seq_len, head_dim]
            seq_len: current sequence length
        Returns:
            q_rot, k_rot: same shape, rotated
        """
        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len)

        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # [1,1,seq,dim]
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ─────────────────────────────────────────────
# 2.  ALiBi  –  Attention with Linear Biases
# ─────────────────────────────────────────────
class ALiBi(nn.Module):
    """
    ALiBi from "Train Short, Test Long" (Press et al., 2021).

    Adds a fixed, non-learned linear penalty to attention logits:
        Attn(i,j)  +=  -m_h * |i - j|
    where m_h is a head-specific slope.  No position embeddings are added to
    the token embeddings at all.

    Key property: extrapolates *very* well beyond training length because the
    bias is defined for any distance, not just distances seen during training.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        slopes = self._get_slopes(num_heads)          # [num_heads]
        self.register_buffer("slopes", slopes)

    @staticmethod
    def _get_slopes(n: int) -> torch.Tensor:
        """
        Geometric sequence of slopes as in the ALiBi paper.
        Start from 2^(-8/n) and halve for each subsequent head.
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]

        if math.log2(n).is_integer():
            return torch.tensor(get_slopes_power_of_2(n), dtype=torch.float32)

        # For non-power-of-2 head counts: interpolate
        closest_pow2 = 2 ** math.floor(math.log2(n))
        base_slopes = get_slopes_power_of_2(closest_pow2)
        extra = get_slopes_power_of_2(2 * closest_pow2)[0::2]
        slopes = base_slopes + extra[: n - closest_pow2]
        return torch.tensor(slopes, dtype=torch.float32)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns an additive bias to be added to attention logits *before* softmax.

        Shape: [1, num_heads, seq_len, seq_len]
        """
        # Relative distance matrix: positions[i] - positions[j]
        positions = torch.arange(seq_len, device=device)
        # distance[i,j] = -(i - j) for causal mask friendly formulation
        # ALiBi penalizes distance so we use |i-j|; for causal, only j<=i matters
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq, seq]
        distances = distances.abs().float()

        # slopes: [num_heads, 1, 1]
        slopes = self.slopes.to(device).view(self.num_heads, 1, 1)
        bias = -slopes * distances.unsqueeze(0)   # [num_heads, seq, seq]
        return bias.unsqueeze(0)                   # [1, num_heads, seq, seq]


# ─────────────────────────────────────────────
# 3.  Relative Positional Encoding (Shaw et al.)
# ─────────────────────────────────────────────
class RelativePositionalEncoding(nn.Module):
    """
    Relative PE from "Self-Attention with Relative Position Representations"
    (Shaw et al., 2018).

    Learns an embedding for each relative offset clipped to [-max_relative, max_relative].
    These are added to the attention logit as:
        e_{ij} = (q_i W_Q)(k_j W_K)^T  +  q_i r_{i-j}^T
    where r_{i-j} is the learned relative embedding.

    Here we return the full [seq, seq, head_dim] relative key embeddings so
    your attention module can compute the q·r term.
    """

    def __init__(self, head_dim: int, max_relative_position: int = 128):
        super().__init__()
        self.max_relative_position = max_relative_position
        vocab_size = 2 * max_relative_position + 1   # offsets: -max .. 0 .. +max
        self.embeddings = nn.Embedding(vocab_size, head_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def _get_relative_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Returns clipped relative position indices [seq_len, seq_len]."""
        range_vec = torch.arange(seq_len, device=device)
        distance = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)          # [seq, seq]
        distance_clipped = distance.clamp(-self.max_relative_position,
                                           self.max_relative_position)
        return distance_clipped + self.max_relative_position                 # shift to [0, 2*max]

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns relative key embeddings: [seq_len, seq_len, head_dim]
        Usage in attention:
            rel_emb = rel_pe(seq_len, device)          # [T, T, d_head]
            rel_bias = torch.einsum('bhid,ijd->bhij', q, rel_emb)
            attn_logits = attn_logits + rel_bias
        """
        rel_positions = self._get_relative_positions(seq_len, device)       # [T, T]
        return self.embeddings(rel_positions)                                # [T, T, d_head]


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────
def build_pos_enc(pos_enc_type: str, d_model: int, num_heads: int = 8,
                  max_seq_len: int = 4096, max_relative_position: int = 128,
                  rope_scale: float = 1.0):
    """
    Returns the appropriate positional encoding module.

    pos_enc_type: one of "sinusoidal", "rope", "alibi", "relative"
    rope_scale  : set < 1.0 to enable positional interpolation for RoPE
                  e.g. rope_scale = train_len / eval_len = 512/2048 = 0.25
    """
    pos_enc_type = pos_enc_type.lower()
    head_dim = d_model // num_heads

    if pos_enc_type == "sinusoidal":
        return SinusoidalPE(d_model, max_seq_len)
    elif pos_enc_type == "rope":
        return RotaryEmbedding(head_dim, max_seq_len, scale=rope_scale)
    elif pos_enc_type == "alibi":
        return ALiBi(num_heads)
    elif pos_enc_type == "relative":
        return RelativePositionalEncoding(head_dim, max_relative_position)
    else:
        raise ValueError(f"Unknown pos_enc_type: {pos_enc_type}. "
                         "Choose from: sinusoidal, rope, alibi, relative")


# ─────────────────────────────────────────────
# Sinusoidal baseline (for reference / swap)
# ─────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    """Standard fixed sinusoidal PE added to token embeddings."""

    def __init__(self, d_model: int, max_seq_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_seq_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        return x + self.pe[:, :x.size(1)]


# ─────────────────────────────────────────────
# Attention module wrappers (how to integrate each PE)
# ─────────────────────────────────────────────
class MultiHeadAttentionWithPE(nn.Module):
    """
    A standard MHA that accepts any of the three PE variants.
    Shows exactly *where* each encoding plugs in.
    """

    def __init__(self, d_model: int, num_heads: int, pos_enc_type: str = "sinusoidal",
                 max_seq_len: int = 4096, dropout: float = 0.1, rope_scale: float = 1.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pos_enc_type = pos_enc_type.lower()

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.pos_enc = build_pos_enc(pos_enc_type, d_model, num_heads,
                                     max_seq_len=max_seq_len,
                                     rope_scale=rope_scale)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, H, T, d_head]"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        B, T, _ = x.shape

        q = self._split_heads(self.q_proj(x))   # [B, H, T, d_head]
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # ── apply PE ──────────────────────────────────────────────────────
        attn_bias = None

        if self.pos_enc_type == "rope":
            q, k = self.pos_enc(q, k, T)          # rotates q and k in place

        elif self.pos_enc_type == "alibi":
            attn_bias = self.pos_enc(T, x.device)  # [1, H, T, T], added to logits

        elif self.pos_enc_type == "relative":
            rel_emb = self.pos_enc(T, x.device)    # [T, T, d_head]
            # compute q·r^T term: [B, H, T, T]
            rel_bias = torch.einsum("bhid,ijd->bhij", q, rel_emb)
            attn_bias = rel_bias

        # sinusoidal: nothing to do here; PE was added before this module
        # ──────────────────────────────────────────────────────────────────

        scale = math.sqrt(self.head_dim)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / scale   # [B,H,T,T]

        if attn_bias is not None:
            attn_logits = attn_logits + attn_bias

        if causal_mask:
            mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)                          # [B,H,T,d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


# ─────────────────────────────────────────────
# Full Transformer block + model (modular)
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, pos_enc_type,
                 max_seq_len=4096, dropout=0.1, rope_scale=1.0):
        super().__init__()
        self.attn = MultiHeadAttentionWithPE(
            d_model, num_heads, pos_enc_type, max_seq_len, dropout, rope_scale)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class TransformerLM(nn.Module):
    """
    Language-model Transformer with swappable positional encoding.

    pos_enc_type: "sinusoidal" | "rope" | "alibi" | "relative"
    rope_scale  : only used when pos_enc_type="rope"; set to train_len/eval_len
                  for positional interpolation at longer context.
    """

    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4,
                 ffn_dim=1024, max_seq_len=4096, dropout=0.1,
                 pos_enc_type="sinusoidal", rope_scale=1.0):
        super().__init__()
        self.pos_enc_type = pos_enc_type.lower()

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Sinusoidal/learned PEs are added to embeddings; others are inside attn
        if self.pos_enc_type == "sinusoidal":
            self.pos_enc = SinusoidalPE(d_model, max_seq_len)
        else:
            self.pos_enc = None

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ffn_dim, pos_enc_type,
                             max_seq_len, dropout, rope_scale)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: [B, T] token indices -> logits [B, T, vocab_size]"""
        x = self.tok_emb(idx)                             # [B, T, D]
        if self.pos_enc is not None:
            x = self.pos_enc(x)                           # sinusoidal added here
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


# ─────────────────────────────────────────────
# Extrapolation Test  (Part 3c)
# ─────────────────────────────────────────────
import time
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    """Chunks a flat token tensor into fixed-length windows."""
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n = (len(tokens) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        start = i * self.seq_len
        x = self.tokens[start: start + self.seq_len]
        y = self.tokens[start + 1: start + self.seq_len + 1]
        return x, y


def evaluate_perplexity(model, tokens, seq_len, batch_size=16, device="cpu"):
    """Compute validation perplexity on token tensor at a given seq_len."""
    model.eval()
    dataset = TokenDataset(tokens, seq_len)
    loader  = DataLoader(dataset, batch_size=batch_size)
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            try:
                logits = model(x)                         # [B, T, V]
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
                total_loss   += loss.item()
                total_tokens += y.numel()
            except Exception as e:
                print(f"  [skip batch at seq_len={seq_len}] {e}")
                break

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def run_extrapolation_test(
    train_tokens: torch.Tensor,
    val_tokens:   torch.Tensor,
    vocab_size:   int,
    pos_enc_types = ("sinusoidal", "rope", "alibi", "relative"),
    train_seq_len: int = 512,
    eval_seq_lens  = (512, 1024, 2048),
    # Model hparams
    d_model: int = 256, num_heads: int = 8, num_layers: int = 4,
    ffn_dim: int = 1024, dropout: float = 0.1,
    # Training hparams
    batch_size: int = 32, lr: float = 3e-4, epochs: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    For each PE variant:
      1. Train on seq_len=512
      2. Evaluate perplexity at seq_len=512, 1024, 2048

    Prints a results table; returns dict of results.
    """
    results = {}

    for pe_type in pos_enc_types:
        print(f"\n{'='*60}")
        print(f"  PE type: {pe_type.upper()}")
        print(f"{'='*60}")

        # For RoPE positional interpolation, we train a separate model with
        # scale = 1.0; at eval time we rebuild with scale = train/eval
        model = TransformerLM(
            vocab_size, d_model, num_heads, num_layers, ffn_dim,
            max_seq_len=max(eval_seq_lens) + 1,
            dropout=dropout, pos_enc_type=pe_type, rope_scale=1.0,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

        train_dataset = TokenDataset(train_tokens, train_seq_len)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, drop_last=True)

        # ── Training ──────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss, n_batches = 0.0, 0
            t0 = time.time()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1
            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch}/{epochs} | loss={avg_loss:.4f} "
                  f"| time={time.time()-t0:.1f}s")

        # ── Extrapolation Evaluation ───────────────────────────
        ppl_results = {}
        for eval_len in eval_seq_lens:
            # Positional Interpolation for RoPE: rescale positions to fit
            if pe_type == "rope" and eval_len > train_seq_len:
                rope_scale = train_seq_len / eval_len    # e.g. 512/2048 = 0.25
                # Rebuild RoPE caches with interpolation scale
                for block in model.blocks:
                    block.attn.pos_enc.scale = rope_scale
                    block.attn.pos_enc._build_cache(eval_len)

            ppl = evaluate_perplexity(model, val_tokens, eval_len,
                                       batch_size=batch_size, device=device)
            ppl_results[eval_len] = ppl
            print(f"  eval_len={eval_len:5d} | perplexity={ppl:.2f}")

        results[pe_type] = ppl_results

    # ── Summary Table ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  EXTRAPOLATION TEST SUMMARY (train_len={train_seq_len})")
    print(f"{'─'*60}")
    header = f"{'PE Type':<12}" + "".join(f"{'L='+str(l):>12}" for l in eval_seq_lens)
    print(header)
    print("─" * len(header))
    for pe_type, ppls in results.items():
        row = f"{pe_type:<12}" + "".join(f"{ppls[l]:>12.2f}" for l in eval_seq_lens)
        print(row)

    return results


# ─────────────────────────────────────────────
# Quick self-test (no real data)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running shape/forward-pass sanity checks...\n")
    B, T, D, H = 2, 16, 128, 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # RoPE
    rope = RotaryEmbedding(D // H, max_seq_len=512).to(device)
    q = torch.randn(B, H, T, D // H, device=device)
    k = torch.randn(B, H, T, D // H, device=device)
    q_r, k_r = rope(q, k, T)
    print(f"RoPE      q_rot: {q_r.shape} ✓")

    # ALiBi
    alibi = ALiBi(H).to(device)
    bias = alibi(T, device)
    print(f"ALiBi     bias:  {bias.shape} ✓")

    # Relative PE
    rel_pe = RelativePositionalEncoding(D // H).to(device)
    rel_emb = rel_pe(T, device)
    print(f"RelativePE emb: {rel_emb.shape} ✓")

    # Full model forward pass for each variant
    vocab_size = 1000
    for pe_type in ("sinusoidal", "rope", "alibi", "relative"):
        model = TransformerLM(vocab_size, d_model=D, num_heads=H,
                              num_layers=2, ffn_dim=256,
                              pos_enc_type=pe_type).to(device)
        idx = torch.randint(0, vocab_size, (B, T), device=device)
        out = model(idx)
        print(f"TransformerLM [{pe_type:12s}] output: {out.shape} ✓")

    print("\nAll checks passed.")
    print("\nTo run the extrapolation test on WikiText-2, call:")
    print("  run_extrapolation_test(train_tokens, val_tokens, vocab_size)")
