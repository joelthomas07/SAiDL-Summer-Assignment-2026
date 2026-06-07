"""
config.py  –  TransformerConfig dataclass + get_config helper.
"""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    # ── Model architecture ──────────────────────────────────────────
    d_model:    int   = 256     # Embedding dimension
    n_heads:    int   = 8       # Number of attention heads
    n_layers:   int   = 4       # Number of decoder layers
    d_ff:       int   = 1024    # Feed-forward hidden dimension
    dropout:    float = 0.1

    # ── Sequence / data ─────────────────────────────────────────────
    seq_len:    int   = 1024    # Context window length
    vocab_size: int   = 50257   # Set dynamically after tokenizer load

    # ── Training ────────────────────────────────────────────────────
    batch_size:       int   = 8
    grad_accum_steps: int   = 1     # Effective batch = batch_size * grad_accum
    lr:               float = 3e-4
    weight_decay:     float = 0.1
    warmup_steps:     int   = 200
    max_steps:        int   = 3000
    eval_interval:    int   = 300
    eval_iters:       int   = 5    # Batches averaged for validation perplexity

    # ── Paths ────────────────────────────────────────────────────────
    data_dir:       str = "data"
    checkpoint_dir: str = "checkpoints"
    run_name:       str = "baseline"

    # ── Swappable components ─────────────────────────────────────────
    attention_type:    str = "standard"    # see ATTENTION_REGISTRY in model.py
    pos_encoding_type: str = "sinusoidal"  # see POS_ENCODING_REGISTRY in model.py

    # ── Sliding-window attention ─────────────────────────────────────
    window_size: int = 256      # local context window for sliding_window attention

    # ── GQA / MQA ────────────────────────────────────────────────────
    # n_kv_heads < n_heads → GQA;  n_kv_heads = 1 → MQA
    n_kv_heads:  int = 2

    # ── RoPE positional interpolation ────────────────────────────────
    # scale = train_len / eval_len; e.g. 512/2048 = 0.25 for 4× extrapolation
    rope_scale:  float = 1.0

    # ── Convolution + Attention hybrids (Part 4) ─────────────────────
    # conv_arch: "none" | "conv_prefix" | "interleaved"  (see conv_variants.py)
    conv_arch:        str  = "none"
    conv_kernel_size: int  = 5      # local n-gram window for the Conv1D
    conv_depthwise:   bool = True   # depthwise-separable (efficient) vs full conv
    conv_first:       bool = True   # interleaved: even layers are conv if True

    # ── Device ───────────────────────────────────────────────────────
    device: str = "cuda"        # overridden at runtime if CUDA unavailable


def get_config(**overrides) -> TransformerConfig:
    """Return a TransformerConfig with any fields overridden by kwargs."""
    cfg = TransformerConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown config field: '{k}'")
        setattr(cfg, k, v)
    return cfg
