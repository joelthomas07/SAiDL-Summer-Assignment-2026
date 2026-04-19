from dataclasses import dataclass

@dataclass
class TransformerConfig:
    # Model architecture
    d_model: int = 256          # Embedding dimension (smaller for faster baseline)
    n_heads: int = 8            # Number of attention heads
    n_layers: int = 4           # Number of decoder layers
    d_ff: int = 1024            # Feed-forward hidden dimension
    dropout: float = 0.1

    # Sequence / data
    seq_len: int = 1024         # Context window length
    vocab_size: int = 50257     # GPT-2 BPE tokenizer vocab size (set after tokenizer load)

    # Training
    batch_size: int = 8         # Sequences per batch
    grad_accum_steps: int = 4   # Effective batch = batch_size * grad_accum_steps
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50        # Batches to average for validation perplexity

    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    run_name: str = "baseline"

    # Attention type: "standard" | "flash" (extend here for ablations)
    attention_type: str = "standard"

    # Positional encoding type: "sinusoidal" | "learned" (extend here for ablations)
    pos_encoding_type: str = "sinusoidal"

    # Device
    device: str = "cuda"        # overridden at runtime if cuda unavailable


def get_config(**overrides) -> TransformerConfig:
    cfg = TransformerConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
