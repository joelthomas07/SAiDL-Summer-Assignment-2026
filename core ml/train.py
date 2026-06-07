"""
train.py  â€“  Train a decoder-only Transformer on WikiText-2.

Usage:
    python train.py                                               # default config
    python train.py --attention_type mqa --pos_encoding_type learned
    python train.py --attention_type sliding_window --window_size 128
    python train.py --attention_type gqa --n_kv_heads 2
"""

import os
import math
import time
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm.auto import tqdm

from config import get_config
from model  import build_transformer
import attention_variants      # registers sliding_window / mqa / linear / gqa
import pos_encoding_variants   # registers rope / alibi / relative + wrapped attention combos
import conv_variants           # registers conv_prefix / interleaved hybrid architectures


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def download_and_tokenize(cfg) -> None:
    """Download WikiText-2, tokenise with GPT-2 BPE, cache as uint16 .bin files."""
    cache_train = Path(cfg.data_dir) / "train.bin"
    cache_val   = Path(cfg.data_dir) / "val.bin"

    if cache_train.exists() and cache_val.exists():
        print("[data] Using cached tokenised data.")
        return

    print("[data] Downloading and tokenising WikiText-2 â€¦")
    from datasets     import load_dataset
    from transformers import GPT2TokenizerFast

    os.makedirs(cfg.data_dir, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    dataset   = load_dataset("wikitext", "wikitext-2-raw-v1")

    def encode_split(split_name: str) -> int:
        hf_split = "validation" if split_name == "val" else split_name
        texts    = "\n\n".join(
            t for t in dataset[hf_split]["text"] if t.strip()
        )
        ids = tokenizer.encode(texts)
        np.array(ids, dtype=np.uint16).tofile(
            str(Path(cfg.data_dir) / f"{split_name}.bin")
        )
        print(f"  {split_name}: {len(ids):,} tokens")
        return len(ids)

    encode_split("train")
    encode_split("val")
    print("[data] Done.")


def total_tokens_in_split(cfg, split: str) -> int:
    """Return number of tokens in the cached .bin file (uint16 â†’ 2 bytes each)."""
    path = Path(cfg.data_dir) / ("train.bin" if split == "train" else "val.bin")
    return path.stat().st_size // 2


def get_batch(split: str, cfg, device: torch.device):
    """Sample a random batch of (inputs, targets) from the cached token file."""
    path = Path(cfg.data_dir) / ("train.bin" if split == "train" else "val.bin")
    data = np.frombuffer(path.read_bytes(), dtype=np.uint16).astype(np.int64)
    ix   = torch.randint(len(data) - cfg.seq_len, (cfg.batch_size,))
    x    = torch.stack([torch.from_numpy(data[i     : i + cfg.seq_len    ]) for i in ix])
    y    = torch.stack([torch.from_numpy(data[i + 1 : i + cfg.seq_len + 1]) for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, cfg, device) -> dict:
    """
    Returns {split: {loss, perplexity}} averaged over cfg.eval_iters batches.
    Timing is intentionally excluded from the caller's throughput window.
    """
    model.eval()
    results = {}
    for split in ("train", "val"):
        losses = [
            model(*get_batch(split, cfg, device))[1].item()
            for _ in range(cfg.eval_iters)
        ]
        mean   = float(np.mean(losses))
        results[split] = {"loss": mean, "perplexity": math.exp(mean)}
    model.train()
    return results


# ---------------------------------------------------------------------------
# Learning-rate schedule  (linear warmup â†’ cosine decay to 0)
# ---------------------------------------------------------------------------
def get_lr(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------
def peak_memory_mb(device: torch.device) -> float:
    """Peak allocated GPU memory in MB since last reset. Returns 0 on CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return 0.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(cfg) -> None:
    # â”€â”€ device â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU.")
        cfg.device = "cpu"
    device = torch.device(cfg.device)
    print(f"[train] device = {device}")

    # â”€â”€ data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    download_and_tokenize(cfg)

    # Tokens-per-epoch: how many tokens constitute one full pass over train data.
    # Used to fire epoch-boundary events even though sampling is random.
    train_total_tokens = total_tokens_in_split(cfg, "train")
    tokens_per_step    = cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps
    steps_per_epoch    = max(1, train_total_tokens // tokens_per_step)
    print(
        f"[train] train tokens={train_total_tokens:,} | "
        f"tokens/step={tokens_per_step:,} | "
        f"~steps/epoch={steps_per_epoch}"
    )

    # â”€â”€ model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from transformers import GPT2TokenizerFast
    cfg.vocab_size = GPT2TokenizerFast.from_pretrained("gpt2").vocab_size  # 50 257
    model          = build_transformer(cfg).to(device)

    # â”€â”€ optimiser (AdamW, weight-decay on weight matrices only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    decay     = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay  = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() <  2]
    optimiser = torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )

    # â”€â”€ checkpoint / log dirs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ckpt_dir = Path(cfg.checkpoint_dir) / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "metrics.jsonl"
    log_file = open(log_path, "w")

    # â”€â”€ GPU memory: reset once so max_memory_allocated tracks full-run peak â”€â”€
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # â”€â”€ epoch-time tracking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    epoch_times: list[float] = []
    epoch_start              = time.perf_counter()
    current_epoch            = 0

    # â”€â”€ throughput window: t0 resets AFTER eval so eval time is excluded â”€â”€â”€â”€â”€
    t0 = time.perf_counter()

    best_val_ppl = float("inf")
    step         = 0
    model.train()
    print(f"[train] Starting â€“ {cfg.max_steps} steps â€¦\n")

    while step < cfg.max_steps:
        # â”€â”€ learning rate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        lr = get_lr(step, cfg)
        for pg in optimiser.param_groups:
            pg["lr"] = lr

        # â”€â”€ gradient accumulation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        optimiser.zero_grad()
        for _ in range(cfg.grad_accum_steps):
            x, y  = get_batch("train", cfg, device)
            _, loss = model(x, y)
            (loss / cfg.grad_accum_steps).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        step += 1

        # â”€â”€ epoch boundary detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Fire when cumulative steps cross a full epoch boundary.
        if step >= (current_epoch + 1) * steps_per_epoch:
            epoch_wall = time.perf_counter() - epoch_start
            epoch_times.append(epoch_wall)
            current_epoch += 1
            epoch_start = time.perf_counter()

        # â”€â”€ periodic evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            # PATCH: measure elapsed BEFORE eval so eval latency is excluded
            # from the throughput window.
            t_before_eval  = time.perf_counter()
            training_secs  = t_before_eval - t0
            tok_per_sec    = (
                cfg.eval_interval * tokens_per_step / max(training_secs, 1e-6)
            )

            metrics       = estimate_loss(model, cfg, device)
            peak_mem_mb   = peak_memory_mb(device)
            avg_epoch_sec = float(np.mean(epoch_times)) if epoch_times else None

            train_loss = metrics["train"]["loss"]
            val_loss   = metrics["val"]["loss"]
            val_ppl    = metrics["val"]["perplexity"]

            print(
                f"step {step:>5}/{cfg.max_steps} | "
                f"train_loss={train_loss:>7.4f} | "
                f"val_loss={val_loss:>7.4f} | "
                f"val_ppl={val_ppl:>8.2f} | "
                f"peak_mem={peak_mem_mb:>7.1f} MB | "
                + (f"epoch_t={avg_epoch_sec:.1f}s | " if avg_epoch_sec else "")
                + f"tok/s={tok_per_sec:>9,.0f}"
            )

            record = {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "peak_memory_mb": peak_mem_mb,
                "avg_epoch_time_sec": avg_epoch_sec,
                "tokens_per_sec": tok_per_sec,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                torch.save(
                    {
                        "step":      step,
                        "model":     model.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "val_ppl":   best_val_ppl,
                        "config":    cfg.__dict__,
                    },
                    ckpt_dir / "best.pt",
                )
                print(f"  âœ“ New best checkpoint  (val_ppl={best_val_ppl:.2f})")

            # PATCH: reset t0 AFTER eval so eval latency doesn't inflate
            # the next interval's throughput measurement.
            t0 = time.perf_counter()

    log_file.close()
    print(f"\n[train] Done.  Best val perplexity: {best_val_ppl:.2f}")
    print(f"[train] Metrics log: {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a decoder-only Transformer on WikiText-2"
    )

    # Architecture
    p.add_argument("--d_model",           type=int,   default=256)
    p.add_argument("--n_heads",           type=int,   default=8)
    p.add_argument("--n_layers",          type=int,   default=4)
    p.add_argument("--d_ff",              type=int,   default=1024)
    p.add_argument("--dropout",           type=float, default=0.1)

    # Sequence / data
    p.add_argument("--seq_len",           type=int,   default=1024)

    # Training
    p.add_argument("--batch_size",        type=int,   default=8)
    p.add_argument("--grad_accum_steps",  type=int,   default=4)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--weight_decay",      type=float, default=0.1)
    p.add_argument("--warmup_steps",      type=int,   default=200)
    p.add_argument("--max_steps",         type=int,   default=5000)
    p.add_argument("--eval_interval",     type=int,   default=250)
    p.add_argument("--eval_iters",        type=int,   default=50)

    # Paths
    p.add_argument("--data_dir",          type=str,   default="data")
    p.add_argument("--checkpoint_dir",    type=str,   default="checkpoints")
    p.add_argument("--run_name",          type=str,   default="baseline")

    # Swappable components
    p.add_argument(
        "--attention_type",
        type=str,
        default="standard",
        choices=["standard", "sliding_window", "mqa", "linear", "gqa"],
    )
    p.add_argument(
        "--pos_encoding_type",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learned", "rope", "alibi", "relative"],
    )

    # Attention-variant hyperparams
    p.add_argument("--window_size",  type=int, default=256,
                   help="Local window for sliding_window attention")
    p.add_argument("--n_kv_heads",   type=int, default=2,
                   help="KV head count for gqa attention (1 = MQA, n_heads = MHA)")
    # RoPE positional interpolation: set to train_len/eval_len for longer contexts
    # e.g. --rope_scale 0.25 when training on 512 and evaluating on 2048
    p.add_argument("--rope_scale",   type=float, default=1.0,
                   help="Positional interpolation scale for RoPE (train_len/eval_len)")

    # Convolution + Attention hybrids (Part 4)
    p.add_argument(
        "--conv_arch",
        type=str,
        default="none",
        choices=["none", "conv_prefix", "interleaved"],
        help="Conv/attention block layout (Part 4). none = plain transformer.",
    )
    p.add_argument("--conv_kernel_size", type=int, default=5,
                   help="Local n-gram window for the Conv1D component")
    p.add_argument("--conv_depthwise", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Depthwise-separable conv (--no-conv_depthwise for full conv)")
    p.add_argument("--conv_first", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="interleaved: even layers are conv if set (default), else odd")

    # Device
    p.add_argument("--device",       type=str, default="cuda")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = get_config(**vars(args))
    train(cfg)

