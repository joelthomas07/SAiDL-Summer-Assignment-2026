"""
train.py  –  Train a decoder-only Transformer on WikiText-2.

Usage:
    python train.py                          # default config
    python train.py --d_model 512 --n_layers 6 --n_heads 8
    python train.py --attention_type standard --pos_encoding_type learned
"""

import os
import math
from tqdm.auto import tqdm
import time
import argparse
import json
from pathlib import Path

import torch
import numpy as np

from config import get_config
from model  import build_transformer


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def download_and_tokenize(cfg):
    """
    Download WikiText-2, tokenize with GPT-2 BPE, and cache as .bin files.
    Requires:  pip install datasets transformers
    """
    cache_train = Path(cfg.data_dir) / "train.bin"
    cache_val   = Path(cfg.data_dir) / "val.bin"

    if cache_train.exists() and cache_val.exists():
        print("[data] Using cached tokenised data.")
        return

    print("[data] Downloading and tokenising WikiText-2 …")
    from datasets    import load_dataset
    from transformers import GPT2TokenizerFast

    os.makedirs(cfg.data_dir, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def encode_split(split_name):
        texts = "\n\n".join(
            t for t in dataset["validation" if split_name == "val" else split_name]["text"] if t.strip()
        )
        ids = tokenizer.encode(texts)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(str(Path(cfg.data_dir) / f"{split_name}.bin"))
        print(f"  {split_name}: {len(ids):,} tokens → "
              f"{Path(cfg.data_dir) / f'{split_name}.bin'}")
        return len(ids)

    encode_split("train")
    encode_split("val")
    print("[data] Done.")


def get_batch(split: str, cfg, device: torch.device):
    """Sample a random batch of (inputs, targets) from cached token file."""
    path = Path(cfg.data_dir) / ("train.bin" if split == "train" else "val.bin")
    data = np.frombuffer(path.read_bytes(), dtype=np.uint16).astype(np.int64)

    # Random starting positions
    ix = torch.randint(len(data) - cfg.seq_len, (cfg.batch_size,))
    x  = torch.stack([torch.from_numpy(data[i     : i + cfg.seq_len    ]) for i in ix])
    y  = torch.stack([torch.from_numpy(data[i + 1 : i + cfg.seq_len + 1]) for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, cfg, device):
    model.eval()
    results = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch(split, cfg, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        mean_loss = float(np.mean(losses))
        results[split] = {
            "loss":       mean_loss,
            "perplexity": math.exp(mean_loss),
        }
    model.train()
    return results


# ---------------------------------------------------------------------------
# Learning-rate schedule  (linear warmup → cosine decay)
# ---------------------------------------------------------------------------
def get_lr(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(cfg):
    # --- device ---
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU.")
        cfg.device = "cpu"
    device = torch.device(cfg.device)
    print(f"[train] device = {device}")

    # --- data ---
    download_and_tokenize(cfg)

    # --- model ---
    from transformers import GPT2TokenizerFast
    tokenizer    = GPT2TokenizerFast.from_pretrained("gpt2")
    cfg.vocab_size = tokenizer.vocab_size   # 50257
    import attention_variants  # must run before registry is read

    model = build_transformer(cfg).to(device)

    # --- optimiser (AdamW with weight-decay on weights only) ---
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimiser = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )

    # --- checkpoint dir ---
    ckpt_dir = Path(cfg.checkpoint_dir) / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- logging ---
    log_path = ckpt_dir / "metrics.jsonl"
    log_file = open(log_path, "w")

    # --- training ---
    best_val_ppl = float("inf")
    step         = 0
    t0           = time.time()

    print(f"[train] Starting training for {cfg.max_steps} steps …")
    model.train()

    while step < cfg.max_steps:
        # -- lr update --
        lr = get_lr(step, cfg)
        for pg in optimiser.param_groups:
            pg["lr"] = lr

        # -- gradient accumulation --
        optimiser.zero_grad()
        accum_loss = 0.0

        for micro_step in range(cfg.grad_accum_steps):
            x, y = get_batch("train", cfg, device)
            _, loss = model(x, y)
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            accum_loss += loss.item() if loss.numel() == 1 else loss.mean().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        step += 1

        # -- periodic evaluation --
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            elapsed = time.time() - t0
            metrics = estimate_loss(model, cfg, device)

            tok_per_sec = (
                cfg.eval_interval * cfg.grad_accum_steps
                * cfg.batch_size * cfg.seq_len / elapsed
            )

            print(
                f"step {step:>5d}/{cfg.max_steps} | "
                f"train_ppl={metrics['train']['perplexity']:>8.2f} | "
                f"val_ppl={metrics['val']['perplexity']:>8.2f} | "
                f"lr={lr:.2e} | "
                f"tok/s={tok_per_sec:,.0f}"
            )

            record = {
                "step":           step,
                "train_loss":     metrics["train"]["loss"],
                "train_ppl":      metrics["train"]["perplexity"],
                "val_loss":       metrics["val"]["loss"],
                "val_ppl":        metrics["val"]["perplexity"],
                "lr":             lr,
                "tokens_per_sec": tok_per_sec,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            # save best checkpoint
            if metrics["val"]["perplexity"] < best_val_ppl:
                best_val_ppl = metrics["val"]["perplexity"]
                ckpt_path = ckpt_dir / "best.pt"
                torch.save(
                    {
                        "step":      step,
                        "model":     model.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "val_ppl":   best_val_ppl,
                        "config":    cfg.__dict__,
                    },
                    ckpt_path,
                )
                print(f"  ✓ Saved best checkpoint  (val_ppl={best_val_ppl:.2f})")

            t0 = time.time()

    log_file.close()
    print(f"\n[train] Done.  Best val perplexity: {best_val_ppl:.2f}")
    print(f"[train] Metrics log: {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train decoder-only Transformer on WikiText-2")

    # Expose all config fields as CLI flags
    parser.add_argument("--d_model",           type=int,   default=256)
    parser.add_argument("--n_heads",           type=int,   default=8)
    parser.add_argument("--n_layers",          type=int,   default=4)
    parser.add_argument("--d_ff",              type=int,   default=1024)
    parser.add_argument("--dropout",           type=float, default=0.1)
    parser.add_argument("--seq_len",           type=int,   default=1024)
    parser.add_argument("--batch_size",        type=int,   default=8)
    parser.add_argument("--grad_accum_steps",  type=int,   default=4)
    parser.add_argument("--lr",                type=float, default=3e-4)
    parser.add_argument("--weight_decay",      type=float, default=0.1)
    parser.add_argument("--warmup_steps",      type=int,   default=200)
    parser.add_argument("--max_steps",         type=int,   default=5000)
    parser.add_argument("--eval_interval",     type=int,   default=250)
    parser.add_argument("--eval_iters",        type=int,   default=50)
    parser.add_argument("--data_dir",          type=str,   default="data")
    parser.add_argument("--checkpoint_dir",    type=str,   default="checkpoints")
    parser.add_argument("--run_name",          type=str,   default="baseline")
    parser.add_argument("--attention_type",    type=str,   default="standard",
                        choices=["standard", "sliding_window", "mqa", "linear"])
    parser.add_argument("--pos_encoding_type", type=str,   default="sinusoidal",
                        choices=["sinusoidal", "learned"])
    parser.add_argument("--device",            type=str,   default="cuda")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = get_config(**vars(args))
    train(cfg)
