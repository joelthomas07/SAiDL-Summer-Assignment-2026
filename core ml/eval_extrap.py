"""
eval_extrap.py  –  Part 3c Extrapolation Test

Loads every best.pt checkpoint under --checkpoint_dir, evaluates validation
perplexity at L=512, L=1024, and L=2048, then prints a clean table.

Usage (run after training all four 512-length models):
    python eval_extrap.py
    python eval_extrap.py --checkpoint_dir checkpoints --data_dir data
"""

import argparse
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import get_config
from model  import build_transformer, POS_ENCODING_REGISTRY
import attention_variants
import pos_encoding_variants
import conv_variants


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_val_tokens(data_dir: str) -> np.ndarray:
    path = Path(data_dir) / "val.bin"
    if not path.exists():
        raise FileNotFoundError(
            f"val.bin not found at {path}. "
            "Run train.py first to download and tokenise WikiText-2."
        )
    return np.frombuffer(path.read_bytes(), dtype=np.uint16).astype(np.int64)


@torch.no_grad()
def eval_perplexity(model, val_tokens: np.ndarray, seq_len: int,
                    batch_size: int, device: torch.device) -> float:
    """Slide a window of seq_len over val_tokens, return perplexity."""
    model.eval()
    data = torch.from_numpy(val_tokens)
    n    = (len(data) - 1) // seq_len   # number of non-overlapping windows

    if n == 0:
        return float("inf")

    total_loss, total_tokens = 0.0, 0
    for start in range(0, n * seq_len, batch_size * seq_len):
        batch_xs, batch_ys = [], []
        for i in range(start, min(start + batch_size * seq_len, n * seq_len), seq_len):
            batch_xs.append(data[i     : i + seq_len])
            batch_ys.append(data[i + 1 : i + seq_len + 1])
            if len(batch_xs) == batch_size:
                break

        if not batch_xs:
            break

        x = torch.stack(batch_xs).to(device)
        y = torch.stack(batch_ys).to(device)

        try:
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
            )
            total_loss   += loss.item()
            total_tokens += y.numel()
        except Exception as e:
            # Sequence length unsupported by this PE at this eval length
            print(f"      [skip at seq_len={seq_len}] {e}")
            return float("inf")

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Part 3c Extrapolation Test")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--data_dir",       type=str, default="data")
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--eval_lens",      type=int, nargs="+",
                        default=[512, 1024, 2048])
    parser.add_argument("--device",         type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[extrap] device = {device}")

    val_tokens = load_val_tokens(args.data_dir)
    print(f"[extrap] val tokens = {len(val_tokens):,}\n")

    ckpt_root = Path(args.checkpoint_dir)
    run_dirs  = sorted(
        d for d in ckpt_root.iterdir()
        if d.is_dir() and (d / "best.pt").exists()
    )

    if not run_dirs:
        print(f"No best.pt checkpoints found under '{ckpt_root}'.")
        print("Make sure you've run train.py for each PE variant first.")
        return

    results = {}   # run_name -> {eval_len: ppl}

    for run_dir in run_dirs:
        ckpt_path = run_dir / "best.pt"
        print(f"{'─'*55}")
        print(f"  Run: {run_dir.name}")

        ckpt = torch.load(ckpt_path, map_location=device)
        saved_cfg = ckpt["config"]

        # Rebuild config from saved values, keeping seq_len flexible
        cfg = get_config(**{
            k: v for k, v in saved_cfg.items()
            if k != "seq_len"   # we'll override per eval
        })
        cfg.device = str(device)

        pe_type   = saved_cfg.get("pos_encoding_type", "sinusoidal")
        train_len = saved_cfg.get("seq_len", 512)
        max_eval_len = max(args.eval_lens)
        print(f"  pos_encoding_type = {pe_type}  |  trained at seq_len = {train_len}")

        run_results = {}

        for eval_len in args.eval_lens:
            # Rebuild model with eval_len as the max seq_len for the PE cache
            cfg.seq_len = eval_len

            # For sinusoidal/learned PE, the buffer must be >= eval_len.
            # We build the model at max_eval_len once, then reuse for shorter evals.
            cfg.seq_len = max_eval_len

            # RoPE positional interpolation: compress positions to stay in
            # the trained range when eval_len > train_len
            if pe_type == "rope" and eval_len > train_len:
                cfg.rope_scale = train_len / eval_len
                print(f"  [RoPE] applying positional interpolation "
                      f"scale={cfg.rope_scale:.4f} for eval_len={eval_len}")
            else:
                cfg.rope_scale = 1.0

            model = build_transformer(cfg).to(device)

            # Sinusoidal PE has a fixed-size buffer saved in the checkpoint
            # (shape [1, train_len, d_model]). The model was built at max_eval_len
            # so its buffer is already the right size. Pop the saved buffer so
            # load_state_dict doesn't try to overwrite it with the wrong shape.
            state = ckpt["model"]
            state = {k: v for k, v in state.items() if k != "pos_encoding.pe"}
            missing, unexpected = model.load_state_dict(state, strict=False)
            other_issues = [k for k in missing + unexpected if "pos_encoding" not in k]
            if other_issues:
                print(f"  [warn] unexpected mismatches: {other_issues}")

            ppl = eval_perplexity(model, val_tokens, eval_len,
                                  args.batch_size, device)
            run_results[eval_len] = ppl
            print(f"    eval_len={eval_len:5d}  →  val_ppl = {ppl:.2f}")

        results[run_dir.name] = run_results

    # ── Summary table ────────────────────────────────────────────────────────
    eval_lens = args.eval_lens
    col_w     = 12
    name_w    = max(len(n) for n in results) + 2

    header = f"{'Run':<{name_w}}" + "".join(f"{'L='+str(l):>{col_w}}" for l in eval_lens)
    sep    = "─" * len(header)

    print(f"\n{'═'*len(header)}")
    print("  EXTRAPOLATION TEST RESULTS  (trained on L=512)")
    print(f"{'═'*len(header)}")
    print(header)
    print(sep)
    for run_name, ppls in sorted(results.items()):
        row = f"{run_name:<{name_w}}"
        for l in eval_lens:
            val = ppls.get(l, float("inf"))
            cell = f"{val:.2f}" if val != float("inf") else "OOM/err"
            row += f"{cell:>{col_w}}"
        print(row)
    print(f"{'═'*len(header)}\n")

    # Save to JSON for your report
    out_path = Path("extrap_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[extrap] Results saved → {out_path}")


if __name__ == "__main__":
    main()
