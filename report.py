"""
report.py  –  Generate the comparative table required by Core ML Part 5.

Reads every metrics.jsonl under --checkpoint_dir, pulls the best snapshot
from each run, and prints a formatted table + saves report.csv.

Usage:
    python report.py                          # scans ./checkpoints/
    python report.py --checkpoint_dir runs/
    python report.py --sort val_ppl           # default sort column
"""

import json
import argparse
import csv
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_run(run_dir: Path) -> dict | None:
    """
    Parse metrics.jsonl in run_dir.
    Returns a dict of summary statistics, or None if the file is missing/empty.
    """
    log_path = run_dir / "metrics.jsonl"
    if not log_path.exists():
        return None

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return None

    # Best = lowest val_ppl
    best = min(records, key=lambda r: r.get("val_ppl", float("inf")))
    # Final step (for throughput / memory which tend to stabilise)
    final = records[-1]

    # epoch time: average over all logged values (skipping None entries)
    epoch_times = [
        r["avg_epoch_time_sec"]
        for r in records
        if r.get("avg_epoch_time_sec") is not None
    ]
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else None

    # Peak memory: take the maximum reported (it's a running peak anyway)
    peak_mems = [r["peak_memory_mb"] for r in records if r.get("peak_memory_mb")]
    peak_mem  = max(peak_mems) if peak_mems else None

    return {
        "run":              run_dir.name,
        "steps":            best["step"],
        "best_val_ppl":     best.get("val_ppl"),
        "best_val_loss":    best.get("val_loss"),
        "final_train_ppl":  final.get("train_ppl"),
        "final_train_loss": final.get("train_loss"),
        "peak_memory_mb":   peak_mem,
        "peak_memory_gb":   round(peak_mem / 1024, 3) if peak_mem else None,
        "avg_epoch_time_s": round(avg_epoch_time, 1) if avg_epoch_time else None,
        "tokens_per_sec":   round(final.get("tokens_per_sec", 0)),
    }


def fmt(value, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{decimals}f}{suffix}"
    return str(value) + suffix


def print_table(rows: list[dict]) -> None:
    """Pretty-print the comparative table to stdout."""
    # Column definitions: (header, key, decimals, suffix)
    columns = [
        ("Run",              "run",              0,  ""),
        ("Steps",            "steps",            0,  ""),
        ("Val PPL ↓",        "best_val_ppl",     2,  ""),
        ("Val Loss ↓",       "best_val_loss",    4,  ""),
        ("Train PPL",        "final_train_ppl",  2,  ""),
        ("Peak Mem (MB)",    "peak_memory_mb",   1,  ""),
        ("Peak Mem (GB)",    "peak_memory_gb",   3,  ""),
        ("Epoch Time (s)",   "avg_epoch_time_s", 1,  ""),
        ("Throughput (tok/s)","tokens_per_sec",  0,  ""),
    ]

    # Build display rows
    display = []
    for r in rows:
        display.append([
            r["run"],
            fmt(r["steps"], 0),
            fmt(r["best_val_ppl"],     2),
            fmt(r["best_val_loss"],    4),
            fmt(r["final_train_ppl"],  2),
            fmt(r["peak_memory_mb"],   1),
            fmt(r["peak_memory_gb"],   3),
            fmt(r["avg_epoch_time_s"], 1),
            fmt(r["tokens_per_sec"],   0),
        ])

    headers = [c[0] for c in columns]
    widths  = [
        max(len(h), max((len(row[i]) for row in display), default=0))
        for i, h in enumerate(headers)
    ]

    sep  = "  ".join("-" * w for w in widths)
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))

    print("\n" + "═" * len(sep))
    print("  COMPARATIVE RESULTS TABLE  (Core ML – Part 5)")
    print("═" * len(sep))
    print(line)
    print(sep)
    for row in display:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
    print("═" * len(sep) + "\n")


def save_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[report] CSV saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparative results table")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--sort",
        type=str,
        default="best_val_ppl",
        choices=[
            "best_val_ppl", "tokens_per_sec", "peak_memory_mb",
            "avg_epoch_time_s", "run",
        ],
        help="Column to sort results by (ascending)",
    )
    parser.add_argument("--csv_out", type=str, default="report.csv")
    args = parser.parse_args()

    ckpt_root = Path(args.checkpoint_dir)
    if not ckpt_root.exists():
        print(f"[report] Checkpoint dir '{ckpt_root}' not found.")
        return

    # Collect all run directories that contain a metrics.jsonl
    run_dirs = sorted(
        d for d in ckpt_root.iterdir()
        if d.is_dir() and (d / "metrics.jsonl").exists()
    )

    if not run_dirs:
        print(
            f"[report] No metrics.jsonl files found under '{ckpt_root}'.\n"
            "         Run train.py first, then re-run report.py."
        )
        return

    rows = []
    for d in run_dirs:
        row = load_run(d)
        if row:
            rows.append(row)
        else:
            print(f"[report] Skipped '{d.name}' – empty or unreadable log.")

    if not rows:
        print("[report] No valid runs to display.")
        return

    # Sort
    rows.sort(
        key=lambda r: (
            r[args.sort] if r[args.sort] is not None else float("inf")
        )
    )

    print_table(rows)
    save_csv(rows, Path(args.csv_out))

    # Highlight best run
    best = rows[0]
    print(
        f"[report] Best run by {args.sort}: '{best['run']}'\n"
        f"         val_ppl={fmt(best['best_val_ppl'], 2)}  |  "
        f"tok/s={fmt(best['tokens_per_sec'], 0)}  |  "
        f"peak_mem={fmt(best['peak_memory_mb'], 1)} MB"
    )


if __name__ == "__main__":
    main()
