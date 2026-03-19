"""
eval.py — Evaluation & Side-by-Side Comparison
================================================
Evaluates trained checkpoints and reports loss, perplexity, and throughput.
Supports comparing two models (e.g. vanilla vs attnres_block) head-to-head.

Single model evaluation:
    python eval.py \\
        --config configs/tiny.yaml \\
        --model_type attnres_block \\
        --checkpoint runs/attnres_block_small_0101_1200/ckpt_final.pt

Side-by-side comparison (vanilla vs AttnRes):
    python eval.py \\
        --config configs/tiny.yaml \\
        --model_type vanilla \\
        --checkpoint runs/vanilla_small_.../ckpt_final.pt \\
        --compare_type attnres_block \\
        --compare_checkpoint runs/attnres_block_small_.../ckpt_final.pt

Outputs:
  - Formatted table to stdout
  - JSON report saved to the same directory as the first checkpoint
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets import build_dataloader
from logger import ExperimentLogger
from models import build_model, count_params


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp_ctx,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Full evaluation pass: computes average loss, perplexity, and throughput.

    Args:
        model:       Model to evaluate (should be in eval mode).
        loader:      DataLoader for the split to evaluate.
        device:      Target device.
        amp_ctx:     AMP autocast context.
        max_batches: Cap on number of batches (None = full dataset).

    Returns:
        Dict with keys: loss, ppl, tokens_per_sec, total_tokens, elapsed_sec.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    total_tokens = 0
    t0 = time.perf_counter()

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with amp_ctx:
            _, loss = model(x, y)

        total_loss += loss.item()
        n_batches += 1
        total_tokens += x.numel()

    elapsed = time.perf_counter() - t0

    if n_batches == 0:
        return {"loss": float("inf"), "ppl": float("inf"), "tokens_per_sec": 0.0,
                "total_tokens": 0, "elapsed_sec": 0.0}

    avg_loss = total_loss / n_batches
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")

    return {
        "loss": avg_loss,
        "ppl": ppl,
        "tokens_per_sec": total_tokens / max(elapsed, 1e-9),
        "total_tokens": total_tokens,
        "elapsed_sec": elapsed,
    }


@torch.no_grad()
def measure_throughput(
    model: torch.nn.Module,
    device: torch.device,
    amp_ctx,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    n_warmup: int = 10,
    n_bench: int = 50,
) -> float:
    """Measure peak forward-pass throughput in tokens/second.

    Args:
        n_warmup: Number of warm-up iterations (discarded).
        n_bench:  Number of timed iterations.

    Returns:
        Tokens per second (float).
    """
    model.eval()
    dummy_x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dummy_y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warm up
    for _ in range(n_warmup):
        with amp_ctx:
            model(dummy_x, dummy_y)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(n_bench):
        with amp_ctx:
            model(dummy_x, dummy_y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return (batch_size * seq_len * n_bench) / elapsed


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(
    checkpoint_path: Path,
    model_type: str,
    model_cfg: dict,
    device: torch.device,
) -> torch.nn.Module:
    """Build model, load weights from checkpoint, return in eval mode."""
    model = build_model(model_type, model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)  # support bare state_dict too
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _sep(width: int = 72) -> str:
    return "─" * width


def _header(title: str, width: int = 72) -> str:
    pad = (width - len(title) - 2) // 2
    return "─" * pad + f" {title} " + "─" * pad


def print_results_table(
    results: Dict[str, Dict],
    param_counts: Dict[str, Dict],
    throughputs: Dict[str, float],
) -> None:
    """Print a formatted comparison table to stdout."""
    col_w = 20
    names = list(results.keys())

    print()
    print(_header("AttnRes Scaling-Law Benchmark"))
    print()

    # Header row
    row = f"{'Metric':<28}" + "".join(f"{n:>{col_w}}" for n in names)
    print(row)
    print(_sep(28 + col_w * len(names)))

    # Parameters
    for comp, count in list(param_counts[names[0]].items()):
        vals = "".join(
            f"{param_counts[n].get(comp, 0):>{col_w},}" for n in names
        )
        label = comp[:27]
        print(f"  {label:<26}{vals}")
    print()

    # Metrics per split
    all_splits = sorted({s for r in results.values() for s in r.keys()})
    for split in all_splits:
        print(f"  [{split.upper()}]")
        metrics = ["loss", "ppl"]
        for m in metrics:
            vals = "".join(
                f"{results[n][split].get(m, float('nan')):>{col_w}.4f}"
                for n in names
            )
            print(f"    {m:<24}{vals}")
        print()

    # Throughput
    print("  [THROUGHPUT]")
    tp_row = "".join(f"{throughputs.get(n, 0):>{col_w},.0f}" for n in names)
    print(f"    {'tokens/sec':<24}{tp_row}")

    # Improvement deltas (only meaningful when comparing exactly 2 models)
    if len(names) == 2:
        n1, n2 = names
        print()
        print(_header("Delta (second - first)"))
        for split in all_splits:
            delta_loss = results[n2][split]["loss"] - results[n1][split]["loss"]
            delta_ppl = results[n2][split]["ppl"] - results[n1][split]["ppl"]
            sign_l = "▼" if delta_loss < 0 else "▲"
            sign_p = "▼" if delta_ppl < 0 else "▲"
            print(
                f"  {split}: loss {sign_l}{abs(delta_loss):.4f}  "
                f"ppl {sign_p}{abs(delta_ppl):.2f}"
            )
        tp1 = throughputs.get(n1, 1)
        tp2 = throughputs.get(n2, 1)
        if tp1 > 0:
            ratio = tp2 / tp1
            print(f"  throughput ratio: {ratio:.3f}× ({n2} / {n1})")

    print(_sep())
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate and compare AttnRes / Vanilla transformer checkpoints."
    )
    p.add_argument("--config", required=True, help="YAML config used for training.")
    p.add_argument(
        "--model_type",
        required=True,
        help="Type of the primary model: vanilla | attnres_block | attnres_full",
    )
    p.add_argument("--checkpoint", required=True, help="Path to primary checkpoint .pt")
    p.add_argument(
        "--compare_type",
        default=None,
        help="(Optional) Model type for the comparison model.",
    )
    p.add_argument(
        "--compare_checkpoint",
        default=None,
        help="(Optional) Path to comparison checkpoint .pt",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="Splits to evaluate: val, test, train (default: val test)",
    )
    p.add_argument(
        "--max_batches",
        type=int,
        default=200,
        help="Max batches per eval split (default: 200). Use 0 for full.",
    )
    p.add_argument(
        "--throughput_batches",
        type=int,
        default=50,
        help="Number of forward passes for throughput benchmark.",
    )
    p.add_argument(
        "--dtype",
        default=None,
        help="Override dtype: bfloat16 | float16 | float32",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    ds_scale = cfg["dataset"]["scale"]

    # ---- AMP context -------------------------------------------------------
    dtype_str = args.dtype or cfg.get("training", {}).get("dtype", "bfloat16")
    if dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif dtype_str == "float16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    amp_ctx = torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=(amp_dtype != torch.float32),
    )

    # ---- Build logger (stdout only for eval) --------------------------------
    log = ExperimentLogger(
        log_dir=str(Path(args.checkpoint).parent / "eval"),
        name="eval",
        rank=0,
    )
    log.info(f"Device: {device} | AMP dtype: {amp_dtype}")

    # ---- Load models -------------------------------------------------------
    models_to_eval = {args.model_type: Path(args.checkpoint)}
    if args.compare_checkpoint and args.compare_type:
        models_to_eval[args.compare_type] = Path(args.compare_checkpoint)

    loaded_models = {}
    param_counts_map = {}
    for mtype, ckpt_path in models_to_eval.items():
        log.info(f"Loading {mtype} from {ckpt_path} …")
        m = load_model_from_checkpoint(ckpt_path, mtype, model_cfg, device)
        loaded_models[mtype] = m
        param_counts_map[mtype] = count_params(m)
        log.info(f"  Parameters: {param_counts_map[mtype]['total']:,}")

    # ---- Throughput benchmarks (use val loader shape) ----------------------
    throughputs: Dict[str, float] = {}
    bench_batch = cfg["dataset"]["batch_size"]
    bench_seq = model_cfg["max_seq_len"]
    bench_vocab = model_cfg["vocab_size"]

    for mtype, m in loaded_models.items():
        log.info(f"Benchmarking throughput for {mtype} …")
        tp = measure_throughput(
            m, device, amp_ctx, bench_batch, bench_seq, bench_vocab,
            n_warmup=10, n_bench=args.throughput_batches,
        )
        throughputs[mtype] = tp
        log.info(f"  {mtype} throughput: {tp:,.0f} tokens/sec")

    # ---- Evaluation per split ----------------------------------------------
    all_results: Dict[str, Dict] = {mtype: {} for mtype in loaded_models}
    max_batches_arg = args.max_batches if args.max_batches > 0 else None

    for split in args.splits:
        log.info(f"Evaluating split: {split} (max_batches={max_batches_arg}) …")
        loader, _ = build_dataloader(
            scale=ds_scale,
            split=split,
            max_seq_len=model_cfg["max_seq_len"],
            batch_size=cfg["dataset"]["batch_size"],
            rank=0,
            world_size=1,
            num_workers=cfg["dataset"].get("num_workers", 2),
            cache_dir=cfg["dataset"].get("cache_dir"),
            max_docs=cfg["dataset"].get("max_docs"),
        )
        for mtype, m in loaded_models.items():
            metrics = run_eval(m, loader, device, amp_ctx, max_batches=max_batches_arg)
            all_results[mtype][split] = metrics
            log.info(
                f"  [{split}] {mtype}: loss={metrics['loss']:.4f} "
                f"ppl={metrics['ppl']:.2f} "
                f"tok/s={metrics['tokens_per_sec']:,.0f}"
            )

    # ---- Print comparison table --------------------------------------------
    print_results_table(all_results, param_counts_map, throughputs)

    # ---- Save JSON report --------------------------------------------------
    report = {
        "config": cfg,
        "checkpoints": {k: str(v) for k, v in models_to_eval.items()},
        "results": {
            mtype: {
                split: {k: float(v) for k, v in metrics.items()}
                for split, metrics in splits.items()
            }
            for mtype, splits in all_results.items()
        },
        "throughputs": {k: float(v) for k, v in throughputs.items()},
        "param_counts": {
            mtype: {k: int(v) for k, v in pc.items()}
            for mtype, pc in param_counts_map.items()
        },
    }
    report_path = Path(args.checkpoint).parent / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
