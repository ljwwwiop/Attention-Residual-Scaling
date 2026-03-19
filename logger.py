"""
Experiment Logger
=================
Handles all logging for AttnRes vs Vanilla training runs.

Outputs per experiment run:
  runs/<name>/
      train.log      -- human-readable log (console mirror)
      metrics.csv    -- step-level metrics for plotting
      config.json    -- frozen config dump

Usage:
    logger = ExperimentLogger("runs/attnres_tiny", name="attnres_tiny", rank=0)
    logger.log_config(cfg)
    logger.log_step(step=100, total_steps=10000, loss=3.21, lr=3e-4, ...)
    logger.log_eval(step=500, split="val", loss=2.98, ppl=197.2)
"""

import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Unified logger: stdout + rotating log file + CSV metrics file.

    Only rank-0 in distributed training actually writes anything.
    All other ranks silently ignore all calls.

    Args:
        log_dir:  Directory where all outputs are written.
        name:     Experiment name (used as logger name and in log messages).
        rank:     Global process rank. Only rank 0 produces output.
    """

    def __init__(self, log_dir: str, name: str, rank: int = 0) -> None:
        self.log_dir = Path(log_dir)
        self.name = name
        self.rank = rank
        self.start_time = time.time()
        self._csv_header_written = False

        if rank != 0:
            # Dummy logger that discards everything
            self._logger = logging.getLogger(f"{name}_dummy_{rank}")
            self._logger.addHandler(logging.NullHandler())
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "metrics.csv"
        self._setup_logger()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_logger(self) -> None:
        """Configure console + file handlers."""
        fmt_console = logging.Formatter(
            fmt="[%(asctime)s][%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        fmt_file = logging.Formatter(
            fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.propagate = False

        # Console (INFO+)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt_console)
        logger.addHandler(ch)

        # File (DEBUG+)
        log_file = self.log_dir / "train.log"
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt_file)
        logger.addHandler(fh)

        self._logger = logger
        self._logger.info(f"=== Experiment: {self.name} ===")
        self._logger.info(f"Log directory : {self.log_dir.resolve()}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_config(self, cfg: Dict[str, Any]) -> None:
        """Dump config to JSON and print a summary."""
        if self.rank != 0:
            return
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        self._logger.info(f"Config saved → {config_path}")
        self._logger.info("--- Config ---")
        for k, v in cfg.items():
            if isinstance(v, dict):
                self._logger.info(f"  [{k}]")
                for kk, vv in v.items():
                    self._logger.info(f"    {kk}: {vv}")
            else:
                self._logger.info(f"  {k}: {v}")
        self._logger.info("--------------")

    def log_model_summary(self, param_counts: Dict[str, int]) -> None:
        """Log parameter counts broken down by component."""
        if self.rank != 0:
            return
        self._logger.info("--- Model Parameters ---")
        for component, count in param_counts.items():
            self._logger.info(f"  {component:<20s}: {count:>12,}")
        self._logger.info("-" * 40)

    def log_step(
        self,
        step: int,
        total_steps: int,
        loss: float,
        lr: float,
        grad_norm: float,
        tokens_per_sec: float,
        **extra_metrics: float,
    ) -> None:
        """Log a training step.

        Args:
            step:           Current global step (1-based).
            total_steps:    Total number of training steps.
            loss:           Scalar training loss.
            lr:             Current learning rate.
            grad_norm:      Gradient norm before clipping.
            tokens_per_sec: Training throughput.
            **extra_metrics: Any additional float metrics to log.
        """
        if self.rank != 0:
            return

        try:
            ppl = math.exp(loss)
        except OverflowError:
            ppl = float("inf")

        elapsed = time.time() - self.start_time
        pct = 100.0 * step / total_steps

        msg = (
            f"step {step:6d}/{total_steps} ({pct:5.1f}%) | "
            f"loss {loss:.4f} | ppl {ppl:9.2f} | "
            f"lr {lr:.2e} | gnorm {grad_norm:.3f} | "
            f"tok/s {tokens_per_sec:8.0f} | "
            f"elapsed {elapsed / 60:.1f}m"
        )
        for k, v in extra_metrics.items():
            msg += f" | {k} {v:.4f}"
        self._logger.info(msg)

        row: Dict[str, Any] = {
            "step": step,
            "loss": f"{loss:.6f}",
            "ppl": f"{ppl:.4f}",
            "lr": f"{lr:.2e}",
            "grad_norm": f"{grad_norm:.4f}",
            "tokens_per_sec": f"{tokens_per_sec:.1f}",
            "elapsed_sec": f"{elapsed:.1f}",
        }
        for k, v in extra_metrics.items():
            row[k] = f"{v:.6f}"
        self._write_csv(row)

    def log_eval(
        self,
        step: int,
        split: str,
        loss: float,
        ppl: float,
        **extra: float,
    ) -> None:
        """Log an evaluation result."""
        if self.rank != 0:
            return
        parts = [f"[EVAL {split.upper()}] step {step:6d} | loss {loss:.4f} | ppl {ppl:.2f}"]
        for k, v in extra.items():
            parts.append(f"{k} {v:.4f}")
        self._logger.info(" | ".join(parts))

        row: Dict[str, Any] = {
            "step": step,
            f"{split}_loss": f"{loss:.6f}",
            f"{split}_ppl": f"{ppl:.4f}",
        }
        for k, v in extra.items():
            row[k] = f"{v:.6f}"
        self._write_csv(row)

    def log_timing_summary(self, total_tokens: int) -> None:
        """Print final training timing summary."""
        if self.rank != 0:
            return
        elapsed = time.time() - self.start_time
        throughput = total_tokens / elapsed if elapsed > 0 else 0.0
        self._logger.info("=== Training Complete ===")
        self._logger.info(f"  Total time   : {elapsed / 60:.1f} min ({elapsed:.0f} s)")
        self._logger.info(f"  Total tokens : {total_tokens:,}")
        self._logger.info(f"  Avg tok/s    : {throughput:,.0f}")

    def info(self, msg: str) -> None:
        if self.rank == 0:
            self._logger.info(msg)

    def warning(self, msg: str) -> None:
        if self.rank == 0:
            self._logger.warning(msg)

    def debug(self, msg: str) -> None:
        if self.rank == 0:
            self._logger.debug(msg)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_csv(self, row: Dict[str, Any]) -> None:
        """Append one row to the CSV file, writing header on first call."""
        file_existed = self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
            if not file_existed or not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)
