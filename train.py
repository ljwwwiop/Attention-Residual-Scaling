"""
train.py — Multi-GPU DDP Training Entry Point
==============================================
Trains AttnRes or Vanilla transformer on text and logs comprehensive metrics.

Single-GPU:
    python train.py --config configs/tiny.yaml --model_type vanilla
    python train.py --config configs/tiny.yaml --model_type attnres_block

Multi-GPU (torchrun):
    torchrun --nproc_per_node=4 train.py --config configs/small.yaml --model_type attnres_block

Resume from checkpoint:
    python train.py --config configs/tiny.yaml --model_type attnres_block --resume runs/my_run/ckpt_step5000.pt

Arguments:
    --config        Path to YAML config file (required).
    --model_type    "vanilla" | "attnres_block" | "attnres_full"  (required).
    --run_name      Override experiment run name (default: auto-generated).
    --resume        Path to checkpoint to resume from.
    --compile       Use torch.compile for faster training (PyTorch ≥ 2.0).
    --max_docs      Override dataset max_docs (for "large" scale).
"""

import argparse
import contextlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

# ---------------------------------------------------------------------------
# Resolve project root on sys.path so local packages are importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import build_dataloader
from logger import ExperimentLogger
from models import ModelType, build_model, count_params


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def setup_distributed():
    """Initialise NCCL process group when launched via torchrun."""
    if "RANK" not in os.environ:
        # Single-process fallback
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
) -> None:
    """Save model + optimiser state to disk (rank 0 only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "step": step,
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> int:
    """Load checkpoint; returns the saved step number."""
    ckpt = torch.load(path, map_location=device)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    amp_ctx,
    max_batches: int = 100,
) -> Dict[str, float]:
    """Compute validation loss + perplexity over `max_batches` batches."""
    model.eval()
    total_loss = 0.0
    n = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_ctx:
            raw_model = model.module if isinstance(model, DDP) else model
            _, loss = raw_model(x, y)
        total_loss += loss.item()
        n += 1

    if n == 0:
        return {"loss": float("inf"), "ppl": float("inf")}

    avg_loss = total_loss / n
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")

    # Average across DDP ranks if applicable
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([avg_loss], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        avg_loss = t.item()
        try:
            ppl = math.exp(avg_loss)
        except OverflowError:
            ppl = float("inf")

    model.train()
    return {"loss": avg_loss, "ppl": ppl}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    # ---- Distributed setup -----------------------------------------------
    rank, world_size, device = setup_distributed()
    is_main = rank == 0

    # ---- Load config -------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Allow CLI overrides
    model_type: str = args.model_type
    if args.max_docs is not None:
        cfg.setdefault("dataset", {})["max_docs"] = args.max_docs

    # ---- Run name & logger -------------------------------------------------
    ds_scale = cfg["dataset"]["scale"]
    if args.run_name:
        run_name = args.run_name
    else:
        ts = time.strftime("%m%d_%H%M")
        run_name = f"{model_type}_{ds_scale}_{ts}"

    run_dir = Path(cfg["output"]["run_dir"]) / run_name
    log = ExperimentLogger(log_dir=str(run_dir), name=run_name, rank=rank)

    # TensorboardX writer — only rank 0 writes
    tb_dir = run_dir / "tensorboard"
    writer: Optional[SummaryWriter] = None
    if rank == 0:
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(logdir=str(tb_dir))
        log.info(f"TensorboardX log dir → {tb_dir.resolve()}")

    flat_cfg = {"model_type": model_type, "run_name": run_name, **cfg}
    log.log_config(flat_cfg)

    # ---- Reproducibility ---------------------------------------------------
    seed = cfg["experiment"]["seed"] + rank  # unique seed per rank
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ---- Build model -------------------------------------------------------
    log.info(f"Building model: {model_type}")
    model_cfg = cfg["model"]
    model = build_model(model_type, model_cfg).to(device)

    if args.compile and hasattr(torch, "compile"):
        log.info("torch.compile() enabled.")
        model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[device.index])

    param_counts = count_params(model.module if isinstance(model, DDP) else model)
    log.log_model_summary(param_counts)

    # ---- Optimizer ---------------------------------------------------------
    train_cfg = cfg["training"]

    # Separate weight-decay params (no decay for bias, norm weights)
    raw_model = model.module if isinstance(model, DDP) else model
    decay_params, no_decay_params = [], []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not any(k in name for k in ("norm", "bias", "embed")):
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg["max_lr"],
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device.type == "cuda",
    )
    log.info(
        f"Optimizer: AdamW | decay={len(decay_params)} params | "
        f"no_decay={len(no_decay_params)} params"
    )

    # ---- Mixed precision ---------------------------------------------------
    dtype_str = train_cfg.get("dtype", "bfloat16")
    if dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif dtype_str == "float16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    use_scaler = amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32))
    log.info(f"AMP dtype: {amp_dtype} | GradScaler: {use_scaler}")

    # ---- Data loaders ------------------------------------------------------
    log.info(f"Loading dataset: scale={ds_scale}")
    train_loader, train_sampler = build_dataloader(
        scale=ds_scale,
        split="train",
        max_seq_len=model_cfg["max_seq_len"],
        batch_size=cfg["dataset"]["batch_size"],
        rank=rank,
        world_size=world_size,
        num_workers=cfg["dataset"].get("num_workers", 4),
        cache_dir=cfg["dataset"].get("cache_dir"),
        max_docs=cfg["dataset"].get("max_docs"),
        seed=cfg["experiment"]["seed"],
    )
    val_loader, _ = build_dataloader(
        scale=ds_scale,
        split="val",
        max_seq_len=model_cfg["max_seq_len"],
        batch_size=cfg["dataset"]["batch_size"],
        rank=rank,
        world_size=world_size,
        num_workers=cfg["dataset"].get("num_workers", 4),
        cache_dir=cfg["dataset"].get("cache_dir"),
        max_docs=cfg["dataset"].get("max_docs"),
        seed=cfg["experiment"]["seed"],
    )
    log.info(f"Train batches/epoch: {len(train_loader):,} | Val batches: {len(val_loader):,}")

    # ---- Resume from checkpoint --------------------------------------------
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_step = load_checkpoint(resume_path, model, optimizer, device)
            log.info(f"Resumed from {resume_path} at step {start_step}")
        else:
            log.warning(f"Checkpoint not found: {resume_path}. Starting from scratch.")

    # ---- Training loop config ----------------------------------------------
    max_steps = train_cfg["max_steps"]
    warmup_steps = train_cfg["warmup_steps"]
    max_lr = train_cfg["max_lr"]
    min_lr = train_cfg["min_lr"]
    grad_clip = train_cfg["grad_clip"]
    grad_accum = cfg["dataset"]["grad_accum_steps"]
    eval_interval = train_cfg["eval_interval"]
    save_interval = train_cfg["save_interval"]
    log_interval = train_cfg["log_interval"]

    tokens_per_step = (
        cfg["dataset"]["batch_size"]
        * grad_accum
        * model_cfg["max_seq_len"]
        * world_size
    )
    log.info(
        f"Training: max_steps={max_steps} | "
        f"tokens/step={tokens_per_step:,} | "
        f"grad_accum={grad_accum} | world_size={world_size}"
    )

    # ---- Epoch / infinite data iterator ------------------------------------
    def _infinite_loader(loader, sampler):
        epoch = 0
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            yield from loader
            epoch += 1

    data_iter = _infinite_loader(train_loader, train_sampler)

    # ---- Main loop ---------------------------------------------------------
    model.train()
    optimizer.zero_grad()
    step = start_step
    total_tokens = 0
    t0 = time.perf_counter()

    while step < max_steps:
        step += 1

        # Set learning rate
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        micro_loss_sum = 0.0
        step_start = time.perf_counter()

        for micro in range(grad_accum):
            x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            is_last_micro = micro == grad_accum - 1
            sync_ctx = (
                contextlib.nullcontext()
                if (not isinstance(model, DDP)) or is_last_micro
                else model.no_sync()
            )

            with sync_ctx:
                with amp_ctx:
                    # Call through DDP wrapper so gradient-reduce hooks fire
                    # correctly on the last micro-step (sync_ctx controls when).
                    _raw = model.module if isinstance(model, DDP) else model
                    _, loss = _raw(x, y)
                loss_scaled = loss / grad_accum
                scaler.scale(loss_scaled).backward()

            micro_loss_sum += loss.item()

        avg_loss = micro_loss_sum / grad_accum

        # Gradient clip + optimiser step
        if use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip
        ).item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step_dt = time.perf_counter() - step_start
        tokens_this_step = (
            cfg["dataset"]["batch_size"]
            * grad_accum
            * model_cfg["max_seq_len"]
            * world_size
        )
        tokens_per_sec = tokens_this_step / max(step_dt, 1e-9)
        total_tokens += tokens_this_step

        # ---- Logging -------------------------------------------------------
        if step % log_interval == 0:
            log.log_step(
                step=step,
                total_steps=max_steps,
                loss=avg_loss,
                lr=lr,
                grad_norm=grad_norm,
                tokens_per_sec=tokens_per_sec,
            )
            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", grad_norm, step)
                writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)

        # ---- Evaluation ----------------------------------------------------
        if step % eval_interval == 0:
            val_metrics = evaluate(model, val_loader, device, amp_ctx)
            log.log_eval(step=step, split="val", **val_metrics)
            if writer is not None:
                writer.add_scalar("val/loss", val_metrics["loss"], step)
                writer.add_scalar("val/ppl", val_metrics["ppl"], step)

        # ---- Checkpoint ----------------------------------------------------
        if is_main and step % save_interval == 0:
            ckpt_path = run_dir / f"ckpt_step{step:07d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step, flat_cfg)
            log.info(f"Checkpoint saved → {ckpt_path}")

    # ---- Final evaluation --------------------------------------------------
    val_metrics = evaluate(model, val_loader, device, amp_ctx)
    log.log_eval(step=max_steps, split="val", **val_metrics)
    if writer is not None:
        writer.add_scalar("val/loss", val_metrics["loss"], max_steps)
        writer.add_scalar("val/ppl", val_metrics["ppl"], max_steps)

    if is_main:
        ckpt_path = run_dir / "ckpt_final.pt"
        save_checkpoint(ckpt_path, model, optimizer, max_steps, flat_cfg)
        log.info(f"Final checkpoint → {ckpt_path}")

    log.log_timing_summary(total_tokens)
    if writer is not None:
        writer.close()
    cleanup_distributed()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train AttnRes or Vanilla transformer (single/multi-GPU)"
    )
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument(
        "--model_type",
        required=True,
        choices=[t.value for t in ModelType],
        help="Model variant to train.",
    )
    p.add_argument(
        "--run_name",
        default=None,
        help="Override the auto-generated run name.",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint .pt file to resume training.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Apply torch.compile() for faster training (requires PyTorch ≥ 2.0).",
    )
    p.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Override max_docs for the 'large' dataset scale.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
