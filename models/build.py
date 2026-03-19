"""
Model factory for scaling-law comparison experiments.

Both model variants use the SAME standard GPT-2 style architecture defined in
network.py.  The only difference is the residual connection mode:

  "vanilla"       → GPT(use_attnres=False)                  (additive residuals)
  "attnres_block" → GPT(use_attnres=True, variant="block")  (Block AttnRes, O(N·d))
  "attnres_full"  → GPT(use_attnres=True, variant="full")   (Full AttnRes, O(L·d))

This ensures a fair comparison: same architecture, same parameter count
(excluding the tiny pseudo-query vectors added by AttnRes).

YAML config key mapping:
  d_model      → n_embd
  n_layers     → n_layer   (full blocks; each block = 1 attn + 1 MLP sub-layer)
  n_heads      → n_head
  vocab_size   → vocab_size
  max_seq_len  → block_size
  dropout      → dropout
  n_blocks     → num_attnres_blocks  (only used for AttnRes variants)
  bias         → bias                (optional, default False)
  [n_kv_heads, ffn_mult, rope_theta are ignored — GPT-2 style does not use them]

Usage:
    model = build_model("attnres_block", model_cfg={...})
    param_counts = count_params(model)
"""

from enum import Enum
from typing import Dict

import torch.nn as nn

from .network import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Model type enum
# ---------------------------------------------------------------------------


class ModelType(str, Enum):
    VANILLA       = "vanilla"
    ATTNRES_BLOCK = "attnres_block"
    ATTNRES_FULL  = "attnres_full"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(model_type: str, model_cfg: dict) -> GPT:
    """Build and return a GPT model for the given model type.

    Args:
        model_type: One of "vanilla", "attnres_block", "attnres_full".
        model_cfg:  Dict with keys from the YAML config (see module docstring).

    Returns:
        Initialised (not compiled) GPT model.
    """
    mtype = ModelType(model_type)

    use_attnres     = mtype in (ModelType.ATTNRES_BLOCK, ModelType.ATTNRES_FULL)
    attnres_variant = "full" if mtype == ModelType.ATTNRES_FULL else "block"

    cfg = GPTConfig(
        vocab_size         = int(model_cfg["vocab_size"]),
        block_size         = int(model_cfg["max_seq_len"]),
        n_layer            = int(model_cfg["n_layers"]),
        n_head             = int(model_cfg["n_heads"]),
        n_embd             = int(model_cfg["d_model"]),
        dropout            = float(model_cfg.get("dropout", 0.0)),
        bias               = bool(model_cfg.get("bias", False)),
        use_attnres        = use_attnres,
        attnres_variant    = attnres_variant,
        num_attnres_blocks = int(model_cfg.get("n_blocks", 8)),
    )

    return GPT(cfg)


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


def count_params(model: nn.Module) -> Dict[str, int]:
    """Return trainable parameter breakdown by component.

    Returns a dict with keys:
        "embedding (wte + wpe)", "attention sub-layers", "ffn sub-layers",
        "attnres pseudo-queries", "output norm", "total"
    """
    # Unwrap DDP wrapper if present
    raw: GPT = model.module if hasattr(model, "module") else model

    def _n(mod: nn.Module) -> int:
        return sum(p.numel() for p in mod.parameters())

    embedding   = _n(raw.transformer.wte) + _n(raw.transformer.wpe)
    output_norm = _n(raw.transformer.ln_f)

    attn_total = 0
    ffn_total  = 0
    for block in raw.transformer.h:
        attn_total += _n(block.ln_1) + _n(block.attn)
        ffn_total  += _n(block.ln_2) + _n(block.mlp)

    attnres_queries = _n(raw.attnres) if raw.attnres is not None else 0

    # lm_head shares weights with wte → total counts it once
    total = sum(p.numel() for p in raw.parameters())

    return {
        "embedding (wte + wpe)":   embedding,
        "attention sub-layers":    attn_total,
        "ffn sub-layers":          ffn_total,
        "attnres pseudo-queries":  attnres_queries,
        "output norm":             output_norm,
        "total":                   total,
    }
