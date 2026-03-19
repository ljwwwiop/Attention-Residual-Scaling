"""
network.py — Standard GPT-2 style language models for AttnRes scaling-law experiments.

Two model variants, differing ONLY in residual mechanism:

  GPT(use_attnres=False):           Vanilla GPT baseline
      Standard additive skip connections:
          x = x + attn(LN(x))
          x = x + mlp(LN(x))

  GPT(use_attnres=True, attnres_variant="block"):  Block AttnRes
      Block Attention Residuals replace additive skips.
      Memory: O(N·d) where N = num_attnres_blocks (paper recommends N=8).

  GPT(use_attnres=True, attnres_variant="full"):   Full AttnRes
      Full (O(L·d)) Attention Residuals.

Architecture (GPT-2 style):
  Token embedding + learned absolute positional embedding
  → n_layer × GPTBlock [CausalSelfAttention + MLP, pre-LayerNorm]
  → Final LayerNorm
  → LM head (weight-tied with token embedding)

CausalSelfAttention:
  - Standard multi-head attention (no GQA, no RoPE)
  - Flash SDPA with is_causal=True for causal masking (falls back to manual mask)

MLP:
  - Linear(d, 4d) → GELU → Linear(4d, d)

Weight init: GPT-2 style (normal σ=0.02, scaled residual projections).

YAML config key mapping (build.py translates these):
  d_model      → n_embd
  n_layers     → n_layer   (number of full blocks; each block = 1 attn + 1 MLP)
  n_heads      → n_head
  vocab_size   → vocab_size
  max_seq_len  → block_size
  dropout      → dropout
  n_blocks     → num_attnres_blocks  (AttnRes N, ignored for vanilla)
  [n_kv_heads, ffn_mult, rope_theta are not used in GPT-2 style architecture]
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# AttnRes utilities (inlined from attention-residuals/attnres/core/)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no learnable parameters).

    Used to normalise keys before computing AttnRes attention scores,
    preventing layers with large-magnitude outputs from dominating softmax.
    Reference: arXiv:2603.15031.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms


def _merge_attn_stats(
    o1: torch.Tensor, m1: torch.Tensor, lse1: torch.Tensor,
    o2: torch.Tensor, m2: torch.Tensor, lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online softmax merge of two partial attention results (Milakov & Gimelshein 2018).

    Combines (o1, m1, lse1) and (o2, m2, lse2) without re-computing the full
    softmax from scratch.  Used in the two-phase Block AttnRes inference strategy.
    """
    m_merged = torch.maximum(m1, m2)
    e1 = torch.exp(lse1 - m_merged)
    e2 = torch.exp(lse2 - m_merged)
    e_total = e1 + e2
    o_merged = (o1 * e1 + o2 * e2) / e_total
    lse_merged = m_merged + torch.log(e_total)
    return o_merged, m_merged, lse_merged


class BlockAttnRes(nn.Module):
    """Block Attention Residuals (Block AttnRes).

    Partitions the L layers into N blocks of S = L/N layers each.  Within each
    block, layer outputs are summed into a single block representation b_n.
    Across blocks, each layer attends over only N block-level representations
    plus (within the current block) the evolving partial sum.

    This reduces memory from O(L·d) to O(N·d) while recovering most of the
    gain of Full AttnRes (paper shows N≈8 suffices across model scales).

    Args:
        num_layers:  Total number of transformer sub-layers L.
        hidden_dim:  Hidden dimension d.
        num_blocks:  Number of blocks N (default 8, paper recommendation).
        eps:         RMSNorm epsilon.

    Reference: Section 3.2 and Section 4.2 of arXiv:2603.15031.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_blocks: int = 8,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Compute block sizes (last block absorbs remainder)
        base_size = num_layers // num_blocks
        remainder = num_layers % num_blocks
        self.block_sizes = [base_size] * num_blocks
        if remainder:
            self.block_sizes[-1] += remainder

        # Cumulative layer offsets for block membership lookup
        self._block_starts: List[int] = []
        offset = 0
        for s in self.block_sizes:
            self._block_starts.append(offset)
            offset += s

        # One pseudo-query per layer, zero-initialised.
        # Zero init ensures uniform attention at training start → equivalent to
        # standard residual at initialisation, ensuring training stability.
        self.queries = nn.Parameter(torch.zeros(num_layers, hidden_dim))
        self.key_norm = _RMSNorm(hidden_dim, eps=eps)

    # ------------------------------------------------------------------
    # Block membership helpers
    # ------------------------------------------------------------------

    def _block_of(self, layer_idx: int) -> int:
        for n, start in enumerate(self._block_starts):
            if layer_idx < start + self.block_sizes[n]:
                return n
        return self.num_blocks - 1

    # ------------------------------------------------------------------
    # Stateful forward (single-pass, used during training)
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Clear state.  Must be called before each new sequence."""
        self._blocks: List[torch.Tensor] = []
        self._partial_block: Optional[torch.Tensor] = None
        self._intra_idx: int = 0

    def set_embedding(self, embedding: torch.Tensor) -> None:
        """Set b_0 = token embedding.  Call after reset_state()."""
        self._blocks = [embedding]
        self._partial_block = None
        self._intra_idx = 0

    def push_layer_output(self, layer_out: torch.Tensor) -> None:
        """Accumulate a sub-layer output into the current block's partial sum.

        Call AFTER computing f_l(h_l) for layer l and BEFORE calling
        forward() for layer l+1.

        Args:
            layer_out: f_l(h_l), shape (B, T, d).
        """
        if self._partial_block is None:
            self._partial_block = layer_out
        else:
            self._partial_block = self._partial_block + layer_out

        self._intra_idx += 1

        # Check if the current block is complete
        actual_block_n = len(self._blocks) - 1  # 0-based transformer block being filled
        if actual_block_n < self.num_blocks:
            if self._intra_idx >= self.block_sizes[actual_block_n]:
                self._blocks.append(self._partial_block)
                self._partial_block = None
                self._intra_idx = 0

    def forward(self, layer_idx: int) -> torch.Tensor:
        """Compute the AttnRes input h_l for layer `layer_idx` (0-based).

        Must be called sequentially (l=0, 1, 2, ...) after set_embedding()
        and interleaved with push_layer_output() calls.

        Args:
            layer_idx: Which layer's input to compute.

        Returns:
            Tensor of shape (B, T, d).
        """
        i = self._intra_idx  # 0-based position within current block

        # Value sources: completed block representations b_0..b_{n-1}
        # plus the intra-block partial sum (if not the first layer in block)
        sources: List[torch.Tensor] = list(self._blocks)
        if i > 0 and self._partial_block is not None:
            sources.append(self._partial_block)

        if not sources:
            raise RuntimeError("No sources available.  Did you call set_embedding()?")

        # Stack sources: (B, T, num_sources, d)
        v_stack = torch.stack(sources, dim=2)
        k_stack = self.key_norm(v_stack)

        w = self.queries[layer_idx]                        # (d,)
        scores = (k_stack * w).sum(dim=-1)                 # (B, T, num_sources)
        alpha = torch.softmax(scores, dim=-1)              # (B, T, num_sources)
        h = (alpha.unsqueeze(-1) * v_stack).sum(dim=2)    # (B, T, d)

        return h


class FullAttnRes(nn.Module):
    """Full Attention Residuals (Full AttnRes).

    Replaces the standard residual accumulation

        h_l = h_{l-1} + f_{l-1}(h_{l-1})

    with softmax attention over ALL preceding layer outputs:

        h_l = sum_{i=0}^{l-1} alpha_{i→l} * v_i

    where
        v_0        = h_1  (token embedding)
        v_i (i>=1) = f_i(h_i)  (sub-layer output)
        alpha_{i→l} = softmax( w_l^T RMSNorm(v_i) )  over i=0..l-1

    Memory: O(L·d).  Use BlockAttnRes for large models.

    Args:
        num_layers:  Number of transformer sub-layers L.
        hidden_dim:  Hidden dimension d.
        eps:         RMSNorm epsilon.

    Reference: Section 3.1 of arXiv:2603.15031.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # One pseudo-query per layer, zero-initialised (same rationale as Block AttnRes)
        self.queries = nn.Parameter(torch.zeros(num_layers, hidden_dim))
        self.key_norm = _RMSNorm(hidden_dim, eps=eps)

    # ------------------------------------------------------------------
    # Stateful forward (single-pass, used during training)
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Clear state.  Must be called before each new sequence."""
        self._values: List[torch.Tensor] = []

    def set_embedding(self, embedding: torch.Tensor) -> None:
        """Set v_0 = token embedding.  Call after reset_state()."""
        self._values = [embedding]

    def push_layer_output(self, layer_out: torch.Tensor) -> None:
        """Store v_l = f_l(h_l) for future layers to attend over.

        Args:
            layer_out: f_l(h_l), shape (B, T, d).
        """
        self._values.append(layer_out)

    def forward(self, layer_idx: int) -> torch.Tensor:
        """Compute the AttnRes input h_l for layer `layer_idx` (0-based).

        Must be called sequentially (l=0, 1, 2, ...) after set_embedding()
        and interleaved with push_layer_output() calls.

        For layer 0 (the first sub-layer), returns the embedding directly
        since there are no preceding layer outputs to attend over.

        Args:
            layer_idx: Which layer's input to compute.

        Returns:
            Tensor of shape (B, T, d).
        """
        if layer_idx == 0:
            return self._values[0]   # h_0 = embedding

        # Attend over v_0 .. v_{layer_idx-1}  (self._values has exactly layer_idx entries
        # after push_layer_output has been called for layers 0..layer_idx-1)
        v_stack = torch.stack(self._values, dim=2)   # (B, T, l, d)
        k_stack = self.key_norm(v_stack)

        w = self.queries[layer_idx]                        # (d,)
        scores = (k_stack * w).sum(dim=-1)                 # (B, T, l)
        alpha = torch.softmax(scores, dim=-1)
        h = (alpha.unsqueeze(-1) * v_stack).sum(dim=2)    # (B, T, d)

        return h


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    # Core transformer dims
    vocab_size: int = 50257
    block_size: int = 1024       # maximum sequence length
    n_layer: int = 12            # number of transformer blocks
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False           # bias in Linear + LayerNorm (False = faster)

    # AttnRes options (ignored when use_attnres=False)
    use_attnres: bool = False
    attnres_variant: str = "block"   # "block" | "full"
    num_attnres_blocks: int = 8      # N for Block AttnRes (paper recommends 8)


# ---------------------------------------------------------------------------
# GPT sub-modules
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch LayerNorm always has bias)."""

    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Standard causal multi-head self-attention.

    Uses Flash SDPA (is_causal=True) when available (PyTorch ≥ 2.0),
    otherwise falls back to a manual lower-triangular mask.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, (
            f"n_embd ({cfg.n_embd}) must be divisible by n_head ({cfg.n_head})"
        )
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                    1, 1, cfg.block_size, cfg.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        head_dim = C // self.n_head

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward network: Linear(d, 4d) → GELU → Linear(4d, d)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class GPTBlock(nn.Module):
    """Single transformer block: pre-LayerNorm, CausalSelfAttention, MLP.

    The standard vanilla forward is:
        x = x + attn(LN1(x))
        x = x + mlp(LN2(x))

    Two helper methods expose the raw sub-layer outputs so that
    _forward_attnres() can provide AttnRes-computed inputs instead of x.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vanilla forward with standard additive residuals."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def attn_sublayer_out(self, h: torch.Tensor) -> torch.Tensor:
        """Return f_attn(LN1(h)) — attention sub-layer output before residual add."""
        return self.attn(self.ln_1(h))

    def mlp_sublayer_out(self, h: torch.Tensor) -> torch.Tensor:
        """Return f_mlp(LN2(h)) — MLP sub-layer output before residual add."""
        return self.mlp(self.ln_2(h))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class GPT(nn.Module):
    """GPT-style language model with optional Attention Residuals.

    Setting use_attnres=False gives the standard GPT-2 baseline.
    Setting use_attnres=True replaces additive skip connections with
    Block AttnRes (attnres_variant="block") or Full AttnRes ("full").

    The only extra parameters introduced by AttnRes are n_layer * 2 learned
    pseudo-query vectors (one per attention sub-layer and one per MLP
    sub-layer), which are initialised to zero so that training starts from
    an equal-weight average — equivalent to a standard residual at init.

    Forward signature matches the nanoGPT / training-framework convention:
        logits, loss = model(idx, targets)   # targets optional
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layer)]),
            ln_f=LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: token embedding and LM head share parameters
        self.transformer.wte.weight = self.lm_head.weight

        # AttnRes module (None for vanilla)
        if cfg.use_attnres:
            n_sublayers = cfg.n_layer * 2   # attn sub-layer + mlp sub-layer per block
            if cfg.attnres_variant == "block":
                self.attnres = BlockAttnRes(
                    num_layers=n_sublayers,
                    hidden_dim=cfg.n_embd,
                    num_blocks=cfg.num_attnres_blocks,
                )
            elif cfg.attnres_variant == "full":
                self.attnres = FullAttnRes(
                    num_layers=n_sublayers,
                    hidden_dim=cfg.n_embd,
                )
            else:
                raise ValueError(
                    f"Unknown attnres_variant '{cfg.attnres_variant}'. "
                    "Expected 'block' or 'full'."
                )
        else:
            self.attnres = None

        # Weight initialisation (GPT-2 style)
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layer) (GPT-2 paper section 2.3)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Parameter count helper
    # ------------------------------------------------------------------

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return parameter count.

        With non_embedding=True (default), position embeddings are excluded
        (they contribute no compute to the non-embedding layers).
        Token embeddings are always included because they are shared with the
        LM head via weight tying.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"Sequence length {T} exceeds block_size {self.cfg.block_size}"
        )

        pos = torch.arange(T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)           # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)           # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, n_embd)

        if self.attnres is not None:
            x = self._forward_attnres(x)
        else:
            for block in self.transformer.h:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ------------------------------------------------------------------
    # AttnRes forward (single-pass, stateful)
    # ------------------------------------------------------------------

    def _forward_attnres(self, embedding: torch.Tensor) -> torch.Tensor:
        """Single-pass forward using the stateful AttnRes API.

        Each layer receives its AttnRes-computed input h_l based on the TRUE
        outputs of all preceding layers — not an approximation from a
        standard-residual pre-pass.

        Algorithm (sequential, one pass through the network):
            ar.reset_state()
            ar.set_embedding(embedding)           # b_0 = token embedding

            for each transformer block i:
                h_attn = ar.forward(i*2)          # attn layer: AttnRes input
                f_attn = attn_sublayer(h_attn)
                ar.push_layer_output(f_attn)      # update block partial sum
                x      = h_attn + f_attn          # residual merge

                h_mlp  = ar.forward(i*2+1)        # MLP layer: AttnRes input
                f_mlp  = mlp_sublayer(h_mlp)
                ar.push_layer_output(f_mlp)
                x      = h_mlp + f_mlp

        Properties:
          - 1× transformer compute (no pre-pass, no double execution)
          - Mathematically exact AttnRes (uses true f_l(h_l) to build block reps)
          - Fully differentiable through all ar.forward() / push_layer_output() calls
          - Compatible with DDP (state is per-process, reset each forward call)
          - Not compatible with torch.compile() due to Python-level state mutations
        """
        ar = self.attnres
        ar.reset_state()
        ar.set_embedding(embedding)

        x = embedding
        for i, block in enumerate(self.transformer.h):
            # ---- Attention sub-layer (layer index i*2) ----
            h_attn = ar.forward(i * 2)
            attn_o = block.attn_sublayer_out(h_attn)
            ar.push_layer_output(attn_o)
            x = h_attn + attn_o

            # ---- MLP sub-layer (layer index i*2 + 1) ----
            h_mlp = ar.forward(i * 2 + 1)
            mlp_o = block.mlp_sublayer_out(h_mlp)
            ar.push_layer_output(mlp_o)
            x = h_mlp + mlp_o

        return x

    # ------------------------------------------------------------------
    # Generation (autoregressive, temperature + top-k)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively sample max_new_tokens tokens.

        Args:
            idx:            Conditioning token indices, shape (B, T).
            max_new_tokens: Number of tokens to generate.
            temperature:    Softmax temperature (< 1 = sharper, > 1 = flatter).
            top_k:          If set, restrict sampling to top-k logits.

        Returns:
            idx tensor extended by max_new_tokens columns.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Preset configs for reference (matching the YAML configs in configs/)
# ---------------------------------------------------------------------------

MODEL_PRESETS = {
    # ~22 M total params | configs/tiny.yaml
    "tiny":         dict(n_layer=12, n_head=8,  n_embd=256,  block_size=1024),
    # ~97 M total params | configs/small.yaml
    "small":        dict(n_layer=24, n_head=16, n_embd=512,  block_size=1024),
    # ~354 M total params | configs/medium.yaml        (≈ GPT-2 medium)
    "medium":       dict(n_layer=24, n_head=16, n_embd=1024, block_size=1024),
    # ~774 M total params | configs/medium-large.yaml  (≈ GPT-2 large)
    "medium-large": dict(n_layer=36, n_head=20, n_embd=1280, block_size=1024),
    # ~1.56 B total params | configs/medium-xl.yaml    (≈ GPT-2 XL)
    "medium-xl":    dict(n_layer=48, n_head=25, n_embd=1600, block_size=1024),
}
