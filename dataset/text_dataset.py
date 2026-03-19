"""
Text Dataset Module
===================
Provides three corpora at different scales for language model scaling-law
experiments.

Dataset scales:
  small  → wikitext / wikitext-2-raw-v1      (~2 M train tokens)
             loaded via HuggingFace datasets + tiktoken tokenisation + .npy cache
  medium → wikitext / wikitext-103-raw-v1    (~103 M train tokens)
             same pipeline as small
  large  → openwebtext pre-tokenised binary   (~9 B train tokens)
             loaded DIRECTLY from pre-built nanoGPT-format .bin files:
               train.bin  17 GB  uint16  9 035 582 489 tokens
               val.bin     8 MB  uint16      4 434 606 tokens
             Uses np.memmap — the full file is NEVER copied into RAM.
             Each __getitem__ reads only a single (max_seq_len+1)-token chunk.

Tokenizer (small / medium only):
  GPT-2 BPE via `tiktoken` (vocab size 50 257).

Packing strategy (all scales):
  Non-overlapping windows of length max_seq_len + 1.
  Item: (x = window[:-1], y = window[1:])

Multi-GPU:
  Pass rank and world_size to build_dataloader → DistributedSampler.
"""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-built openwebtext binary files (nanoGPT format)
# ---------------------------------------------------------------------------
# These files were prepared by nanoGPT-moe/data/openwebtext/prepare.py:
#   enc.encode_ordinary(text) + eot_token(50256), saved as flat uint16 arrays.
# train.bin : 9 035 582 489 tokens  (17 GB)
# val.bin   :     4 434 606 tokens  (8.5 MB)
_OWT_BIN_DIR = Path(
    "/lpai/volumes/mind-vla-ali-sh-mix/lianjiawei/foundation_model"
    "/nanoGPT-moe/data/openwebtext"
)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


class DatasetScale(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# HuggingFace dataset specifications per scale
_DATASET_REGISTRY = {
    DatasetScale.SMALL: {
        "hf_path": "/lpai/volumes/mind-vla-ali-sh-mix/lianjiawei/foundation_model/attention-residuals/wikitext",
        "hf_name": "wikitext-2-raw-v1",
        "splits": {"train": "train", "val": "validation", "test": "test"},
        "text_field": "text",
        "approx_train_tokens": 2_100_000,
        "max_docs": None,
    },
    DatasetScale.MEDIUM: {
        "hf_path": "/lpai/volumes/mind-vla-ali-sh-mix/lianjiawei/foundation_model/attention-residuals/wikitext",
        "hf_name": "wikitext-103-raw-v1",
        "splits": {"train": "train", "val": "validation", "test": "test"},
        "text_field": "text",
        "approx_train_tokens": 103_000_000,
        "max_docs": None,
    },
    DatasetScale.LARGE: {
        # ── Use pre-built nanoGPT binary files directly (no HF loading) ──
        # bin_dir signals build_dataloader to use MemMapTokenDataset instead
        # of the HF tokenisation pipeline.
        "bin_dir": _OWT_BIN_DIR,
        # Map logical split names to filenames inside bin_dir.
        # openwebtext has no separate test split → reuse val.bin.
        "bin_files": {"train": "train.bin", "val": "val.bin", "test": "val.bin"},
        "approx_train_tokens": 9_035_582_489,
        # max_docs is irrelevant for the binary path; kept for config compat.
        "max_docs": None,
    },
}


def get_dataset_info(scale: str) -> dict:
    """Return registry metadata for a given scale name."""
    return dict(_DATASET_REGISTRY[DatasetScale(scale)])


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def _get_tokenizer():
    """Return a (encode, vocab_size) pair using tiktoken if available."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        # Use <|endoftext|> as document separator (token id 50256)
        sep_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})

        def encode(text: str):
            return enc.encode_ordinary(text)

        return encode, sep_token, enc.n_vocab  # 50257
    except ImportError:
        pass

    # Fallback: HuggingFace tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        sep_token = [tok.eos_token_id]

        def encode(text: str):
            return tok.encode(text, add_special_tokens=False)

        return encode, sep_token, tok.vocab_size

    except ImportError as e:
        raise ImportError(
            "Neither tiktoken nor transformers is available. "
            "Install one: pip install tiktoken  OR  pip install transformers"
        ) from e


# ---------------------------------------------------------------------------
# Tokenisation + packing (with disk cache)
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, scale: str, split: str) -> Path:
    return cache_dir / f"{scale}_{split}.npy"


def _tokenise_and_pack(
    scale: str,
    split: str,
    cache_dir: Path,
    max_docs: Optional[int] = None,
    ) -> np.ndarray:
    """Download, tokenise, and concatenate a dataset split.

    Returns a 1-D numpy array of uint16 token IDs.
    The array is also saved to `cache_dir` for fast reuse.
    """
    encode, sep_token, vocab_size = _get_tokenizer()
    assert vocab_size <= 65535, "Token IDs must fit in uint16"

    info = _DATASET_REGISTRY[DatasetScale(scale)]
    actual_max_docs = max_docs if max_docs is not None else info["max_docs"]

    # ---- Load HuggingFace dataset ----------------------------------------
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    hf_split = info["splits"].get(split)

    # openwebtext has no explicit val/test splits — derive from train
    if hf_split is None and split in ("val", "test"):
        logger.warning(
            f"{scale}/{split} is not available; deriving 5 %% val from train."
        )
        hf_split = "train"

    hf_path = info["hf_path"]
    hf_name = info.get("hf_name", "")

    # ---- Detect local parquet layout: {hf_path}/{hf_name}/*.parquet --------
    # When the dataset is stored locally as parquet shards (e.g. a HF snapshot
    # with sub-directories per config), load directly from disk without any
    # network calls.  Fall back to the HF Hub API for remote identifiers.
    local_data_dir = Path(hf_path) / hf_name if hf_name else Path(hf_path)
    if Path(hf_path).is_dir() and local_data_dir.is_dir():
        hf_split_names = set(info["splits"].values())
        data_files = {
            s: str(local_data_dir / f"{s}-*.parquet")
            for s in hf_split_names
            if list(local_data_dir.glob(f"{s}-*.parquet"))
        }
        logger.info(
            f"Loading local parquet dataset: {local_data_dir} "
            f"[split={hf_split}] files={list(data_files.keys())} ..."
        )
        ds = load_dataset("parquet", data_files=data_files, split=hf_split, streaming=False)
    else:
        load_kwargs = dict(path=hf_path, trust_remote_code=True)
        if hf_name:
            load_kwargs["name"] = hf_name
        logger.info(
            f"Loading HuggingFace dataset: {hf_path} ({hf_name}) [{hf_split}] ..."
        )
        ds = load_dataset(**load_kwargs, split=hf_split, streaming=False)

    # For datasets without an explicit val split, create one deterministically
    if info["splits"].get(split) is None:
        total = len(ds)
        val_size = max(1, total // 20)  # 5 % validation
        if split == "val":
            ds = ds.select(range(0, val_size))
        else:  # split == "test"
            ds = ds.select(range(0, val_size))  # reuse val as test if needed
        # For the train split we exclude the validation portion
        # Note: this is only applied when hf_split == "train" for openwebtext

    # Cap document count
    if actual_max_docs is not None and len(ds) > actual_max_docs:
        logger.info(f"Capping dataset to {actual_max_docs:,} documents (from {len(ds):,}).")
        ds = ds.select(range(actual_max_docs))

    text_field = info["text_field"]
    logger.info(f"Tokenising {len(ds):,} documents …")

    all_tokens = []
    for i, example in enumerate(ds):
        text = example[text_field].strip()
        if not text:
            continue
        ids = encode(text)
        all_tokens.extend(ids)
        all_tokens.extend(sep_token)

        if (i + 1) % 10_000 == 0:
            logger.info(f"  tokenised {i + 1:,} / {len(ds):,} docs, {len(all_tokens):,} tokens")

    token_array = np.array(all_tokens, dtype=np.uint16)
    logger.info(f"Total tokens: {len(token_array):,}")

    # ---- Cache to disk ---------------------------------------------------
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = _cache_path(cache_dir, scale, split)
    np.save(out_path, token_array)
    logger.info(f"Cache saved → {out_path}")

    return token_array


def _load_tokens(
    scale: str,
    split: str,
    cache_dir: Path,
    max_docs: Optional[int] = None,
) -> np.ndarray:
    """Load tokens from cache if available, else tokenise and cache."""
    path = _cache_path(cache_dir, scale, split)
    if path.exists():
        logger.info(f"Loading token cache: {path} …")
        return np.load(path)
    return _tokenise_and_pack(scale, split, cache_dir, max_docs)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class PackedTextDataset(Dataset):
    """Non-overlapping packed sequence dataset for small/medium HF corpora.

    The full token array is loaded into RAM as int64.  Only suitable for
    datasets that fit in memory (≲ a few GB).  For the large openwebtext
    binary use MemMapTokenDataset instead.

    Args:
        tokens:      1-D numpy array of uint16 token IDs (copied to int64).
        max_seq_len: Context length. Each chunk has `max_seq_len + 1` tokens.
    """

    def __init__(self, tokens: np.ndarray, max_seq_len: int) -> None:
        self.max_seq_len = max_seq_len
        chunk = max_seq_len + 1
        n = (len(tokens) // chunk) * chunk
        self.tokens = tokens[:n].astype(np.int64)  # copy OK: small/medium only
        self.chunk = chunk
        self.n_chunks = n // chunk

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.chunk
        chunk = torch.from_numpy(self.tokens[start : start + self.chunk])
        return chunk[:-1], chunk[1:]


class MemMapTokenDataset(Dataset):
    """Memory-mapped packed sequence dataset for the large openwebtext binary.

    The .bin file (up to 17 GB) is opened via np.memmap — the OS maps the
    file into the virtual address space but pages are read from disk only on
    demand.  The token array is NEVER copied into RAM in full.

    Each __getitem__ reads exactly (max_seq_len + 1) consecutive uint16
    values from the mmap, converts them to int64, and returns (x, y).
    Repeated random access is efficient thanks to the OS page cache.

    Args:
        bin_path:    Path to a flat uint16 binary file (nanoGPT format).
        max_seq_len: Context length. Each chunk has max_seq_len + 1 tokens.
    """

    def __init__(self, bin_path: Path, max_seq_len: int) -> None:
        bin_path = Path(bin_path)
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary token file not found: {bin_path}")

        self.bin_path = bin_path
        self.max_seq_len = max_seq_len
        self.chunk = max_seq_len + 1

        # Open read-only memmap — shape is determined from file size
        self._mmap = np.memmap(bin_path, dtype=np.uint16, mode="r")
        n_tokens = len(self._mmap)
        self.n_chunks = n_tokens // self.chunk  # discard trailing partial chunk

        logger.info(
            f"MemMapTokenDataset: {bin_path.name} | "
            f"{n_tokens:,} tokens | {self.n_chunks:,} chunks "
            f"(seq_len={max_seq_len})"
        )

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.chunk
        # Slice from memmap → copy into contiguous int64 array (only chunk_size
        # elements, ~8 KB for seq_len=1024, never the full 17 GB)
        chunk = self._mmap[start : start + self.chunk].astype(np.int64)
        t = torch.from_numpy(chunk)
        return t[:-1], t[1:]


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_dataloader(
    scale: str,
    split: str,
    max_seq_len: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    max_docs: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """Build a DataLoader for the given dataset scale and split.

    Routing logic:
      large  → MemMapTokenDataset (np.memmap on pre-built .bin files)
      small / medium → PackedTextDataset (local load_dataset + tiktoken BPE)

    Args:
        scale:       "small", "medium", or "large".
        split:       "train", "val", or "test".
        max_seq_len: Sequence length for language modelling.
        batch_size:  Per-GPU micro-batch size.
        rank:        Global DDP rank (0 for single-GPU).
        world_size:  Total DDP world size (1 for single-GPU).
        num_workers: DataLoader worker processes.
        cache_dir:   Unused (kept for API compatibility).
        max_docs:    Override the per-scale document cap (small/medium only).
        seed:        Random seed for DistributedSampler.

    Returns:
        (dataloader, sampler).  sampler is None for single-GPU runs.
        Call sampler.set_epoch(epoch) each epoch when sampler is not None.
    """
    info = _DATASET_REGISTRY[DatasetScale(scale)]

    # ── Branch A: large scale — load from pre-built .bin via memmap ─────────
    if "bin_dir" in info:
        bin_files = info["bin_files"]
        if split not in bin_files:
            raise ValueError(
                f"Unknown split '{split}' for scale '{scale}'. "
                f"Available: {list(bin_files.keys())}"
            )
        bin_path = info["bin_dir"] / bin_files[split]
        dataset: Dataset = MemMapTokenDataset(bin_path, max_seq_len)

    # ── Branch B: small / medium — load directly from local dataset path ─────
    # Mirrors make_wikitext_dataset in train_gpt_demo.py:
    #   load_dataset(path, name, split) → join text → ord(c) % 256 byte-level
    # No tiktoken / transformers / network calls needed. vocab_size = 256.
    else:
        from datasets import load_dataset as _hf_load  # pylint: disable=import-outside-toplevel

        hf_split_name = info["splits"].get(split, split)
        hf_name = info.get("hf_name") or None

        logger.info(
            f"Loading local dataset: {info['hf_path']} / {hf_name or ''} "
            f"[{hf_split_name}] ..."
        )
        ds = (
            _hf_load(info["hf_path"], hf_name, split=hf_split_name)
            if hf_name
            else _hf_load(info["hf_path"], split=hf_split_name)
        )

        if max_docs is not None and len(ds) > max_docs:
            logger.info(f"Capping to {max_docs:,} documents (from {len(ds):,}).")
            ds = ds.select(range(max_docs))

        # Byte-level encoding: ord(c) % 256 — no tokenizer, no network calls.
        # Matches make_wikitext_dataset in train_gpt_demo.py (vocab_size = 256).
        text_field = info["text_field"]
        all_ids: list = []
        for ex in ds:
            text = ex[text_field].strip()
            if not text:
                continue
            all_ids.extend(ord(c) % 256 for c in text)
            all_ids.append(10)  # '\n' as document separator

        tokens = np.array(all_ids, dtype=np.int64)
        logger.info(f"Tokenised {len(ds):,} docs → {len(tokens):,} tokens.")
        dataset = PackedTextDataset(tokens, max_seq_len)

    # ── DistributedSampler for multi-GPU ─────────────────────────────────────
    sampler: Optional[DistributedSampler] = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == "train"),
            seed=seed,
            drop_last=True,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(split == "train") if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    return loader, sampler

