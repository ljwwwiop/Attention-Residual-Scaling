"""Datasets package: HuggingFace text corpora, tokenization, and DataLoader builders."""

from .text_dataset import DatasetScale, build_dataloader, get_dataset_info

__all__ = ["DatasetScale", "build_dataloader", "get_dataset_info"]
