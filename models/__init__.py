"""Models package: factory for building AttnRes and Vanilla transformers."""

from .build import ModelType, build_model, count_params

__all__ = ["ModelType", "build_model", "count_params"]
