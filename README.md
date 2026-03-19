# 🚀 Attention-Residual-Scaling

[![Paper](https://img.shields.io/badge/arXiv-2603.15031-B31B1B.svg)](https://arxiv.org/pdf/2603.15031)
[![Python-Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

This project is dedicated to the **Reproduction** and **Validation** of the core architectural insights proposed by [Kimi] in the technical report: *[Attention Residuals](https://arxiv.org/pdf/2603.15031)*.

Our goal is to verify the **Scaling Laws** performance of the **Attention-Residual** architecture across various data scales and model sizes through rigorous controlled experiments.

---

## 📖 Core Objectives

This project explores the performance of the `Attention-Residual` architecture across three key dimensions:

1.  **Scaling Law Curve Fitting**: Validating the power-law relationship between training loss ($L$) and compute budget ($C$).
2.  **Architectural Benchmarking**: Comparing the convergence efficiency of `Attention-Residual` vs. `Vanilla Transformer` at equivalent parameter counts.
3.  **Multi-modal Generalization**: Beyond Language Modeling (LLM), we investigate the architecture's spatial/geometric perception potential in downstream tasks like Monocular Depth Estimation.

---

## 📂 Project Structure

The repository follows a modular design to ensure experiment traceability and reproducibility:

```text
AttnRes_Scaled/
├── 📁 configs/          # YAML configurations for Tiny, Small, and Large scales
├── 📁 dataset/          # Data pipeline supporting high-performance .bin loading
│   ├── 📄 text_dataset.py
│   └── 📄 __init__.py
├── 📁 models/           # Model implementations (Vanilla vs. Attention-Residual)
├── 📁 runs/             # Artifacts: Checkpoints, training logs, and eval reports
├── 📄 train.py          # Distributed training engine based on `torchrun`
├── 📄 eval.py           # Evaluation tool for Head-to-Head performance comparison
├── 📄 logger.py         # Experiment tracking and logging utilities
├── 📄 requirements.txt   # Environment dependencies
├── 📄 run_atten.sh      # One-click script for Attention-Residual experiments
└── 📄 run_vanilla.sh    # One-click script for Baseline (Vanilla) experiments
```

## ⚡ Run Scripts

We provide unified entry points for both standard Transformer (Vanilla) and Attention-Residual architectures. 

### 1. Single-GPU Training (Rapid Prototyping)
Use these commands to verify code logic or run small-scale experiments on a single device:

```bash
# Run Vanilla Transformer Baseline
CUDA_VISIBLE_DEVICES=7 python train.py --config configs/tiny.yaml --model_type vanilla

# Run Attention-Residual Architecture
CUDA_VISIBLE_DEVICES=7 python train.py --config configs/tiny.yaml --model_type attnres_block

# Distributed training on 8 GPUs
torchrun --nproc_per_node=8 train.py \
    --config configs/tiny.yaml \
    --model_type attnres_block
```

## 🚀 Next Steps: Vision & Multimodal Validation

We aim to extend the **Attention-Residual Scaling Law** analysis beyond text:
* **Vision-Only**: Benchmarking on ImageNet.
* **Multimodal**: Testing on VLA (Vision-Language-Action) datasets to evaluate cross-modal alignment efficiency.

