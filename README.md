# [CVPR 2026] Parallel Jacobi Decoding for Fast Autoregressive Image Generation

[![arXiv](https://img.shields.io/badge/arXiv-2606.05703-b31b1b.svg)](https://arxiv.org/abs/2606.05703)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://boyaliao.github.io/PJD/)

[Boya Liao](https://boyaliao.github.io/), [Ying Li](https://neuraliying.github.io/), [Siyong Jian](https://syjmelody.github.io/), [Huan Wang](https://huanwang.tech/)<sup>*</sup>

Westlake University

<sup>*</sup>Corresponding author

---

## Abstract

> Autoregressive (AR) models have demonstrated remarkable performance in generating high-fidelity images. However, their inherently sequential next-token prediction leads to significantly slower inference. Recent studies have introduced Jacobi-style decoding to accelerate autoregressive image generation. Extending the draft sequence initially improves efficiency, yet the acceleration quickly saturates as error propagation in the one-dimensional sequence hinders convergence. Observing that images exhibit strong local spatial correlations, we propose Parallel Jacobi Decoding (PJD), a training-free decoding approach that expands draft tokens in the two-dimensional spatial domain to enable efficient spatially parallel refinement. PJD adjusts the attention mask to mitigate error accumulation and improve convergence stability. Extensive experiments on diverse datasets show that PJD achieves 4.8×–6.4× acceleration across multiple autoregressive image generation models while maintaining competitive generation quality.

## Installation

```
cd PJD
pip install -e .
```

## Usage

## 1. Download the Image Tokenizer

This project uses the image tokenizer from Meta's Chameleon model. Download the required tokenizer files from the [Meta Chameleon repository](https://github.com/facebookresearch/chameleon).

Place the downloaded files in the following directory:

```text
ckpts/chameleon/tokenizer/
```

The final directory structure should be:

```text
ckpts/
└── chameleon/
    └── tokenizer/
        ├── checklist.chk
        ├── text_tokenizer.json
        ├── vqgan.ckpt
        └── vqgan.yaml
```

Note: Make sure all four files are present before running inference.

## Acknowledgments
This implementation is built upon the official repository for [SJD](https://github.com/tyshiwo1/Accelerating-T2I-AR-with-SJD/).

## Citation

```bibtex
@inproceedings{liao2026parallel,
  title={Parallel Jacobi Decoding for Fast Autoregressive Image Generation},
  author={Liao, Boya and Li, Ying and Jian, Siyong and Wang, Huan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
