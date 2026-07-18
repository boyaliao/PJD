# [CVPR 2026] Parallel Jacobi Decoding for Fast Autoregressive Image Generation

[![arXiv](https://img.shields.io/badge/arXiv-2510.18269-b31b1b.svg)](https://arxiv.org/abs/2606.05703)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://boyaliao.github.io/PJD/)

[Boya Liao](https://boyaliao.github.io/), [Ying Li](https://neuraliying.github.io/), [Siyong Jian](https://syjmelody.github.io/), [Huan Wang](https://huanwang.tech/)<sup>*</sup>

Westlake University
<sup>*</sup>Corresponding author

---

## Abstract

> Autoregressive (AR) models have demonstrated remarkable performance in generating high-fidelity images. However, their inherently sequential next-token prediction leads to significantly slower inference. Recent studies have introduced Jacobi-style decoding to accelerate autoregressive image generation. Extending the draft sequence initially improves efficiency, yet the acceleration quickly saturates as error propagation in the one-dimensional sequence hinders convergence. Observing that images exhibit strong local spatial correlations, we propose Parallel Jacobi Decoding (PJD), a training-free decoding approach that expands draft tokens in the two-dimensional spatial domain to enable efficient spatially parallel refinement. PJD adjusts the attention mask to mitigate error accumulation and improve convergence stability. Extensive experiments on diverse datasets show that PJD achieves 4.8×–6.4× acceleration across multiple autoregressive image generation models while maintaining competitive generation quality.
