# Scaling Optimization-Oriented Hypernetworks for Implicit Neural Representations

[Paper (OpenReview)](https://openreview.net/forum?id=a2C8sbYUCs) · NeurIPS 2026

## Overview

We study optimization-oriented hypernetworks for generating implicit neural representations (INRs). The architecture uses output-neuron weight tokens and attention-based, chain-rule-inspired forward and backward message passing to predict parameter updates.

This is an initial partial code release containing reusable hypernetwork components and a minimal runnable example. The complete model, training and evaluation pipelines, and pretrained checkpoints will follow.

## TODO

- [x] Release core attention, weight tokenization, and optimization-oriented hypernetwork components.
- [ ] Release the complete `Hypernet` controller and target INR implementations.
- [ ] Add data preparation instructions and configurations for CelebA-HQ, ERA5, ShapeNet, ImageNet, and NeRF experiments.
- [ ] Release training, inference, and evaluation scripts.
- [ ] Publish pretrained checkpoints and their configurations.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{cai2026scaling,
  title     = {Scaling Optimization-Oriented Hypernetworks for Implicit Neural Representations},
  author    = {Lulu Cai and Hanxiang Ren and Siyan Dong and Li Sun and Jiefeng Wu and Youyi Zheng and Yi Ma and Yanchao Yang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=a2C8sbYUCs}
}
```
