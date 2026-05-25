from __future__ import annotations

import torch
import torch.nn.functional as F


def wasserstein_distribution_loss(T: torch.Tensor, T_rev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Eq (10), (11): average BCE terms on forward/reverse transport matrices.
    l_fwd = F.binary_cross_entropy(T.clamp(1e-8, 1 - 1e-8), torch.ones_like(T))
    l_rev = F.binary_cross_entropy(T_rev.clamp(1e-8, 1 - 1e-8), torch.zeros_like(T_rev))
    l_w = l_fwd + l_rev
    return l_w, l_fwd, l_rev
