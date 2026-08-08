from __future__ import annotations

import torch

from .wtc import transport_matrices
from .wasserstein import wasserstein_distribution_loss


def compute_wdanet_objective(
    f_src: torch.Tensor,
    f_tgt: torch.Tensor,
    ly: torch.Tensor,
    lg: torch.Tensor,
    ll: torch.Tensor,
    omega: torch.Tensor,
    lam: float = 10.0,
    eps: float = 1e-3,
    max_iter: int = 200,
) -> dict[str, torch.Tensor]:
    T, T_rev, M = transport_matrices(f_src, f_tgt, lam=lam, eps=eps, max_iter=max_iter)
    lw, lfwd, lrev = wasserstein_distribution_loss(T, T_rev)
    # Eq (21): L_total = L_y - ( omega L_g + (1-omega)L_l + L_w )
    l_total = ly - (omega * lg + (1.0 - omega) * ll + lw)
    return {
        "L_total": l_total,
        "L_w": lw,
        "L_forward": lfwd,
        "L_reverse": lrev,
        "T": T,
        "T_rev": T_rev,
        "M": M,
    }
