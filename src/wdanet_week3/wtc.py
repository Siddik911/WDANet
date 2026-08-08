from __future__ import annotations

import torch


def pairwise_cost_matrix(xs: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    # Eq. M_{i,j}=||pos_i-pos_j||^2 approximated in latent space.
    return torch.cdist(xs, xt, p=2).pow(2)


def sinkhorn_wtc(
    M: torch.Tensor,
    lam: float = 10.0,
    eps: float = 1e-3,
    max_iter: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Algorithm 1 style WTC.

    K = exp(-lam*M)
    T = diag(u)Kdiag(v)
    T_rev = diag(v)K^Tdiag(u)
    """
    device = M.device
    ns, nt = M.shape
    K = torch.exp(-lam * M).clamp_min(1e-12)

    u = torch.ones(ns, device=device) / ns
    v = torch.ones(nt, device=device) / nt

    T = torch.diag(u) @ K @ torch.diag(v)
    T_rev = torch.diag(v) @ K.t() @ torch.diag(u)

    for _ in range(max_iter):
        Kv = K @ v + 1e-12
        KTu = K.t() @ u + 1e-12
        u_next = (torch.ones(ns, device=device) / ns) / Kv
        v_next = (torch.ones(nt, device=device) / nt) / KTu

        T_next = torch.diag(u_next) @ K @ torch.diag(v_next)
        T_rev_next = torch.diag(v_next) @ K.t() @ torch.diag(u_next)

        d1 = torch.max(torch.abs(T_next - T))
        d2 = torch.max(torch.abs(T_rev_next - T_rev))

        u, v, T, T_rev = u_next, v_next, T_next, T_rev_next
        if torch.maximum(d1, d2) < eps:
            break

    return T, T_rev, u, v


def transport_matrices(
    f_src: torch.Tensor,
    f_tgt: torch.Tensor,
    lam: float = 10.0,
    eps: float = 1e-3,
    max_iter: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M = pairwise_cost_matrix(f_src, f_tgt)
    T, T_rev, _, _ = sinkhorn_wtc(M, lam=lam, eps=eps, max_iter=max_iter)
    return T, T_rev, M
