from __future__ import annotations

import torch
import torch.nn.functional as F


def label_classification_loss(logits_src: torch.Tensor, y_src: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits_src, y_src)


def global_domain_loss(domain_logits: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(domain_logits, domain_labels.float())


def local_domain_loss(
    local_bank,
    features: torch.Tensor,
    class_ids: torch.Tensor,
    domain_labels: torch.Tensor,
    alpha: float = 1.0,
    n_classes: int = 2,
    return_per_class: bool = False,
):
    losses = []
    per_class = []
    for c in range(n_classes):
        mask = class_ids == c
        if mask.any():
            logits = local_bank.forward_for_class(c, features[mask], alpha=alpha)
            lc = F.binary_cross_entropy_with_logits(logits, domain_labels[mask].float())
            losses.append(lc)
            per_class.append(lc)
        else:
            per_class.append(torch.tensor(0.0, device=features.device))
    if not losses:
        base = torch.tensor(0.0, device=features.device)
        return (base, per_class) if return_per_class else base
    mean_loss = torch.stack(losses).mean()
    return (mean_loss, per_class) if return_per_class else mean_loss


def dynamic_adversarial_factor(lg: torch.Tensor, ll_per_class: list[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """Eq (6)-(8)-style dynamic factor.

    d_A,g = 2(1-2L_g), d_A,l^c = 2(1-2L_l^c)
    omega = d_A,g / (d_A,g + (1/C) * sum_c d_A,l^c)
    """
    d_ag = 2.0 * (1.0 - 2.0 * lg.detach())
    llc = torch.stack([x.detach() for x in ll_per_class])
    d_al = 2.0 * (1.0 - 2.0 * llc)
    mean_dal = d_al.mean()
    omega = d_ag / (d_ag + mean_dal + eps)
    return torch.clamp(omega, 0.0, 1.0)
