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
) -> torch.Tensor:
    losses = []
    for c in range(n_classes):
        mask = class_ids == c
        if mask.any():
            logits = local_bank.forward_for_class(c, features[mask], alpha=alpha)
            losses.append(F.binary_cross_entropy_with_logits(logits, domain_labels[mask].float()))
    if not losses:
        return torch.tensor(0.0, device=features.device)
    return torch.stack(losses).mean()


def dynamic_adversarial_factor(lg: torch.Tensor, ll: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # proxy from paper-style global/local discrepancy balancing
    da_g = 2.0 * (1.0 - 2.0 * lg.detach())
    da_l = 2.0 * (1.0 - 2.0 * ll.detach())
    num = da_g
    den = da_g + da_l + eps
    w = num / den
    return torch.clamp(w, 0.0, 1.0)
