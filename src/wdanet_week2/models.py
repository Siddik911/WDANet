from __future__ import annotations

import torch
from torch import nn

from .grl import grad_reverse


def _mlp(in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float = 0.5) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class FeatureExtractor(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int = 128, hidden_dims: list[int] | None = None, dropout: float = 0.5):
        super().__init__()
        hidden = hidden_dims or [256, 256, 128]
        self.net = _mlp(in_dim, hidden, feat_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LabelClassifier(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int = 2, hidden_dims: list[int] | None = None, dropout: float = 0.5):
        super().__init__()
        self.net = _mlp(feat_dim, hidden_dims or [128], n_classes, dropout)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


class GlobalDomainDiscriminator(nn.Module):
    def __init__(self, feat_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.5):
        super().__init__()
        self.net = _mlp(feat_dim, hidden_dims or [128, 64], 1, dropout)

    def forward(self, f: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(f, alpha)).squeeze(-1)


class LocalDomainDiscriminator(nn.Module):
    def __init__(self, feat_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.5):
        super().__init__()
        self.net = _mlp(feat_dim, hidden_dims or [128, 64], 1, dropout)

    def forward(self, f: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(f, alpha)).squeeze(-1)


class LocalDomainDiscriminatorBank(nn.Module):
    def __init__(self, n_classes: int, feat_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.5):
        super().__init__()
        self.n_classes = n_classes
        self.discriminators = nn.ModuleList(
            [LocalDomainDiscriminator(feat_dim, hidden_dims=hidden_dims, dropout=dropout) for _ in range(n_classes)]
        )

    def forward_for_class(self, class_id: int, f: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return self.discriminators[class_id](f, alpha=alpha)
