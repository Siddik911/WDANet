#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from wdanet_week2 import (
    FeatureExtractor,
    LabelClassifier,
    GlobalDomainDiscriminator,
    LocalDomainDiscriminatorBank,
    label_classification_loss,
    global_domain_loss,
    local_domain_loss,
    dynamic_adversarial_factor,
)


torch.manual_seed(7)
bs, in_dim, feat_dim, n_classes = 32, 95, 128, 2
x_src = torch.randn(bs, in_dim)
x_tgt = torch.randn(bs, in_dim)
y_src = torch.randint(0, n_classes, (bs,))

Gf = FeatureExtractor(in_dim=in_dim, feat_dim=feat_dim)
Gy = LabelClassifier(feat_dim=feat_dim, n_classes=n_classes)
Gd = GlobalDomainDiscriminator(feat_dim=feat_dim)
Gl = LocalDomainDiscriminatorBank(n_classes=n_classes, feat_dim=feat_dim)

f_src = Gf(x_src)
f_tgt = Gf(x_tgt)
logits_y = Gy(f_src)

f_all = torch.cat([f_src, f_tgt], dim=0)
d_all = torch.cat([torch.zeros(bs), torch.ones(bs)], dim=0)
class_all = torch.cat([y_src, torch.argmax(Gy(f_tgt).detach(), dim=1)], dim=0)

lgts_g = Gd(f_all, alpha=1.0)
ly = label_classification_loss(logits_y, y_src)
lg = global_domain_loss(lgts_g, d_all)
ll, llc = local_domain_loss(Gl, f_all, class_all, d_all, alpha=1.0, n_classes=n_classes, return_per_class=True)
w = dynamic_adversarial_factor(lg, llc)

print("f_src", tuple(f_src.shape))
print("logits_y", tuple(logits_y.shape))
print("loss_y", float(ly))
print("loss_g", float(lg))
print("loss_l", float(ll))
print("omega", float(w))
