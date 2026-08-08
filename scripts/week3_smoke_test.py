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
from wdanet_week3 import compute_wdanet_objective


torch.manual_seed(11)
bs, in_dim, feat_dim, n_classes = 24, 95, 128, 2
x_src = torch.randn(bs, in_dim)
x_tgt = torch.randn(bs, in_dim)
y_src = torch.randint(0, n_classes, (bs,))

Gf = FeatureExtractor(in_dim=in_dim, feat_dim=feat_dim)
Gy = LabelClassifier(feat_dim=feat_dim, n_classes=n_classes)
Gd = GlobalDomainDiscriminator(feat_dim=feat_dim)
Gl = LocalDomainDiscriminatorBank(n_classes=n_classes, feat_dim=feat_dim)

f_src = Gf(x_src)
f_tgt = Gf(x_tgt)
logits_src = Gy(f_src)
logits_tgt = Gy(f_tgt)

f_all = torch.cat([f_src, f_tgt], dim=0)
d_all = torch.cat([torch.zeros(bs), torch.ones(bs)], dim=0)
classes_all = torch.cat([y_src, torch.argmax(logits_tgt.detach(), dim=1)], dim=0)

ly = label_classification_loss(logits_src, y_src)
lg = global_domain_loss(Gd(f_all, alpha=1.0), d_all)
ll, llc = local_domain_loss(Gl, f_all, classes_all, d_all, alpha=1.0, n_classes=n_classes, return_per_class=True)
omega = dynamic_adversarial_factor(lg, llc)

outs = compute_wdanet_objective(f_src, f_tgt, ly=ly, lg=lg, ll=ll, omega=omega, lam=5.0, eps=1e-4, max_iter=50)

print("L_total", float(outs["L_total"].detach()))
print("L_w", float(outs["L_w"].detach()))
print("L_forward", float(outs["L_forward"].detach()))
print("L_reverse", float(outs["L_reverse"].detach()))
print("T", tuple(outs["T"].shape), "T_rev", tuple(outs["T_rev"].shape))
