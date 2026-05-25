from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch
from torch import nn
from torch.optim import Adam

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
from .integration import compute_wdanet_objective


@dataclass
class TrainConfig:
    in_dim: int = 95
    feat_dim: int = 128
    n_classes: int = 2
    batch_size: int = 48
    epochs: int = 20
    lr: float = 1e-3
    seed: int = 42
    sinkhorn_lambda: float = 5.0
    sinkhorn_eps: float = 1e-4
    sinkhorn_iters: int = 50
    eval_every: int = 10


class WDANetTrainBundle(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.gf = FeatureExtractor(cfg.in_dim, feat_dim=cfg.feat_dim)
        self.gy = LabelClassifier(cfg.feat_dim, n_classes=cfg.n_classes)
        self.gd = GlobalDomainDiscriminator(cfg.feat_dim)
        self.gl = LocalDomainDiscriminatorBank(cfg.n_classes, cfg.feat_dim)


def _synthetic_batch(bs: int, in_dim: int, n_classes: int, device: torch.device):
    x_src = torch.randn(bs, in_dim, device=device)
    x_tgt = torch.randn(bs, in_dim, device=device)
    y_src = torch.randint(0, n_classes, (bs,), device=device)
    return x_src, x_tgt, y_src


def run_single_experiment(cfg: TrainConfig, out_dir: Path, tag: str, ablation: str = "none") -> dict:
    torch.manual_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WDANetTrainBundle(cfg).to(device)
    opt = Adam(model.parameters(), lr=cfg.lr)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        x_src, x_tgt, y_src = _synthetic_batch(cfg.batch_size, cfg.in_dim, cfg.n_classes, device)

        f_src = model.gf(x_src)
        f_tgt = model.gf(x_tgt)
        logits_src = model.gy(f_src)
        logits_tgt = model.gy(f_tgt)

        f_all = torch.cat([f_src, f_tgt], dim=0)
        d_all = torch.cat([torch.zeros(cfg.batch_size, device=device), torch.ones(cfg.batch_size, device=device)], dim=0)
        cls_all = torch.cat([y_src, torch.argmax(logits_tgt.detach(), dim=1)], dim=0)

        ly = label_classification_loss(logits_src, y_src)
        lg = global_domain_loss(model.gd(f_all, alpha=1.0), d_all)
        ll, llc = local_domain_loss(model.gl, f_all, cls_all, d_all, alpha=1.0, n_classes=cfg.n_classes, return_per_class=True)
        omega = dynamic_adversarial_factor(lg, llc)

        if ablation == "no_local":
            ll = torch.tensor(0.0, device=device)
        if ablation == "static_omega":
            omega = torch.tensor(0.5, device=device)

        outs = compute_wdanet_objective(
            f_src,
            f_tgt,
            ly=ly,
            lg=lg,
            ll=ll,
            omega=omega,
            lam=cfg.sinkhorn_lambda,
            eps=cfg.sinkhorn_eps,
            max_iter=cfg.sinkhorn_iters,
        )
        loss = -outs["L_total"]

        opt.zero_grad()
        loss.backward()
        opt.step()

        rec = {
            "epoch": epoch,
            "loss_train": float(loss.detach().cpu()),
            "L_total": float(outs["L_total"].detach().cpu()),
            "L_y": float(ly.detach().cpu()),
            "L_g": float(lg.detach().cpu()),
            "L_l": float(ll.detach().cpu()),
            "L_w": float(outs["L_w"].detach().cpu()),
            "omega": float(omega.detach().cpu()),
        }
        history.append(rec)

    result = {
        "tag": tag,
        "ablation": ablation,
        "final": history[-1],
        "history": history,
    }
    (out_dir / f"{tag}_{ablation}.json").write_text(json.dumps(result, indent=2))
    return result
