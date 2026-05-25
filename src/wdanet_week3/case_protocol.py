from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from wdanet_week2 import FeatureExtractor, LabelClassifier, GlobalDomainDiscriminator, LocalDomainDiscriminatorBank
from wdanet_week2 import label_classification_loss, global_domain_loss, local_domain_loss, dynamic_adversarial_factor
from wdanet_week3.integration import compute_wdanet_objective
from wdanet_week3.real_data import DEFeatureDataset, infer_target_dim


@dataclass
class CaseConfig:
    epochs: int = 30
    batch_size: int = 48
    feat_dim: int = 128
    lr: float = 5e-4
    sinkhorn_lambda: float = 5.0
    sinkhorn_eps: float = 1e-4
    sinkhorn_iters: int = 50
    seed: int = 42


def _encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    vals = sorted(df["label"].dropna().astype(str).unique())
    mapping = {v: i for i, v in enumerate(vals)}
    out = df.copy()
    out["label"] = out["label"].astype(str).map(mapping)
    return out, mapping


def _build_xy(df: pd.DataFrame, target_dim: int):
    ds = DEFeatureDataset(df, target_dim=target_dim)
    X, y = [], []
    for i in range(len(ds)):
        xi, yi, *_ = ds[i]
        X.append(xi)
        y.append(yi)
    return torch.stack(X, 0), torch.stack(y, 0)


def train_wdanet_uda(source_df: pd.DataFrame, target_df: pd.DataFrame, cfg: CaseConfig, device: torch.device):
    all_df = pd.concat([source_df, target_df], ignore_index=True)
    target_dim = infer_target_dim(all_df, strategy="max")
    xs, ys = _build_xy(source_df, target_dim)
    xt, _ = _build_xy(target_df, target_dim)

    xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
    in_dim = xs.shape[1]
    n_classes = int(max(2, int(ys.max().item()) + 1))

    gf = FeatureExtractor(in_dim=in_dim, feat_dim=cfg.feat_dim).to(device)
    gy = LabelClassifier(feat_dim=cfg.feat_dim, n_classes=n_classes).to(device)
    gd = GlobalDomainDiscriminator(feat_dim=cfg.feat_dim).to(device)
    gl = LocalDomainDiscriminatorBank(n_classes=n_classes, feat_dim=cfg.feat_dim).to(device)
    opt = Adam(list(gf.parameters()) + list(gy.parameters()) + list(gd.parameters()) + list(gl.parameters()), lr=cfg.lr)

    bs = cfg.batch_size
    for ep in range(cfg.epochs):
        gf.train(); gy.train(); gd.train(); gl.train()
        p = (ep + 1) / max(1, cfg.epochs)
        alpha = 2 / (1 + np.exp(-10 * p)) - 1

        nsteps = max(1, min(len(xs), len(xt)) // bs)
        for _ in range(nsteps):
            ixs = torch.randint(0, len(xs), (bs,), device=device)
            ixt = torch.randint(0, len(xt), (bs,), device=device)
            xsb, ysb, xtb = xs[ixs], ys[ixs], xt[ixt]

            f_src = gf(xsb); f_tgt = gf(xtb)
            logits_src = gy(f_src); logits_tgt = gy(f_tgt)
            f_all = torch.cat([f_src, f_tgt], 0)
            d_all = torch.cat([torch.zeros(bs, device=device), torch.ones(bs, device=device)], 0)
            cls_all = torch.cat([ysb, torch.argmax(logits_tgt.detach(), 1)], 0)

            ly = label_classification_loss(logits_src, ysb)
            lg = global_domain_loss(gd(f_all, alpha=float(alpha)), d_all)
            ll, llc = local_domain_loss(gl, f_all, cls_all, d_all, alpha=float(alpha), n_classes=n_classes, return_per_class=True)
            omega = dynamic_adversarial_factor(lg, llc)
            loss = ly if ep < max(3, cfg.epochs // 10) else -compute_wdanet_objective(f_src, f_tgt, ly, lg, ll, omega, lam=cfg.sinkhorn_lambda, eps=cfg.sinkhorn_eps, max_iter=cfg.sinkhorn_iters)["L_total"]

            opt.zero_grad(); loss.backward(); opt.step()

    return gf, gy, target_dim


def eval_classifier(gf, gy, df: pd.DataFrame, target_dim: int, device: torch.device):
    X, y = _build_xy(df, target_dim)
    X, y = X.to(device), y.to(device)
    gf.eval(); gy.eval()
    with torch.no_grad():
        p = gy(gf(X)).argmax(1)
    y_np, p_np = y.cpu().numpy(), p.cpu().numpy()
    acc = float((y_np == p_np).mean())
    tp = int(((p_np == 1) & (y_np == 1)).sum()); tn = int(((p_np == 0) & (y_np == 0)).sum())
    fp = int(((p_np == 1) & (y_np == 0)).sum()); fn = int(((p_np == 0) & (y_np == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"acc": acc, "f1": f1, "sensitivity": sens, "specificity": spec, "n": int(len(df))}


def run_five_cases(manifest_csv: Path, out_json: Path, cfg: CaseConfig):
    df = pd.read_csv(manifest_csv)
    df = df[pd.to_numeric(df["label"], errors="coerce").notna()].copy()
    df, label_map = _encode_labels(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"device": str(device), "label_map": label_map, "cases": {}}

    datasets = sorted(df["dataset"].unique())
    if len(datasets) < 3:
        raise RuntimeError(f"Need 3 datasets in manifest; found {datasets}")

    d1, d2, d3 = datasets[:3]

    # Case1: cross-subject within d1 (UDA subject-split proxy)
    s1 = df[df.dataset == d1]
    subs = sorted(s1.subject_id.unique())
    test_sub = set(subs[::5])
    tr = s1[~s1.subject_id.isin(test_sub)]
    te = s1[s1.subject_id.isin(test_sub)]
    gf, gy, td = train_wdanet_uda(tr, te, cfg, device)
    out["cases"]["case1_d1_cross_subject"] = eval_classifier(gf, gy, te, td, device)

    # Case2: cross-subject within d2
    s2 = df[df.dataset == d2]
    subs2 = sorted(s2.subject_id.unique())
    test_sub2 = set(subs2[::5])
    tr2 = s2[~s2.subject_id.isin(test_sub2)]
    te2 = s2[s2.subject_id.isin(test_sub2)]
    gf, gy, td = train_wdanet_uda(tr2, te2, cfg, device)
    out["cases"]["case2_d2_cross_subject"] = eval_classifier(gf, gy, te2, td, device)

    # Case3: d1 -> d2
    gf, gy, td = train_wdanet_uda(df[df.dataset == d1], df[df.dataset == d2], cfg, device)
    out["cases"]["case3_d1_to_d2"] = eval_classifier(gf, gy, df[df.dataset == d2], td, device)

    # Case4: d2 -> d1
    gf, gy, td = train_wdanet_uda(df[df.dataset == d2], df[df.dataset == d1], cfg, device)
    out["cases"]["case4_d2_to_d1"] = eval_classifier(gf, gy, df[df.dataset == d1], td, device)

    # Case5: cross-subject within d3
    s3 = df[df.dataset == d3]
    subs3 = sorted(s3.subject_id.unique())
    test_sub3 = set(subs3[::5])
    tr3 = s3[~s3.subject_id.isin(test_sub3)]
    te3 = s3[s3.subject_id.isin(test_sub3)]
    gf, gy, td = train_wdanet_uda(tr3, te3, cfg, device)
    out["cases"]["case5_d3_cross_subject"] = eval_classifier(gf, gy, te3, td, device)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    return out
