#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import torch
from torch.optim import Adam

from wdanet_week2 import FeatureExtractor, LabelClassifier, GlobalDomainDiscriminator, LocalDomainDiscriminatorBank
from wdanet_week2 import label_classification_loss, global_domain_loss, local_domain_loss, dynamic_adversarial_factor
from wdanet_week3.integration import compute_wdanet_objective
from wdanet_week3.real_data import SplitConfig, make_loader, subject_kfold, infer_target_dim


def run_fold(train_df, test_df, args, device, target_dim):
    train_loader = make_loader(train_df, SplitConfig(batch_size=args.batch_size, num_workers=args.num_workers, target_dim=target_dim), shuffle=True)
    test_loader = make_loader(test_df, SplitConfig(batch_size=args.batch_size, num_workers=args.num_workers, target_dim=target_dim), shuffle=False)

    in_dim = target_dim
    n_classes = int(max(2, pd.to_numeric(train_df["label"], errors="coerce").fillna(0).astype(int).max() + 1))

    gf = FeatureExtractor(in_dim=in_dim, feat_dim=args.feat_dim).to(device)
    gy = LabelClassifier(feat_dim=args.feat_dim, n_classes=n_classes).to(device)
    gd = GlobalDomainDiscriminator(feat_dim=args.feat_dim).to(device)
    gl = LocalDomainDiscriminatorBank(n_classes=n_classes, feat_dim=args.feat_dim).to(device)

    opt = Adam(list(gf.parameters()) + list(gy.parameters()) + list(gd.parameters()) + list(gl.parameters()), lr=args.lr)

    for _ in range(args.epochs):
        gf.train(); gy.train(); gd.train(); gl.train()
        for xb, yb, _, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # split batch into pseudo source/target halves
            mid = xb.shape[0] // 2
            if mid < 2:
                continue
            x_src, x_tgt = xb[:mid], xb[mid:]
            y_src = yb[:mid].clamp_min(0)

            f_src = gf(x_src); f_tgt = gf(x_tgt)
            logits_src = gy(f_src); logits_tgt = gy(f_tgt)
            f_all = torch.cat([f_src, f_tgt], 0)
            d_all = torch.cat([torch.zeros(mid, device=device), torch.ones(x_tgt.shape[0], device=device)], 0)
            cls_all = torch.cat([y_src, torch.argmax(logits_tgt.detach(), 1)], 0)

            ly = label_classification_loss(logits_src, y_src)
            lg = global_domain_loss(gd(f_all, alpha=1.0), d_all)
            ll, llc = local_domain_loss(gl, f_all, cls_all, d_all, alpha=1.0, n_classes=n_classes, return_per_class=True)
            omega = dynamic_adversarial_factor(lg, llc)
            outs = compute_wdanet_objective(f_src, f_tgt, ly, lg, ll, omega, lam=args.sinkhorn_lambda, eps=args.sinkhorn_eps, max_iter=args.sinkhorn_iters)
            loss = -outs["L_total"]

            opt.zero_grad(); loss.backward(); opt.step()

    # eval
    gf.eval(); gy.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb, *_ in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = gy(gf(xb))
            pred = logits.argmax(1)
            mask = yb >= 0
            correct += int((pred[mask] == yb[mask]).sum().item())
            total += int(mask.sum().item())
    return {"acc": (correct / total) if total else 0.0, "n_test": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/features_de/manifest.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/real_training")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--feat-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sinkhorn-lambda", type=float, default=5.0)
    ap.add_argument("--sinkhorn-eps", type=float, default=1e-4)
    ap.add_argument("--sinkhorn-iters", type=int, default=50)
    ap.add_argument("--dim-strategy", type=str, default="max", choices=["max", "min", "median"])
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[pd.to_numeric(df["label"], errors="coerce").notna()].copy()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_dim = infer_target_dim(df, strategy=args.dim_strategy)
    print(f"Using unified feature dim: {target_dim} (strategy={args.dim_strategy})")

    results = []
    for k, train_df, test_df in subject_kfold(df, n_splits=10):
        r = run_fold(train_df, test_df, args, device, target_dim=target_dim)
        r["fold"] = k
        results.append(r)
        print(f"fold={k} acc={r['acc']:.4f} n_test={r['n_test']}")

    mean_acc = sum(x["acc"] for x in results) / len(results)
    out = {"mean_acc": mean_acc, "folds": results}
    (args.out / "cross_subject_results.json").write_text(json.dumps(out, indent=2))
    print(f"mean_acc={mean_acc:.4f}")


if __name__ == "__main__":
    main()
