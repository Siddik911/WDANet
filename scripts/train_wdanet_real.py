#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
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


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    vals = sorted(df["label"].dropna().astype(str).unique())
    mapping = {v: i for i, v in enumerate(vals)}
    out = df.copy()
    out["label"] = out["label"].astype(str).map(mapping)
    return out, mapping


def split_train_val_by_subject(train_df: pd.DataFrame, val_ratio: float = 0.1, seed: int = 42):
    subs = sorted(train_df["subject_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(subs)
    n_val = max(1, int(len(subs) * val_ratio))
    val_subs = set(subs[:n_val])
    tr = train_df[~train_df["subject_id"].isin(val_subs)].copy()
    va = train_df[train_df["subject_id"].isin(val_subs)].copy()
    return tr, va


def eval_acc(gf, gy, loader, device):
    gf.eval(); gy.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb, *_ in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = gy(gf(xb)).argmax(1)
            mask = yb >= 0
            correct += int((pred[mask] == yb[mask]).sum().item())
            total += int(mask.sum().item())
    return (correct / total) if total else 0.0


def run_fold(train_df, test_df, args, device, target_dim):
    tr_df, va_df = split_train_val_by_subject(train_df, val_ratio=0.1, seed=args.seed)

    tr_loader = make_loader(tr_df, SplitConfig(batch_size=args.batch_size, num_workers=args.num_workers, target_dim=target_dim), shuffle=True)
    va_loader = make_loader(va_df, SplitConfig(batch_size=args.batch_size, num_workers=args.num_workers, target_dim=target_dim), shuffle=False)
    te_loader = make_loader(test_df, SplitConfig(batch_size=args.batch_size, num_workers=args.num_workers, target_dim=target_dim), shuffle=False)

    in_dim = target_dim
    n_classes = int(max(2, train_df["label"].max() + 1))

    gf = FeatureExtractor(in_dim=in_dim, feat_dim=args.feat_dim).to(device)
    gy = LabelClassifier(feat_dim=args.feat_dim, n_classes=n_classes).to(device)
    gd = GlobalDomainDiscriminator(feat_dim=args.feat_dim).to(device)
    gl = LocalDomainDiscriminatorBank(n_classes=n_classes, feat_dim=args.feat_dim).to(device)

    opt = Adam(list(gf.parameters()) + list(gy.parameters()) + list(gd.parameters()) + list(gl.parameters()), lr=args.lr)

    best = {"val": -1.0, "state": None}
    for epoch in range(args.epochs):
        gf.train(); gy.train(); gd.train(); gl.train()
        p = (epoch + 1) / max(1, args.epochs)
        alpha = 2 / (1 + math.exp(-10 * p)) - 1
        for xb, yb, _, db in tr_loader:
            xb = xb.to(device); yb = yb.to(device)
            # source: labeled; target: another domain samples from same batch where possible
            mid = xb.shape[0] // 2
            if mid < 4:
                continue
            x_src, x_tgt = xb[:mid], xb[mid:]
            y_src = yb[:mid].clamp_min(0)

            f_src = gf(x_src); f_tgt = gf(x_tgt)
            logits_src = gy(f_src); logits_tgt = gy(f_tgt)

            f_all = torch.cat([f_src, f_tgt], 0)
            d_all = torch.cat([torch.zeros(mid, device=device), torch.ones(x_tgt.shape[0], device=device)], 0)
            cls_all = torch.cat([y_src, torch.argmax(logits_tgt.detach(), 1)], 0)

            ly = label_classification_loss(logits_src, y_src)
            lg = global_domain_loss(gd(f_all, alpha=alpha), d_all)
            ll, llc = local_domain_loss(gl, f_all, cls_all, d_all, alpha=alpha, n_classes=n_classes, return_per_class=True)
            omega = dynamic_adversarial_factor(lg, llc)

            # warmup: supervised first, then full objective
            if epoch < max(3, args.epochs // 10):
                loss = ly
            else:
                outs = compute_wdanet_objective(f_src, f_tgt, ly, lg, ll, omega, lam=args.sinkhorn_lambda, eps=args.sinkhorn_eps, max_iter=args.sinkhorn_iters)
                loss = -outs["L_total"]

            opt.zero_grad(); loss.backward();
            torch.nn.utils.clip_grad_norm_(list(gf.parameters()) + list(gy.parameters()) + list(gd.parameters()) + list(gl.parameters()), 5.0)
            opt.step()

        val_acc = eval_acc(gf, gy, va_loader, device)
        if val_acc > best["val"]:
            best["val"] = val_acc
            best["state"] = {
                "gf": {k: v.detach().cpu().clone() for k, v in gf.state_dict().items()},
                "gy": {k: v.detach().cpu().clone() for k, v in gy.state_dict().items()},
            }

    if best["state"] is not None:
        gf.load_state_dict(best["state"]["gf"])
        gy.load_state_dict(best["state"]["gy"])

    test_acc = eval_acc(gf, gy, te_loader, device)
    n_test = len(test_df)
    return {"acc": test_acc, "n_test": n_test, "best_val_acc": best["val"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/features_de/manifest.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/real_training")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--feat-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sinkhorn-lambda", type=float, default=5.0)
    ap.add_argument("--sinkhorn-eps", type=float, default=1e-4)
    ap.add_argument("--sinkhorn-iters", type=int, default=50)
    ap.add_argument("--dim-strategy", type=str, default="max", choices=["max", "min", "median"])
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    df = pd.read_csv(args.manifest)
    df = df[pd.to_numeric(df["label"], errors="coerce").notna()].copy()
    df, label_map = encode_labels(df)

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
        print(f"fold={k} acc={r['acc']:.4f} val={r['best_val_acc']:.4f} n_test={r['n_test']}")

    mean_acc = sum(x["acc"] for x in results) / len(results)
    out = {"mean_acc": mean_acc, "folds": results, "label_map": label_map}
    (args.out / "cross_subject_results.json").write_text(json.dumps(out, indent=2))
    print(f"mean_acc={mean_acc:.4f}")


if __name__ == "__main__":
    main()
