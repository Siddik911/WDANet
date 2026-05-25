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

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def load_xy(manifest: pd.DataFrame):
    X, y, groups, ds = [], [], [], []
    for _, r in manifest.iterrows():
        z = np.load(r["feature_path"], allow_pickle=True)
        x = z["x"].astype(np.float32).reshape(z["x"].shape[0], -1).mean(axis=0)
        X.append(x)
        y.append(int(r["label"]))
        groups.append(str(r["subject_id"]))
        ds.append(str(r["dataset"]))
    max_dim = max(len(v) for v in X)
    X2 = np.zeros((len(X), max_dim), dtype=np.float32)
    for i, v in enumerate(X):
        X2[i, : len(v)] = v
    return X2, np.array(y), np.array(groups), np.array(ds)


def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {"acc": acc, "f1": f1, "sensitivity": sens, "specificity": spec}


def run_group_cv(X, y, groups, model, n_splits=10):
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    ys, ps = [], []
    for tr, te in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        p = model.predict(X[te])
        ys.extend(y[te].tolist()); ps.extend(p.tolist())
    return metrics(np.array(ys), np.array(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/features_de/manifest.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/author_eval")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[pd.to_numeric(df["label"], errors="coerce").notna()].copy()
    # normalize labels to 0/1
    labs = {v: i for i, v in enumerate(sorted(df["label"].astype(str).unique()))}
    df["label"] = df["label"].astype(str).map(labs).astype(int)

    X, y, groups, datasets = load_xy(df)

    models = {
        "DT": DecisionTreeClassifier(random_state=42),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
        "NB": GaussianNB(),
        "BNN": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)),
    }

    out = {"cases": {}, "baselines": {}}

    # Case 1/2/5 cross-subject within each dataset
    for ds in sorted(np.unique(datasets)):
        m = datasets == ds
        if m.sum() < 10:
            continue
        out["cases"][f"cross_subject_{ds}"] = run_group_cv(
            X[m], y[m], groups[m], make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42))
        )

    # Case 3/4 cross-dataset transfers (classifier-only approximation)
    uniq = sorted(np.unique(datasets))
    for src in uniq:
        for tgt in uniq:
            if src == tgt:
                continue
            ms, mt = datasets == src, datasets == tgt
            if ms.sum() == 0 or mt.sum() == 0:
                continue
            clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42))
            clf.fit(X[ms], y[ms])
            pred = clf.predict(X[mt])
            out["cases"][f"cross_dataset_{src}_to_{tgt}"] = metrics(y[mt], pred)

    # Baselines over full grouped CV
    for name, model in models.items():
        out["baselines"][name] = run_group_cv(X, y, groups, model)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "author_style_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
