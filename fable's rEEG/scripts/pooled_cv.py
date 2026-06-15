#!/usr/bin/env python3
"""Pooled 10-fold CV: merge all 3 datasets (harmonized 17-ch 10-20) into ONE cohort and
run stratified K-fold CV across the combined subjects. Every fold has all 3 datasets in
both train and test (no held-out domain) — the 'one big cohort' regime, distinct from
within-single-dataset CV and strict LODO.

Usage:  python pooled_cv.py nrde
        python pooled_cv.py spddann
"""
import json
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

import os as _os, sys as _sys  # bootstrap: repo root on path (script moved to scripts/)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from triad import config as C, data as D
from triad.models import build_model
from triad.metrics import FoldReporter, FoldResult

ALL = ["modma", "mumtaz", "opennuero"]

# the two "best settings" (harmonized 17-ch -> identity K=1, per the LODO finding)
SETTINGS = {
    "nrde": dict(max_windows=120, C_reg=1.0, n_nuisance=1),
    # deep model, supervised (pooled = all domains in train, no single gap) + fixed-nr stream
    "spddann": dict(epochs=50, lr=1e-3, subspace=12, max_windows=100, seed=42,
                    domain_adapt=False, nr_stream=True, nr_k=1, device=None),
}


def load_pooled():
    subs = []
    for ds in ALL:
        C.set_active_dataset(ds)
        s = D.load_subjects(harmonize=True)
        print(f"  [{ds}] {len(s)} subjects", flush=True)
        subs += s
    return subs


def run(stage, n_folds=10, overrides=None, out_tag=""):
    mk = dict(SETTINGS[stage])
    if overrides:                       # HP overrides (sweep) applied on top of SETTINGS
        mk.update({k: v for k, v in overrides.items() if v is not None})
    subs = load_pooled()
    n = len(subs)
    labels = np.array([s.label for s in subs])
    print(f"POOLED {n} subjects ({int(labels.sum())} MDD / {int((labels == 0).sum())} HC) "
          f"across {len(ALL)} datasets, harmonized 17-ch\n", flush=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=C.SEED)
    rep = FoldReporter(n_folds=n, stage=stage, mode=f"POOLED {n_folds}-fold CV")
    for tr, te in skf.split(np.zeros(n), labels):
        train = [subs[i] for i in tr]
        test = [subs[i] for i in te]
        model = build_model(stage, **mk)
        if hasattr(model, "set_target"):
            model.set_target(test)
        model.fit(train)
        fit_s = getattr(model, "fit_s", 0.0)
        for t in test:
            sc, nw = model.predict_subject(t)
            rep.update(FoldResult(sid=t.sid, label=t.label, score=sc, n_windows=nw,
                                  fit_s=fit_s / len(test), pred_s=0.0))
    final = rep.finalize()
    pf = [{"sid": r.sid, "label": r.label, "score": r.score, "dataset": d}
          for r, d in zip(rep.results, [None] * len(rep.results))]
    # store labels/scores (datasets not needed for AUC)
    out = {"auc": final["auc"], "bacc": final["bacc"],
           "per_fold": [{"sid": r.sid, "label": r.label, "score": r.score}
                        for r in rep.results]}
    path = f"results/pooled_{stage}{out_tag}_cv{n_folds}.json"
    json.dump(out, open(path, "w"))
    print(f"\nSaved -> {path}")
    return final


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="nrde", choices=["nrde", "spddann"])
    ap.add_argument("--subspace", type=int); ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=int); ap.add_argument("--max-windows", type=int, dest="max_windows")
    ap.add_argument("--nr-k", type=int, dest="nr_k")
    ap.add_argument("--out-tag", default="", dest="out_tag")
    a = ap.parse_args()
    ov = dict(subspace=a.subspace, lr=a.lr, epochs=a.epochs,
              max_windows=a.max_windows, nr_k=a.nr_k)
    run(a.stage, overrides=ov, out_tag=a.out_tag)
