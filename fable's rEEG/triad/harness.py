"""LOSO / K-fold CV harness — shared by every stage.

LOSO (default): train on 52, test on 1, repeat for all 53 subjects.
K-fold CV (--cv-folds K): stratified K-fold, train on ~(K-1)/K subjects,
test on ~1/K subjects per fold. Matches WDANet's 10-fold CV protocol.

The held-out subject labels NEVER touch anything but the final scoring
function. Per-fold lines are printed live with cumulative AUC / balanced
accuracy.
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from . import config as C
from . import data as D
from .metrics import FoldReporter, FoldResult
from .models import build_model


def run_loso(stage: str, model_kwargs: dict | None = None,
             limit_subjects: int | None = None, save: bool = True,
             permute_labels: int | None = None,
             cv_folds: int | None = None) -> dict:
    model_kwargs = dict(model_kwargs or {})
    subs = D.load_subjects(limit=limit_subjects)
    print(D.summary(subs))

    if permute_labels is not None:
        import dataclasses
        rng = np.random.default_rng(permute_labels)
        shuffled = rng.permutation([s.label for s in subs])
        subs = [dataclasses.replace(s, label=int(lab)) for s, lab in zip(subs, shuffled)]
        print(f"** LABEL-PERMUTATION leakage test (seed={permute_labels}) — "
              f"expect final AUC ~ 0.5 **")

    is_loso = cv_folds is None or cv_folds >= len(subs)
    n_folds = len(subs) if is_loso else cv_folds
    mode_label = "LOSO" if is_loso else f"{n_folds}-fold CV"

    reporter = FoldReporter(n_folds=len(subs), stage=stage, mode=mode_label)
    t_start = time.time()

    if is_loso:
        folds = [([j for j in range(len(subs)) if j != i], [i])
                 for i in range(len(subs))]
    else:
        labels = np.array([s.label for s in subs])
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=C.SEED)
        folds = [(list(tr), list(te))
                 for tr, te in skf.split(np.zeros(len(subs)), labels)]

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train = [subs[i] for i in train_idx]
        test_subs = [subs[i] for i in test_idx]

        sids = ", ".join(str(s.sid) for s in test_subs)
        print(f"  ... fold {fold_i+1:>2}/{n_folds}: training on {len(train)} subjects "
              f"(held-out: {sids}) ...", flush=True)

        model = build_model(stage, **model_kwargs)
        model.fit(train)
        fit_s = getattr(model, "fit_s", 0.0)

        for test in test_subs:
            t0 = time.time()
            score, n_win = model.predict_subject(test)
            reporter.update(FoldResult(
                sid=test.sid, label=test.label, score=score, n_windows=n_win,
                fit_s=fit_s / len(test_subs), pred_s=time.time() - t0,
            ))

    final = reporter.finalize()
    final["stage"] = stage
    final["dataset"] = C.DATASET
    final["model_kwargs"] = model_kwargs
    final["cv_folds"] = n_folds
    final["wall_time_s"] = round(time.time() - t_start, 1)
    print(f"\nTotal wall time: {final['wall_time_s']:.1f}s")

    if save:
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        per_fold = [{"sid": r.sid, "label": r.label, "score": r.score,
                     "n_windows": r.n_windows} for r in reporter.results]
        out = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in final.items()}
        out["per_fold"] = per_fold
        suffix = "loso" if is_loso else f"cv{n_folds}"
        # prefix non-MODMA datasets so results don't clobber each other (MODMA keeps its
        # original bare {stage}_{suffix}.json names that the roadmap references)
        prefix = "" if C.DATASET == "modma" else f"{C.DATASET}_"
        path = C.RESULTS_DIR / f"{prefix}{stage}_{suffix}.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"Saved per-fold results -> {path}")
    return final
