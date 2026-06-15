"""Subject-level metrics and the per-fold live reporter.

Under strict LOSO each fold holds out exactly ONE subject, so AUC is not defined
within a fold (a single label). We therefore accumulate one (score, label) pair per
fold and report:
  * a live, *cumulative* AUC / balanced-accuracy as folds complete, and
  * the full panel (AUC, balanced acc, macro-F1, confusion matrix) at the end.

The subject is the unit of generalisation; every subject is weighted equally.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from . import config as C


@dataclass
class FoldResult:
    sid: int
    label: int
    score: float               # P(MDD) for the held-out subject (mean over its windows)
    n_windows: int
    fit_s: float = 0.0
    pred_s: float = 0.0
    extra: dict = field(default_factory=dict)


def subject_metrics(results: list[FoldResult], threshold: float = 0.5) -> dict:
    """Compute the full subject-level panel over completed folds."""
    y = np.array([r.label for r in results])
    s = np.array([r.score for r in results])
    out: dict = {"n": len(results), "n_mdd": int((y == 1).sum()), "n_hc": int((y == 0).sum())}
    if len(np.unique(y)) < 2:
        out.update(auc=float("nan"), bacc=float("nan"), f1=float("nan"), cm=None)
        return out
    pred = (s >= threshold).astype(int)
    out["auc"] = float(roc_auc_score(y, s))
    out["bacc"] = float(balanced_accuracy_score(y, pred))
    out["f1"] = float(f1_score(y, pred, average="macro"))
    out["cm"] = confusion_matrix(y, pred, labels=[0, 1])
    out["threshold"] = threshold
    return out


def youden_threshold(results: list[FoldResult]) -> float:
    """Score threshold maximising Youden's J over the completed folds."""
    y = np.array([r.label for r in results])
    s = np.array([r.score for r in results])
    if len(np.unique(y)) < 2:
        return 0.5
    order = np.argsort(-s)
    ys = y[order]
    P, N = (y == 1).sum(), (y == 0).sum()
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    j = tp / P - fp / N
    k = int(np.argmax(j))
    return float(s[order][k])


class FoldReporter:
    """Prints a one-line update per fold with running cumulative metrics."""

    def __init__(self, n_folds: int, stage: str, mode: str = "LOSO"):
        self.n_folds = n_folds
        self.stage = stage
        self.mode = mode
        self.results: list[FoldResult] = []
        self._print_header()

    def _print_header(self):
        print(f"\n=== {self.mode} training/evaluation — stage {self.stage} "
              f"({self.n_folds} subjects) ===")
        print(f"{'fold':>5} {'subject':>9} {'true':>4} {'score':>6} {'pred':>4} "
              f"{'ok':>3} {'win':>4} {'cumAUC':>7} {'cumBACC':>8}  {'time':>6}")
        print("-" * 78)

    def update(self, r: FoldResult) -> dict:
        self.results.append(r)
        m = subject_metrics(self.results)
        pred = int(r.score >= 0.5)
        ok = "Y" if pred == r.label else "."
        auc = "  -  " if np.isnan(m["auc"]) else f"{m['auc']:.4f}"
        bacc = "  -   " if np.isnan(m["bacc"]) else f"{m['bacc']:.4f}"
        print(f"{len(self.results):>5} {r.sid:>9} "
              f"{C.LABEL_NAMES[r.label]:>4} {r.score:>6.3f} "
              f"{C.LABEL_NAMES[pred]:>4} {ok:>3} {r.n_windows:>4} "
              f"{auc:>7} {bacc:>8}  {r.fit_s + r.pred_s:>5.1f}s")
        return m

    def finalize(self) -> dict:
        thr = youden_threshold(self.results)
        m05 = subject_metrics(self.results, threshold=0.5)
        mj = subject_metrics(self.results, threshold=thr)
        print("-" * 78)
        print(f"\nFINAL  stage {self.stage}  [{self.mode}]  |  {m05['n']} subjects "
              f"({m05['n_mdd']} MDD / {m05['n_hc']} HC)")
        print(f"  Subject-level ROC-AUC : {m05['auc']:.4f}   (primary)")
        print(f"  Balanced accuracy     : {m05['bacc']:.4f}  @thr=0.50   |  "
              f"{mj['bacc']:.4f}  @thr={thr:.3f} (Youden)")
        print(f"  Macro-F1              : {m05['f1']:.4f}  @thr=0.50   |  "
              f"{mj['f1']:.4f}  @thr={thr:.3f}")
        if m05["cm"] is not None:
            cm = m05["cm"]
            print("  Confusion matrix @0.50 (rows=true HC/MDD, cols=pred HC/MDD):")
            print(f"        pred_HC  pred_MDD")
            print(f"  HC  {cm[0,0]:>8} {cm[0,1]:>9}")
            print(f"  MDD {cm[1,0]:>8} {cm[1,1]:>9}")
        m05["youden_threshold"] = thr
        m05["bacc_youden"] = mj["bacc"]
        m05["f1_youden"] = mj["f1"]
        return m05
