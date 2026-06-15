#!/usr/bin/env python3
"""Final hyperparameter sweep for SpectSPD-DANN.

7 HP configs x { within-dataset 10-fold CV (modma, mumtaz, opennuero)
               + cross-dataset LODO   (modma, mumtaz, opennuero)
               + pooled 10-fold CV (all 3 merged) }   = 49 runs.

Regime-appropriate flags (the final-model design):
  within  : --no-domain-adapt --nr-stream --nr-k 3   (native montage)
  LODO    : domain-adversarial (default), no nr-stream (nr hurts transfer)
  pooled  : --no-domain-adapt --nr-stream --nr-k 1   (harmonized 17-ch)

Metrics (subject-level, threshold 0.50, leakage-free): AUC, Acc, F1(macro),
Sensitivity, Specificity — parsed from each run's FINAL block + confusion matrix.

Streams rows to results/final_sweep.csv as runs finish (partial results survive),
then writes final_results_v1.md with the per-regime tables and the BEST config.
Run:  nohup python final_sweep.py > results/final_sweep.log 2>&1 &
"""
import csv
import re
import subprocess
import sys
import time

DATASETS = ["modma", "mumtaz", "opennuero"]
CSV_PATH = "results/final_sweep.csv"
SEED = "42"

# 7 configs (subspace / lr / epochs / max-windows)
CONFIGS = {
    "H1_base":        dict(subspace="12", lr="1e-2", epochs="80",  max_windows="100"),
    "H2_ss8":         dict(subspace="8",  lr="1e-2", epochs="80",  max_windows="100"),
    "H3_ss20":        dict(subspace="20", lr="1e-2", epochs="80",  max_windows="100"),
    "H4_lr1e-3":      dict(subspace="12", lr="1e-3", epochs="80",  max_windows="100"),
    "H5_lr5e-3ep100": dict(subspace="12", lr="5e-3", epochs="100", max_windows="100"),
    "H6_ss16":        dict(subspace="16", lr="1e-2", epochs="80",  max_windows="100"),
    "H7_mw150":       dict(subspace="12", lr="1e-2", epochs="80",  max_windows="150"),
}


def hp_args(cfg):
    return ["--subspace", cfg["subspace"], "--lr", cfg["lr"],
            "--epochs", cfg["epochs"], "--max-windows", cfg["max_windows"]]


def parse_metrics(out):
    """AUC + confusion matrix (rows true HC/MDD, cols pred HC/MDD) -> metrics @0.50."""
    m = re.search(r"ROC-AUC\s*:\s*([0-9.]+)", out)
    hc = re.search(r"^\s*HC\s+(\d+)\s+(\d+)", out, re.M)
    md = re.search(r"^\s*MDD\s+(\d+)\s+(\d+)", out, re.M)
    if not (m and hc and md):
        return None
    auc = float(m.group(1))
    TN, FP = int(hc.group(1)), int(hc.group(2))
    FN, TP = int(md.group(1)), int(md.group(2))
    tot = TN + FP + FN + TP
    sens = TP / (TP + FN) if (TP + FN) else 0.0
    spec = TN / (TN + FP) if (TN + FP) else 0.0
    acc = (TP + TN) / tot if tot else 0.0
    f1_mdd = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
    f1_hc = 2 * TN / (2 * TN + FN + FP) if (2 * TN + FN + FP) else 0.0
    return dict(auc=auc, acc=acc, f1=(f1_mdd + f1_hc) / 2, sens=sens, spec=spec)


def run(cmd):
    try:
        out = subprocess.run([sys.executable] + cmd, capture_output=True,
                             text=True, timeout=14400).stdout
    except Exception as e:
        return None, str(e)
    return parse_metrics(out), out[-400:]


def main():
    t0 = time.time()
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "eval", "dataset", "auc", "acc", "f1", "sens", "spec"]); f.flush()
        for name, cfg in CONFIGS.items():
            hp = hp_args(cfg)
            print(f"\n===== {name}  {cfg} =====", flush=True)
            # within-dataset 10-fold (native, supervised + nr-stream)
            for ds in DATASETS:
                met, _ = run(["run.py", "--stage", "spddann", "--dataset", ds, "--cv-folds", "10",
                              "--seed", SEED, "--no-domain-adapt", "--nr-stream", "--nr-k", "3",
                              "--no-save"] + hp)
                _row(w, f, name, "within", ds, met)
            # cross-dataset LODO (domain-adversarial, no nr-stream)
            for T in DATASETS:
                met, _ = run(["run.py", "--stage", "spddann", "--test", T,
                              "--seed", SEED, "--no-save"] + hp)
                _row(w, f, name, "LODO", T, met)
            # pooled 10-fold (harmonized, supervised + nr-stream k=1)
            met, _ = run(["scripts/pooled_cv.py", "spddann", "--nr-k", "1"] + hp)
            _row(w, f, name, "pooled", "all3", met)
    write_report()
    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min -> {CSV_PATH} + final_results_v1.md", flush=True)


def _row(w, f, cfg, ev, ds, met):
    if met is None:
        w.writerow([cfg, ev, ds, "ERR", "", "", "", ""]); f.flush()
        print(f"  {ev:7} {ds:10} ERR", flush=True); return
    w.writerow([cfg, ev, ds, f"{met['auc']:.4f}", f"{met['acc']:.4f}",
                f"{met['f1']:.4f}", f"{met['sens']:.4f}", f"{met['spec']:.4f}"]); f.flush()
    print(f"  {ev:7} {ds:10} AUC={met['auc']:.3f} acc={met['acc']:.3f} "
          f"f1={met['f1']:.3f} sens={met['sens']:.3f} spec={met['spec']:.3f}", flush=True)


def write_report():
    """Aggregate the CSV into final_results_v1.md with per-regime tables + best config."""
    import collections
    rows = list(csv.DictReader(open(CSV_PATH)))
    def fval(r, k):
        try: return float(r[k])
        except (ValueError, TypeError): return float("nan")
    # mean AUC per (config, regime)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["config"], r["eval"])].append(fval(r, "auc"))
    import statistics as st
    def mean(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else float("nan")
    L = ["# Final Results v1 — SpectSPD-DANN hyperparameter sweep\n",
         "Metrics: subject-level, threshold 0.50 (leakage-free). 7 configs x "
         "{within-CV x3, LODO x3, pooled}.\n"]
    # best config per regime by mean AUC
    L.append("## Best config per regime (by mean AUC)\n")
    L.append("| Regime | Best config | mean AUC |")
    L.append("|---|---|---:|")
    best_overall = None
    for reg in ["within", "LODO", "pooled"]:
        cand = {c: mean(by[(c, reg)]) for c in CONFIGS if (c, reg) in by}
        if not cand: continue
        bc = max(cand, key=lambda c: cand[c])
        L.append(f"| {reg} | **{bc}** ({_flags(bc)}) | {cand[bc]:.3f} |")
        if reg == "LODO":
            best_overall = bc
    L.append("")
    # full per-metric table for the best LODO config
    if best_overall:
        L.append(f"## All metrics for best LODO config: {best_overall}  ({_flags(best_overall)})\n")
        L.append("| Regime | Dataset | AUC | Acc | F1 | Sens | Spec |")
        L.append("|---|---|---:|---:|---:|---:|---:|")
        for r in rows:
            if r["config"] == best_overall:
                L.append(f"| {r['eval']} | {r['dataset']} | {r['auc']} | {r['acc']} | "
                         f"{r['f1']} | {r['sens']} | {r['spec']} |")
        L.append("")
    # per-regime AUC comparison across configs
    for reg in ["within", "LODO", "pooled"]:
        regrows = [r for r in rows if r["eval"] == reg]
        if not regrows: continue
        dss = sorted({r["dataset"] for r in regrows})
        L.append(f"## {reg} — AUC by config")
        L.append("| Config | " + " | ".join(dss) + " | mean |")
        L.append("|---" * (len(dss) + 2) + "|")
        for c in CONFIGS:
            vals = {r["dataset"]: r["auc"] for r in regrows if r["config"] == c}
            cells = [vals.get(d, "-") for d in dss]
            L.append(f"| {c} | " + " | ".join(cells) + f" | {mean(by[(c, reg)]):.3f} |")
        L.append("")
    open("final_results_v1.md", "w").write("\n".join(L))


def _flags(cfg):
    c = CONFIGS[cfg]
    return f"subspace {c['subspace']}, lr {c['lr']}, epochs {c['epochs']}, max-win {c['max_windows']}"


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    main()
