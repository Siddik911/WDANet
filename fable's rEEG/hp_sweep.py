#!/usr/bin/env python3
"""SpecSPD-DANN hyperparameter sweep: 7 configs x {LODO (3 targets) + 10-fold CV (3 datasets)}.

LODO  runs: domain-adversarial (the cross-dataset model), per held-out target dataset.
CV10  runs: supervised pure-deep (--no-domain-adapt), per dataset — shows HP sensitivity of
            the learned encoder (the nr-stream add-on is evaluated separately).
Single seed (42). --no-save (we read AUC from stdout; avoids clobbering result JSONs).
Results stream to results/hp_sweep.csv as each run finishes (partial results survive a crash).
"""
import csv
import re
import subprocess
import sys
import time

DATASETS = ["modma", "mumtaz", "opennuero"]

# 7 hyperparameter configs (vary subspace / lr / epochs / max-windows around the baseline)
CONFIGS = {
    "H1_base":      ["--subspace", "12", "--lr", "1e-2", "--epochs", "80",  "--max-windows", "100"],
    "H2_ss8":       ["--subspace", "8",  "--lr", "1e-2", "--epochs", "80",  "--max-windows", "100"],
    "H3_ss20":      ["--subspace", "20", "--lr", "1e-2", "--epochs", "80",  "--max-windows", "100"],
    "H4_lr1e-3":    ["--subspace", "12", "--lr", "1e-3", "--epochs", "80",  "--max-windows", "100"],
    "H5_lr5e-3ep100": ["--subspace", "12", "--lr", "5e-3", "--epochs", "100", "--max-windows", "100"],
    "H6_ss16":      ["--subspace", "16", "--lr", "1e-2", "--epochs", "80",  "--max-windows", "100"],
    "H7_mw150":     ["--subspace", "12", "--lr", "1e-2", "--epochs", "80",  "--max-windows", "150"],
}

CSV_PATH = "results/hp_sweep.csv"


def run_one(args) -> str:
    out = subprocess.run([sys.executable, "run.py"] + args,
                         capture_output=True, text=True).stdout
    m = re.search(r"ROC-AUC\s*:\s*([0-9.]+)", out)
    return m.group(1) if m else "ERR"


def main():
    t0 = time.time()
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "eval", "dataset", "auc"]); f.flush()
        for label, hp in CONFIGS.items():
            print(f"\n===== {label}  {' '.join(hp)} =====", flush=True)
            for T in DATASETS:                          # LODO (domain-adversarial)
                auc = run_one(["--stage", "spddann", "--test", T, "--seed", "42",
                               "--no-save"] + hp)
                w.writerow([label, "LODO", T, auc]); f.flush()
                print(f"  LODO  {T:10} AUC={auc}", flush=True)
            for D in DATASETS:                          # within-dataset 10-fold (supervised pure-deep)
                auc = run_one(["--stage", "spddann", "--dataset", D, "--cv-folds", "10",
                               "--seed", "42", "--no-domain-adapt", "--no-save"] + hp)
                w.writerow([label, "CV10", D, auc]); f.flush()
                print(f"  CV10  {D:10} AUC={auc}", flush=True)
    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min -> {CSV_PATH}", flush=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    main()
