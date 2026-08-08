#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wdanet_week3.trainer import TrainConfig, run_single_experiment


PAIRS = [
    ("dataset1", "dataset2"),
    ("dataset1", "dataset3"),
    ("dataset2", "dataset1"),
    ("dataset2", "dataset3"),
    ("dataset3", "dataset1"),
    ("dataset3", "dataset2"),
]
ABLATIONS = ["none", "no_local", "static_omega"]


def main() -> None:
    cfg = TrainConfig()
    out_dir = ROOT / "outputs/week5_cross_dataset"
    for src, tgt in PAIRS:
        for ab in ABLATIONS:
            run_single_experiment(cfg, out_dir=out_dir, tag=f"{src}_to_{tgt}", ablation=ab)
    print(f"Wrote cross-dataset outputs to {out_dir}")


if __name__ == "__main__":
    main()
