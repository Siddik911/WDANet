#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wdanet_week3.trainer import TrainConfig, run_single_experiment


def main() -> None:
    cfg = TrainConfig()
    out_dir = ROOT / "outputs/week4_cross_subject"
    for fold in range(10):
        run_single_experiment(cfg, out_dir=out_dir, tag=f"fold_{fold}", ablation="none")
    print(f"Wrote cross-subject outputs to {out_dir}")


if __name__ == "__main__":
    main()
