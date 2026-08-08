#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wdanet_week3.case_protocol import CaseConfig, run_five_cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/features_de/manifest.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/wdanet_five_cases/results.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    out = run_five_cases(args.manifest, args.out, CaseConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr))
    print(out)


if __name__ == "__main__":
    main()
