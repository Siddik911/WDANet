#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wdanet_week1.feature_store import BuildFeatureConfig, build_de_feature_store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=ROOT / "data/week1_dataset_index.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/features_de")
    ap.add_argument("--win-sec", type=float, default=4.0)
    ap.add_argument("--stride-sec", type=float, default=4.0)
    args = ap.parse_args()

    manifest = build_de_feature_store(
        args.index,
        BuildFeatureConfig(out_dir=args.out, win_sec=args.win_sec, stride_sec=args.stride_sec),
    )
    print(f"Wrote feature manifest: {manifest}")
    meta = args.out / "meta.json"
    err = args.out / "errors.csv"
    if meta.exists():
        print(f"Meta: {meta}")
    if err.exists():
        print(f"Errors logged: {err}")


if __name__ == "__main__":
    main()
