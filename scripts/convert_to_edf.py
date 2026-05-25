#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mne

from wdanet_week1.io import load_raw_with_mne


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EEG file (.set/.bdf/.fif/.edf) to EDF using MNE")
    parser.add_argument("input", type=Path, help="Input EEG file")
    parser.add_argument("output", type=Path, help="Output EDF path")
    args = parser.parse_args()

    raw = load_raw_with_mne(args.input, preload=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(args.output, raw, fmt="edf", overwrite=True)
    print(f"Converted: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
