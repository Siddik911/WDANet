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


def _set_1020_montage_if_possible(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    out = raw.copy()
    montage = mne.channels.make_standard_montage("standard_1020")
    # Ignore missing names so non-10-20 channels do not fail conversion.
    out.set_montage(montage, on_missing="ignore", verbose="ERROR")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert EEG file (.set/.bdf/.fif/.edf and gzip variants) to EDF using MNE"
    )
    parser.add_argument("input", type=Path, help="Input EEG file")
    parser.add_argument("output", type=Path, help="Output EDF path")
    parser.add_argument(
        "--apply-proj",
        action="store_true",
        help="Apply projectors before export so they are preserved in exported data",
    )
    parser.add_argument(
        "--set-1020-montage",
        action="store_true",
        help="Attach standard_1020 montage before export (recommended for 10-20 datasets)",
    )
    args = parser.parse_args()

    raw = load_raw_with_mne(args.input, preload=True)

    # Ensure EEG channels are explicitly typed as EEG for EDF channel prefixes/semantics.
    eeg_map = {ch: "eeg" for ch in raw.ch_names if ch.upper() not in {"EOG", "ECG", "EMG"}}
    if eeg_map:
        raw.set_channel_types(eeg_map, verbose="ERROR")

    if args.set_1020_montage:
        raw = _set_1020_montage_if_possible(raw)

    if args.apply_proj:
        raw.apply_proj()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(args.output, raw, fmt="edf", overwrite=True)
    print(f"Converted: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
