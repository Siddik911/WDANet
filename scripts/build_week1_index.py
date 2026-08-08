#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wdanet_week1.config import DatasetConfig
from wdanet_week1.datasets import build_dataset_index


def _first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def main() -> None:
    # Resolve common local path variants so this script works without manual PYTHONPATH edits.
    ds1 = _first_existing(
        [
            Path("/home/hasan/EEG/Dataset/MODMA_128_channel_resting/EEG_128channels_ERP_lanzhou_2015"),
            Path("/home/hasan/EEG/Dataset/MODMA_128_channel_resting"),
            Path.home() / "EEG/Dataset/MODMA_128_channel_resting/EEG_128channels_ERP_lanzhou_2015",
            Path.home() / "EEG/Dataset/MODMA_128_channel_resting",
        ]
    )
    ds2 = _first_existing(
        [
            Path("/home/hasan/EEG/Dataset/ds003478-download"),
            Path.home() / "EEG/Dataset/ds003478-download",
            Path("/home/hasan/EEG/Dataset/Mumtaz/ds003478-download"),
        ]
    )
    ds3 = _first_existing(
        [
            Path("/home/hasan/EEG/Dataset/Mumtaz"),
            Path.home() / "EEG/Dataset/Mumtaz",
        ]
    )

    configs = [
        DatasetConfig(name="dataset1_modma128", path=ds1, fmt="modma_raw_128"),
        DatasetConfig(name="dataset2_openneuro_ds003478", path=ds2, fmt="openneuro_bids", label_key="group"),
        DatasetConfig(name="dataset3_mumtaz_edf", path=ds3, fmt="mumtaz_edf"),
    ]

    print("Using dataset roots:")
    for cfg in configs:
        print(f"- {cfg.name}: {cfg.path} (exists={cfg.path.exists()})")

    df = build_dataset_index(configs)
    out = ROOT / "data/week1_dataset_index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} records to {out}")

    if not df.empty:
        print(df.groupby(["dataset", "modality", "processable_eeg"]).size())


if __name__ == "__main__":
    main()
