#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from wdanet_week1.config import DatasetConfig
from wdanet_week1.datasets import build_dataset_index


def main() -> None:
    configs = [
        DatasetConfig(
            name="dataset1_modma128",
            path=Path("/home/hasan/EEG/Dataset/MODMA_128_channel_resting/EEG_128channels_ERP_lanzhou_2015"),
            fmt="modma_raw_128",
        ),
        DatasetConfig(
            name="dataset2_openneuro_ds002748",
            path=Path("/home/hasan/EEG/Dataset/Mumtaz/open neuro ds002748"),
            fmt="openneuro_bids",
            label_key="group",
        ),
        DatasetConfig(
            name="dataset3_mumtaz_edf",
            path=Path("/home/hasan/EEG/Dataset/Mumtaz"),
            fmt="mumtaz_edf",
        ),
    ]

    df = build_dataset_index(configs)
    out = Path("data/week1_dataset_index.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} records to {out}")
    if not df.empty:
        print(df.groupby(["dataset", "modality", "processable_eeg"]).size())


if __name__ == "__main__":
    main()
