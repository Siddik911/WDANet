from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from .config import DatasetConfig


def _safe_subject_id(name: str) -> str:
    cleaned = re.sub(r"\s+", "_", name.strip())
    return re.sub(r"[^a-zA-Z0-9_\-]", "", cleaned)


def _extract_mumtaz_label(file_name: str) -> int | None:
    # Heuristic based on provided examples: H => healthy (0), D/MDD => depression (1)
    upper = file_name.upper()
    if "_H" in upper or upper.startswith("H "):
        return 0
    if "_D" in upper or "MDD" in upper or upper.startswith("D "):
        return 1
    return None


def index_modma_128(cfg: DatasetConfig) -> pd.DataFrame:
    info_files = list(cfg.path.glob("*subjects_information*.xlsx"))
    recordings = sorted(cfg.path.glob("*.raw"))

    info_df = pd.read_excel(info_files[0]) if info_files else pd.DataFrame()

    rows = []
    for rec in recordings:
        subject_id = _safe_subject_id(rec.stem)
        rows.append(
            {
                "dataset": cfg.name,
                "subject_id": subject_id,
                "session": "rest",
                "label": None,
                "file_path": str(rec),
                "format": "raw",
                "channels": 128,
                "metadata_found": not info_df.empty,
            }
        )
    return pd.DataFrame(rows)


def index_openneuro_bids(cfg: DatasetConfig) -> pd.DataFrame:
    participants_tsv = cfg.path / "participants.tsv"
    pmap = {}
    if participants_tsv.exists():
        pdf = pd.read_csv(participants_tsv, sep="\t")
        for _, row in pdf.iterrows():
            sub = str(row.get("participant_id", "")).strip()
            label_val = row.get(cfg.label_key, None) if cfg.label_key else None
            pmap[sub] = label_val

    rows = []
    for eeg in sorted(cfg.path.glob("sub-*/**/*")):
        if eeg.suffix.lower() not in {".edf", ".set", ".bdf"}:
            continue
        sub = eeg.parts[-3] if len(eeg.parts) >= 3 else "unknown"
        rows.append(
            {
                "dataset": cfg.name,
                "subject_id": sub,
                "session": "rest",
                "label": pmap.get(sub, None),
                "file_path": str(eeg),
                "format": eeg.suffix.lower().lstrip("."),
                "channels": None,
                "metadata_found": participants_tsv.exists(),
            }
        )
    return pd.DataFrame(rows)


def index_mumtaz_edf(cfg: DatasetConfig) -> pd.DataFrame:
    rows = []
    for edf in sorted(cfg.path.glob("*.edf")):
        label = _extract_mumtaz_label(edf.name)
        condition = "EO" if " EO" in edf.name else "EC" if " EC" in edf.name else "TASK"
        rows.append(
            {
                "dataset": cfg.name,
                "subject_id": _safe_subject_id(edf.stem),
                "session": condition.lower(),
                "label": label,
                "file_path": str(edf),
                "format": "edf",
                "channels": 3,
                "metadata_found": True,
            }
        )
    return pd.DataFrame(rows)


def build_dataset_index(configs: Iterable[DatasetConfig]) -> pd.DataFrame:
    frames = []
    for cfg in configs:
        if cfg.fmt == "modma_raw_128":
            frames.append(index_modma_128(cfg))
        elif cfg.fmt == "openneuro_bids":
            frames.append(index_openneuro_bids(cfg))
        elif cfg.fmt == "mumtaz_edf":
            frames.append(index_mumtaz_edf(cfg))
        else:
            raise ValueError(f"Unsupported dataset format: {cfg.fmt}")

    cols = ["dataset", "subject_id", "session", "label", "file_path", "format", "channels", "metadata_found"]
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    return out if not out.empty else pd.DataFrame(columns=cols)
