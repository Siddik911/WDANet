from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from .config import DatasetConfig


EEG_EXTENSIONS = {".edf", ".set", ".bdf", ".fif", ".edf.gz", ".bdf.gz", ".set.gz"}
FMRI_EXTENSIONS = {".nii.gz", ".nii"}


def _safe_subject_id(name: str) -> str:
    cleaned = re.sub(r"\s+", "_", name.strip())
    return re.sub(r"[^a-zA-Z0-9_\-]", "", cleaned)


def _extract_mumtaz_label(file_name: str) -> int | None:
    upper = file_name.upper()
    if "_H" in upper or upper.startswith("H "):
        return 0
    if "_D" in upper or "MDD" in upper or upper.startswith("D "):
        return 1
    return None


def _suffix2(path: Path) -> str:
    if len(path.suffixes) >= 2:
        return "".join(path.suffixes[-2:]).lower()
    return path.suffix.lower()


def index_modma_128(cfg: DatasetConfig) -> pd.DataFrame:
    info_files = list(cfg.path.glob("*subjects_information*.xlsx"))
    recordings = sorted(cfg.path.glob("*.raw"))
    info_df = pd.read_excel(info_files[0]) if info_files else pd.DataFrame()

    rows = []
    for rec in recordings:
        rows.append(
            {
                "dataset": cfg.name,
                "subject_id": _safe_subject_id(rec.stem),
                "session": "rest",
                "label": None,
                "file_path": str(rec),
                "format": "raw",
                "modality": "eeg",
                "channels": 128,
                "metadata_found": not info_df.empty,
                "processable_eeg": False,
                "note": "NetStation .raw needs conversion to EDF/SET/BDF/FIF for MNE",
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
    for fp in sorted(cfg.path.glob("sub-*/**/*")):
        if not fp.is_file():
            continue
        sfx = _suffix2(fp)
        if sfx not in EEG_EXTENSIONS and sfx not in FMRI_EXTENSIONS:
            continue

        sub = next((part for part in fp.parts if part.startswith("sub-")), "unknown")
        modality = "fmri" if sfx in FMRI_EXTENSIONS else "eeg"
        processable_eeg = modality == "eeg"

        rows.append(
            {
                "dataset": cfg.name,
                "subject_id": sub,
                "session": "rest",
                "label": pmap.get(sub, None),
                "file_path": str(fp),
                "format": sfx.lstrip("."),
                "modality": modality,
                "channels": None,
                "metadata_found": participants_tsv.exists(),
                "processable_eeg": processable_eeg,
                "note": "fMRI file in ds002748" if modality == "fmri" else "",
            }
        )
    return pd.DataFrame(rows)


def index_mumtaz_edf(cfg: DatasetConfig) -> pd.DataFrame:
    rows = []
    patterns = ["*.edf", "*.edf.gz"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(cfg.path.glob(pat)))

    for edf in sorted(set(files)):
        label = _extract_mumtaz_label(edf.name)
        condition = "EO" if " EO" in edf.name else "EC" if " EC" in edf.name else "TASK"
        rows.append(
            {
                "dataset": cfg.name,
                "subject_id": _safe_subject_id(edf.stem.replace(".edf", "")),
                "session": condition.lower(),
                "label": label,
                "file_path": str(edf),
                "format": _suffix2(edf).lstrip("."),
                "modality": "eeg",
                "channels": 3,
                "metadata_found": True,
                "processable_eeg": True,
                "note": "",
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

    cols = [
        "dataset",
        "subject_id",
        "session",
        "label",
        "file_path",
        "format",
        "modality",
        "channels",
        "metadata_found",
        "processable_eeg",
        "note",
    ]
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    return out if not out.empty else pd.DataFrame(columns=cols)
