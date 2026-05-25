from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .preprocess import preprocess_file_to_windows
from .de_features import extract_de_features


@dataclass
class BuildFeatureConfig:
    out_dir: Path
    win_sec: float = 4.0
    stride_sec: float = 4.0
    band: tuple[float, float] = (1.0, 40.0)
    notch_hz: float | None = 50.0
    apply_car: bool = True


def build_de_feature_store(index_csv: Path, cfg: BuildFeatureConfig) -> Path:
    df = pd.read_csv(index_csv)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, r in df.iterrows():
        if not bool(r.get("processable_eeg", True)):
            continue
        fp = Path(str(r["file_path"]))
        if not fp.exists():
            continue
        try:
            windows, fs, ch_names = preprocess_file_to_windows(
                str(fp),
                win_sec=cfg.win_sec,
                stride_sec=cfg.stride_sec,
                band=cfg.band,
                notch_hz=cfg.notch_hz,
                apply_car=cfg.apply_car,
            )
            if windows.shape[0] == 0:
                continue
            feats, bands = extract_de_features(windows, fs=fs)
        except Exception:
            continue

        out_fp = cfg.out_dir / f"sample_{i:06d}.npz"
        np.savez_compressed(
            out_fp,
            x=feats.astype(np.float32),  # [n_win, n_ch, n_band]
            label=np.array(r.get("label", -1)),
            subject_id=np.array(str(r.get("subject_id", "unknown"))),
            dataset=np.array(str(r.get("dataset", "unknown"))),
            fs=np.array(float(fs)),
            ch_names=np.array(ch_names, dtype=object),
            bands=np.array(bands, dtype=object),
        )
        rows.append(
            {
                "feature_path": str(out_fp),
                "subject_id": str(r.get("subject_id", "unknown")),
                "dataset": str(r.get("dataset", "unknown")),
                "label": r.get("label", -1),
            }
        )

    manifest = cfg.out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    meta = {
        "n_samples": len(rows),
        "win_sec": cfg.win_sec,
        "stride_sec": cfg.stride_sec,
        "band": list(cfg.band),
        "notch_hz": cfg.notch_hz,
        "apply_car": cfg.apply_car,
    }
    (cfg.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return manifest
