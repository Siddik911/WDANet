from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class DEFeatureDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, target_dim: int | None = None):
        self.df = manifest_df.reset_index(drop=True)
        self.target_dim = target_dim

    def __len__(self):
        return len(self.df)

    def _fix_dim(self, x: np.ndarray) -> np.ndarray:
        if self.target_dim is None:
            return x
        if x.shape[0] == self.target_dim:
            return x
        if x.shape[0] > self.target_dim:
            return x[: self.target_dim]
        out = np.zeros((self.target_dim,), dtype=np.float32)
        out[: x.shape[0]] = x
        return out

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        z = np.load(r["feature_path"], allow_pickle=True)
        x = z["x"].astype(np.float32)  # [n_win, n_ch, n_band]
        x = x.reshape(x.shape[0], -1).mean(axis=0)  # sample-level vector
        x = self._fix_dim(x)
        y = int(r["label"]) if str(r["label"]) not in {"nan", "None"} else -1
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), r["subject_id"], r["dataset"]


@dataclass
class SplitConfig:
    batch_size: int = 48
    num_workers: int = 4
    target_dim: int | None = None


def infer_target_dim(manifest_df: pd.DataFrame, strategy: str = "max") -> int:
    dims = []
    for fp in manifest_df["feature_path"].tolist():
        z = np.load(fp, allow_pickle=True)
        x = z["x"]
        dims.append(int(x.shape[1] * x.shape[2]))
    if not dims:
        return 0
    if strategy == "min":
        return min(dims)
    if strategy == "median":
        return int(np.median(dims))
    return max(dims)


def make_loader(df: pd.DataFrame, cfg: SplitConfig, shuffle: bool) -> DataLoader:
    ds = DEFeatureDataset(df, target_dim=cfg.target_dim)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, drop_last=False)


def subject_kfold(manifest: pd.DataFrame, n_splits: int = 10):
    subjects = sorted(manifest["subject_id"].unique())
    folds = [subjects[i::n_splits] for i in range(n_splits)]
    for k in range(n_splits):
        test_sub = set(folds[k])
        train_df = manifest[~manifest["subject_id"].isin(test_sub)].copy()
        test_df = manifest[manifest["subject_id"].isin(test_sub)].copy()
        yield k, train_df, test_df
