from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


DEFAULT_BANDS = {
    "delta": (1.0, 3.0),
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (14.0, 30.0),
    "gamma": (31.0, 35.0),
}


@dataclass
class DatasetConfig:
    name: str
    path: Path
    fmt: str
    label_key: str | None = None


@dataclass
class PipelineConfig:
    sample_rate: float | None = None
    window_sec: float = 4.0
    stride_sec: float = 4.0
    notch_hz: float | None = 50.0
    bandpass_hz: tuple[float, float] = (1.0, 40.0)
    apply_car: bool = True
    bands: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BANDS))
    valid_extensions: Sequence[str] = (".raw", ".edf", ".edf.gz", ".set", ".bdf", ".fif")


DATASET_PRESETS: dict[str, PipelineConfig] = {
    "dataset1_modma128": PipelineConfig(sample_rate=250.0, window_sec=4.0, stride_sec=4.0),
    "dataset2_openneuro_ds003478": PipelineConfig(sample_rate=500.0, window_sec=4.0, stride_sec=4.0),
    "dataset3_mumtaz_edf": PipelineConfig(sample_rate=256.0, window_sec=4.0, stride_sec=4.0),
}
