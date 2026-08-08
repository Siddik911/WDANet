from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


DEFAULT_BANDS = {
    "delta": (1.0, 3.0),
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (14.0, 30.0),
    "gamma": (31.0, 35.0),
}


def _bandpass_epoch(epoch: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = fs * 0.5
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, epoch, axis=-1)


def _differential_entropy(x: np.ndarray) -> np.ndarray:
    var = np.var(x, axis=-1) + 1e-12
    return 0.5 * np.log(2 * np.pi * np.e * var)


def extract_de_features(
    windows: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return DE features with shape [n_windows, n_channels, n_bands]."""
    if windows.ndim != 3:
        raise ValueError("Expected windows shape [n_windows, n_channels, n_samples]")

    band_map = bands or DEFAULT_BANDS
    band_names = list(band_map.keys())

    feats = []
    for b in band_names:
        lo, hi = band_map[b]
        band_data = _bandpass_epoch(windows, fs, lo, hi)
        feats.append(_differential_entropy(band_data))

    features = np.stack(feats, axis=-1)
    return features, band_names
