from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1)


def notch_filter(x: np.ndarray, fs: float, f0: float, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(w0=f0, Q=q, fs=fs)
    return filtfilt(b, a, x, axis=-1)


def common_average_reference(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=0, keepdims=True)


def window_signal(x: np.ndarray, fs: float, win_sec: float, stride_sec: float) -> np.ndarray:
    win = int(win_sec * fs)
    step = int(stride_sec * fs)
    if x.shape[-1] < win:
        return np.empty((0, x.shape[0], win), dtype=x.dtype)

    starts = range(0, x.shape[-1] - win + 1, step)
    return np.stack([x[:, s : s + win] for s in starts], axis=0)


def preprocess_signal(
    signal: np.ndarray,
    fs: float,
    band: tuple[float, float] = (1.0, 40.0),
    notch_hz: float | None = 50.0,
    apply_car: bool = True,
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]

    x = bandpass_filter(x, fs, band[0], band[1])
    if notch_hz is not None:
        x = notch_filter(x, fs, notch_hz)
    if apply_car and x.shape[0] > 1:
        x = common_average_reference(x)

    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
    return window_signal(x, fs, win_sec, stride_sec)
