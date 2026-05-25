from __future__ import annotations

import numpy as np
import mne

from .io import load_raw_with_mne


def _window_signal(x: np.ndarray, fs: float, win_sec: float, stride_sec: float) -> np.ndarray:
    win = int(win_sec * fs)
    step = int(stride_sec * fs)
    if x.shape[-1] < win:
        return np.empty((0, x.shape[0], win), dtype=x.dtype)
    starts = range(0, x.shape[-1] - win + 1, step)
    return np.stack([x[:, s : s + win] for s in starts], axis=0)


def preprocess_raw_mne(
    raw: mne.io.BaseRaw,
    band: tuple[float, float] = (1.0, 40.0),
    notch_hz: float | None = 50.0,
    apply_car: bool = True,
) -> mne.io.BaseRaw:
    out = raw.copy().load_data()
    out.pick("eeg")
    out.filter(l_freq=band[0], h_freq=band[1], verbose="ERROR")
    if notch_hz is not None:
        out.notch_filter(freqs=[notch_hz], verbose="ERROR")
    if apply_car:
        out.set_eeg_reference("average", verbose="ERROR")
    return out


def preprocess_signal(
    signal: np.ndarray,
    fs: float,
    band: tuple[float, float] = (1.0, 40.0),
    notch_hz: float | None = 50.0,
    apply_car: bool = True,
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[None, :]

    ch_names = [f"EEG{i+1}" for i in range(signal.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(signal, info, verbose="ERROR")
    raw = preprocess_raw_mne(raw, band=band, notch_hz=notch_hz, apply_car=apply_car)

    x = raw.get_data()
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
    return _window_signal(x, fs=raw.info["sfreq"], win_sec=win_sec, stride_sec=stride_sec)


def preprocess_file_to_windows(
    file_path: str,
    win_sec: float = 10.0,
    stride_sec: float = 10.0,
    band: tuple[float, float] = (1.0, 40.0),
    notch_hz: float | None = 50.0,
    apply_car: bool = True,
) -> tuple[np.ndarray, float, list[str]]:
    raw = load_raw_with_mne(file_path, preload=True)
    raw = preprocess_raw_mne(raw, band=band, notch_hz=notch_hz, apply_car=apply_car)
    x = raw.get_data()
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
    windows = _window_signal(x, fs=raw.info["sfreq"], win_sec=win_sec, stride_sec=stride_sec)
    return windows, float(raw.info["sfreq"]), raw.ch_names
