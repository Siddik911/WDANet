#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from wdanet_week1.de_features import extract_de_features
from wdanet_week1.preprocess import preprocess_signal


fs = 250.0
seconds = 30
samples = int(fs * seconds)
raw = np.random.randn(3, samples)
windows = preprocess_signal(raw, fs=fs, win_sec=10, stride_sec=10)
feats, bands = extract_de_features(windows, fs=fs)
print("windows", windows.shape)
print("features", feats.shape)
print("bands", bands)
