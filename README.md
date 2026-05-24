# WDANet

## Week 1 implementation (data loaders + preprocessing + DE features)

Implemented under `src/wdanet_week1`:

- Dataset indexing for three dataset formats:
  - MODMA 128-channel `.raw`
  - OpenNeuro BIDS-style directory
  - Mumtaz `.edf` flat directory
- EEG preprocessing:
  - bandpass filtering (default 1–40 Hz)
  - notch filtering (default 50 Hz)
  - common average reference (CAR)
  - z-score normalization
  - windowing (default 10s with 10s stride)
- Differential Entropy (DE) feature extraction for bands:
  - delta (1–3), theta (4–7), alpha (8–13), beta (14–30), gamma (31–35)

## Quick start

```bash
PYTHONPATH=src python scripts/build_week1_index.py
PYTHONPATH=src python scripts/week1_smoke_test.py
```

Output index file:

- `data/week1_dataset_index.csv`
