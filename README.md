# WDANet

## Week 1 implementation (MNE-based data loaders + preprocessing + DE features)

Implemented under `src/wdanet_week1`:

- Dataset indexing for three dataset formats:
  - MODMA 128-channel listing (`.raw` files indexed; loading requires conversion)
  - OpenNeuro BIDS-style EEG directory (`sub-*/eeg/*.set|*.edf|*.bdf|*.fif`)
  - Mumtaz `.edf` / `.edf.gz` flat directory
- EEG preprocessing (via **MNE-Python**):
  - bandpass filtering (default 1–40 Hz)
  - notch filtering (default 50 Hz)
  - common average reference (CAR)
  - z-score normalization
  - windowing
- Differential Entropy (DE) feature extraction for bands:
  - delta (1–3), theta (4–7), alpha (8–13), beta (14–30), gamma (31–35)

## Dataset-specific preprocessing presets (requested)

- MODMA (128-ch): 250 Hz, 4s window, 1000 samples
- OpenNeuro ds003478: 500 Hz, 4s window, 2000 samples
- MUMTAZ EEG: 256 Hz, 4s window, 1024 samples

Presets are defined in `wdanet_week1.config.DATASET_PRESETS`.

## Quick start

```bash
python -m pip install mne numpy scipy pandas openpyxl
python scripts/build_week1_index.py
python scripts/week1_smoke_test.py
```

## Convert EEG files to EDF (MNE)

Use:

```bash
python scripts/convert_to_edf.py <input_file> <output_file.edf>
```

Example:

```bash
python scripts/convert_to_edf.py \
  ~/EEG/Dataset/ds003478-download/sub-001/eeg/sub-001_task-Rest_run-01_eeg.set \
  ~/EEG/Dataset/ds003478-download/sub-001/eeg/sub-001_task-Rest_run-01_eeg.edf
```


### EDF export notes

This project uses `mne.export.export_raw(..., fmt="edf")` in `scripts/convert_to_edf.py`.
For 10-20 datasets, use `--set-1020-montage`.
If you have unapplied projectors, use `--apply-proj` before export.

Example:

```bash
python scripts/convert_to_edf.py \
  --set-1020-montage --apply-proj \
  <input_eeg_file> <output_file.edf>
```


## Week 2 implementation (WDANet modules + GRL + core losses)

Implemented under `src/wdanet_week2`:

- `grl.py`: Gradient Reversal Layer (`GradientReversal`, `grad_reverse`)
- `models.py`:
  - `FeatureExtractor` (MLP)
  - `LabelClassifier`
  - `GlobalDomainDiscriminator` (with GRL)
  - `LocalDomainDiscriminator` and `LocalDomainDiscriminatorBank`
- `losses.py`:
  - source label CE loss
  - global domain BCE loss
  - local class-conditional domain BCE loss
  - dynamic adversarial factor `omega`

Smoke test:

```bash
python scripts/week2_smoke_test.py
```
