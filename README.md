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


## Week 3 implementation (WTC/Sinkhorn + dynamic integration)

Implemented under `src/wdanet_week3`:

- `wtc.py`:
  - cost matrix `M` via squared Euclidean distance
  - Sinkhorn/WTC iterative updates for `u`, `v`, `T`, `T_rev`
- `wasserstein.py`:
  - forward/reverse Wasserstein distribution discriminator losses
- `integration.py`:
  - Eq. (21) integration:
    `L_total = L_y - (omega * L_g + (1 - omega) * L_l + L_w)`

Also updated Week 2 dynamic adversarial factor to Eq. (6)-(8)-style using per-class local losses.

Smoke test:

```bash
python scripts/week3_smoke_test.py
```


## Week 4–6 execution

### Week 4: Cross-subject experiments
```bash
python -m pip install torch numpy
python scripts/run_week4_cross_subject.py
```

### Week 5: Cross-dataset + ablations
```bash
python scripts/run_week5_cross_dataset.py
```

### Week 6: Final metrics + report
```bash
python scripts/run_week6_report.py
cat outputs/week6_report/summary.json
cat outputs/week6_report/report.md
```

### One-shot run (Week 4 -> 6)
```bash
python scripts/run_week4_cross_subject.py && \
python scripts/run_week5_cross_dataset.py && \
python scripts/run_week6_report.py
```


## Real dataset pipeline (RTX 3060 ready)

1) Build index (edit dataset paths if needed):
```bash
python scripts/build_week1_index.py
```

2) Build DE feature store from EEG files:
```bash
python scripts/build_week1_features.py --index data/week1_dataset_index.csv --out data/features_de --win-sec 4 --stride-sec 4
```

3) Train WDANet with 10-fold cross-subject:
```bash
python scripts/train_wdanet_real.py --manifest data/features_de/manifest.csv --epochs 30 --batch-size 48 --num-workers 4 --dim-strategy max
```

4) Results:
```bash
cat outputs/real_training/cross_subject_results.json
```

### Suggested RTX 3060 settings
- `--batch-size 48` (paper-aligned starting point)
- if OOM: `--batch-size 24`
- keep `--num-workers 4` (or 2 on Windows)


If you mix datasets with different channel counts, keep `--dim-strategy max` to pad smaller feature vectors and avoid DataLoader stack errors.


### Accuracy debugging updates
- Labels are now remapped to contiguous class IDs at train time.
- Per-fold subject-level validation split is used for model selection.
- Training includes warmup epochs (supervised-only) before full adversarial+WTC objective.
- GRL strength `alpha` follows a DANN schedule across epochs.


## Author-style evaluation (Cases 1-5 + baselines)

Run:
```bash
python scripts/evaluate_author_protocol.py --manifest data/features_de/manifest.csv
```

Outputs `outputs/author_eval/author_style_eval.json` with:
- cross-subject per dataset (Cases 1/2/5)
- cross-dataset transfers (Cases 3/4 approximation)
- baseline metrics (DT, kNN, SVM, NB, BNN)
- metrics: accuracy, F1, sensitivity, specificity
