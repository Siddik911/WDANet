"""Frozen configuration for the TRIAD pipeline.

Everything in the "FROZEN PIPELINE" block is set once (Stage 0) and must NOT change
between stages — that invariance is what makes cross-stage AUC deltas interpretable
(roadmap rule 1.1.2). Only model/training hyper-parameters may vary per stage.

The pipeline is now *dataset-aware*: the working-rate epoching/filtering constants are
identical for every dataset (so a stage behaves the same everywhere), while the raw
source location, original sampling rate, channel count and label source are selected
per dataset via ``set_active_dataset()``.  MODMA stays the default so all existing
behaviour and the frozen ``cache/`` are untouched.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# FROZEN PIPELINE  (working-rate constants — identical for every dataset)
# ---------------------------------------------------------------------------
FS = 125                      # working sampling rate after resampling (Hz)
BANDPASS = (1.0, 40.0)        # band-pass cut-offs (Hz) — covers delta/theta/alpha/beta
FILTER_ORDER = 4              # Butterworth order (applied zero-phase via filtfilt)
WIN_SEC = 4.0                 # epoch length (s)
STRIDE_SEC = 2.0              # epoch hop (s) -> 50% overlap
WIN_LEN = int(round(WIN_SEC * FS))      # 500 samples
STRIDE = int(round(STRIDE_SEC * FS))    # 250 samples
REJECT_Z = 8.0                # per-recording robust-z artifact threshold for window rejection
BAD_CH_FRAC = 0.30            # a channel is "bad" if it trips REJECT_Z in >30% of a subject's
                              # windows; the union of bad channels across subjects is dropped
                              # from every subject (keeps covariance dimension consistent)
COV_ESTIMATOR = "lwf"         # Ledoit-Wolf shrinkage covariance (pyriemann estimator id)

SEED = 42
LABELS = {"HC": 0, "MDD": 1}  # positive class = MDD (depression), per AUC convention
LABEL_NAMES = {0: "HC", 1: "MDD"}

# ---------------------------------------------------------------------------
# Dataset registry  (raw source / sampling rate / channels / label source)
# ---------------------------------------------------------------------------
# kind drives the loader + label dispatch in data.py.  Each dataset is resampled to
# the common working rate FS=125 Hz so the epoching and model architecture are shared.
DATASET_SPECS = {
    "modma": dict(                                      # Lanzhou 2015 128-ch resting (.mat)
        kind="modma",
        data_dir=PROJECT_ROOT / "EEG_128channels_resting_lanzhou_2015",
        cache_dir=CACHE_ROOT,                           # existing flat (frozen) cache
        fs_orig=250,
        n_channels=128,
        bad_channel_union=True,                         # drop union of per-subject bad chans
    ),
    "mumtaz": dict(                                     # Mumtaz 2016 MDD 19-ch resting (.edf)
        kind="mumtaz",
        data_dir=PROJECT_ROOT / "Mumtaz",
        cache_dir=CACHE_ROOT / "mumtaz",
        fs_orig=256,
        n_channels=19,
        condition="EC",                                 # eyes-closed resting (analog of MODMA)
        bad_channel_union=False,                         # keep full 10-20 montage
    ),
    "opennuero": dict(                                  # Cavanagh ds003478 64-ch resting (.set)
        kind="opennuero",
        data_dir=PROJECT_ROOT / "ds003478-download",
        cache_dir=CACHE_ROOT / "opennuero",
        fs_orig=500,
        n_channels=64,
        run="01",                                       # first rest run (high quality per README)
        bdi_hc_max=7,                                   # BDI<=7  -> HC  (recruited low group)
        bdi_mdd_min=13,                                 # BDI>=13 -> MDD (recruited high group)
        bad_channel_union=False,                         # keep full 64-ch montage
    ),
}

# Active-dataset module globals (mutated by set_active_dataset). MODMA is the default
# so importing config without a call behaves exactly as before.
DATASET = "modma"
DATASET_OPTS = DATASET_SPECS["modma"]
DATA_DIR = DATASET_OPTS["data_dir"]
CACHE_DIR = DATASET_OPTS["cache_dir"]
FS_ORIG = DATASET_OPTS["fs_orig"]
N_CHANNELS = DATASET_OPTS["n_channels"]
DATASET_KIND = DATASET_OPTS["kind"]
# MODMA-only: subject-info spreadsheet (labels).  Other datasets ignore this.
INFO_XLSX = DATA_DIR / "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx"


def set_active_dataset(name: str) -> None:
    """Point the pipeline at one of DATASET_SPECS (modma | mumtaz | opennuero).

    Mutates the module-level paths/sizes that data.py reads at call-time. The frozen
    working-rate constants (FS, WIN_LEN, BANDPASS, ...) never change.
    """
    global DATASET, DATASET_OPTS, DATA_DIR, CACHE_DIR, FS_ORIG, N_CHANNELS
    global DATASET_KIND, INFO_XLSX
    if name not in DATASET_SPECS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASET_SPECS)}")
    spec = DATASET_SPECS[name]
    DATASET = name
    DATASET_OPTS = spec
    DATA_DIR = spec["data_dir"]
    CACHE_DIR = spec["cache_dir"]
    FS_ORIG = spec["fs_orig"]
    N_CHANNELS = spec["n_channels"]
    DATASET_KIND = spec["kind"]
    INFO_XLSX = DATA_DIR / "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx"
