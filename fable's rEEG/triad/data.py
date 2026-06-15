"""Frozen data pipeline for the resting-state EEG depression datasets.

Responsibilities (identical for every stage S0..S9 and every dataset):
  1. Load each subject's raw recording and keep only its EEG channels.
  2. Band-pass {1..40 Hz}, common-average reference, resample to the working FS=125 Hz.
  3. Cache the preprocessed continuous signal to disk (computed once).
  4. Epoch into fixed windows with robust artifact rejection.

Three datasets share this pipeline (selected via ``config.set_active_dataset``):
  * modma     — Lanzhou 2015 128-ch resting (.mat); zero-reference channel dropped.
  * mumtaz    — Mumtaz 2016 19-ch (10-20) resting (.edf), eyes-closed condition.
  * opennuero — Cavanagh ds003478 64-ch resting (.set/.fdt EEGLAB), first rest run.

Labels: MDD = 1 (positive), HC = 0. The label source is dataset-specific (see
``load_labels``). The covariance/correlation features are scale-invariant, so the
differing channel counts and physical units across datasets do not need harmonising.

Nothing in the working-rate path is allowed to change once Stage 0 is frozen.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from math import gcd

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, decimate, filtfilt, resample_poly

from . import config as C


# Canonical 10-20 EEG channels for Mumtaz (.edf), in a fixed order so every subject's
# covariance indexes the same electrodes. The 3 trailing aux channels in the recordings
# (A2-A1 linked-ear reference, 23A-23R, 24A-24R) are dropped.
MUMTAZ_CHANNELS = [
    "EEG Fp1-LE", "EEG F3-LE", "EEG C3-LE", "EEG P3-LE", "EEG O1-LE",
    "EEG F7-LE", "EEG T3-LE", "EEG T5-LE", "EEG Fz-LE", "EEG Fp2-LE",
    "EEG F4-LE", "EEG C4-LE", "EEG P4-LE", "EEG O2-LE", "EEG F8-LE",
    "EEG T4-LE", "EEG T6-LE", "EEG Cz-LE", "EEG Pz-LE",
]

# Non-EEG channels to drop from ds003478 (everything else is kept as the 64-ch montage).
OPENNUERO_DROP = {"HEOG", "VEOG", "EKG"}


# ---------------------------------------------------------------------------
# Subject container
# ---------------------------------------------------------------------------
@dataclass
class Subject:
    sid: int
    label: int                 # 0 = HC, 1 = MDD
    data: np.ndarray           # (N_CHANNELS, n_samples) float32, preprocessed
    dataset: str = ""          # source dataset id (modma|mumtaz|opennuero); set at load.
                               # Lets the feature cache + LODO recentring keep subjects from
                               # different datasets (which can share a sid) distinct.

    @property
    def name(self) -> str:
        return C.LABEL_NAMES[self.label]


# ---------------------------------------------------------------------------
# Labels  (dataset-specific source; always {subject_id: 0/1})
# ---------------------------------------------------------------------------
def load_labels() -> dict[int, int]:
    """Return {subject_id: label} for the active dataset (MDD->1, HC->0)."""
    if C.DATASET_KIND == "modma":
        df = pd.read_excel(C.INFO_XLSX)
        df = df[df["type"].isin(["MDD", "HC"])]
        return {int(row["subject id"]): C.LABELS[row["type"]] for _, row in df.iterrows()}

    if C.DATASET_KIND == "mumtaz":
        # Label comes straight from the filename group token (H -> HC, MDD -> MDD).
        cond = C.DATASET_OPTS.get("condition", "EC")
        labels: dict[int, int] = {}
        for f in C.DATA_DIR.glob(f"*{cond}.edf"):
            sid = _subject_id_from_path(f)
            labels[sid] = 1 if re.search(r"MDD", f.name, re.I) else 0
        return labels

    if C.DATASET_KIND == "opennuero":
        # BDI was bimodal by recruitment design (consistently low vs high, gap 7..13):
        # BDI <= bdi_hc_max -> HC, BDI >= bdi_mdd_min -> MDD, the lone NaN is dropped.
        df = pd.read_csv(C.DATA_DIR / "participants.tsv", sep="\t")
        bdi = pd.to_numeric(df["BDI"], errors="coerce")
        hc_max = C.DATASET_OPTS["bdi_hc_max"]
        mdd_min = C.DATASET_OPTS["bdi_mdd_min"]
        labels = {}
        for pid, b in zip(df["participant_id"], bdi):
            if pd.isna(b):
                continue
            sid = int(re.search(r"(\d+)", str(pid)).group(1))
            if b <= hc_max:
                labels[sid] = 0
            elif b >= mdd_min:
                labels[sid] = 1
        return labels

    raise ValueError(f"unknown dataset kind {C.DATASET_KIND!r}")


def _subject_id_from_path(path) -> int:
    """Parse an integer subject id from a recording path for the active dataset.

    modma:     leading digit run of the filename (``02010002rest...mat`` -> 2010002).
    mumtaz:    group + S-number; HC -> n, MDD -> 1000 + n (kept unique and sortable).
    opennuero: the ``sub-NNN`` number (-> NNN).
    """
    name = path.name if hasattr(path, "name") else str(path)
    if C.DATASET_KIND == "modma":
        m = re.match(r"(\d+)", name)
        if m is None:
            raise ValueError(f"cannot parse subject id from {name!r}")
        return int(m.group(1))
    if C.DATASET_KIND == "mumtaz":
        m = re.search(r"(MDD|H)\s*S\s*(\d+)", name, re.I)
        if m is None:
            raise ValueError(f"cannot parse Mumtaz subject id from {name!r}")
        n = int(m.group(2))
        return (1000 + n) if m.group(1).upper() == "MDD" else n
    if C.DATASET_KIND == "opennuero":
        m = re.search(r"sub-(\d+)", name)
        if m is None:
            raise ValueError(f"cannot parse sub-id from {name!r}")
        return int(m.group(1))
    raise ValueError(f"unknown dataset kind {C.DATASET_KIND!r}")


def _recording_files() -> list:
    """Sorted list of raw recording paths for the active dataset."""
    if C.DATASET_KIND == "modma":
        files = sorted(C.DATA_DIR.glob("*.mat"))
        ext = ".mat"
    elif C.DATASET_KIND == "mumtaz":
        cond = C.DATASET_OPTS.get("condition", "EC")
        files = sorted(C.DATA_DIR.glob(f"*{cond}.edf"))
        ext = f"*{cond}.edf"
    elif C.DATASET_KIND == "opennuero":
        run = C.DATASET_OPTS.get("run", "01")
        files = sorted(C.DATA_DIR.glob(f"sub-*/eeg/sub-*_task-Rest_run-{run}_eeg.set"))
        ext = f"task-Rest_run-{run}_eeg.set"
    else:
        raise ValueError(f"unknown dataset kind {C.DATASET_KIND!r}")
    if not files:
        raise FileNotFoundError(f"no {ext} recordings in {C.DATA_DIR}")
    return files


# ---------------------------------------------------------------------------
# Raw load + preprocessing (frozen working-rate path)
# ---------------------------------------------------------------------------
def _load_raw(path) -> np.ndarray:
    """Load one recording -> (N_CHANNELS, n_samples) float64 of EEG channels only."""
    if C.DATASET_KIND == "modma":
        return _load_raw_modma(path)
    if C.DATASET_KIND == "mumtaz":
        return _load_raw_mumtaz(path)
    if C.DATASET_KIND == "opennuero":
        return _load_raw_opennuero(path)
    raise ValueError(f"unknown dataset kind {C.DATASET_KIND!r}")


def _load_raw_modma(path) -> np.ndarray:
    """Load a .mat recording -> (128, n_samples); drops the zero reference channel.

    Files contain a continuous EEG array of shape (129, N) plus, in some recordings,
    auxiliary variables (sampling rate, impedances, ``DIN_*`` event markers). The data
    array is selected as the 2-D variable with >= 129 rows and the most samples.
    """
    mat = sio.loadmat(path)
    candidates = []
    for k, v in mat.items():
        if k.startswith("__") or not isinstance(v, np.ndarray) or v.ndim != 2:
            continue
        if v.shape[0] >= C.N_CHANNELS + 1:                # 129 EEG rows (128 + reference)
            candidates.append((v.shape[1], k))
    if not candidates:
        raise ValueError(f"no (>=129, N) EEG array found in {path}: {list(mat)}")
    key = max(candidates)[1]                              # most time samples
    x = np.asarray(mat[key], dtype=np.float64)           # (>=129, N)
    return x[: C.N_CHANNELS]                              # drop zero reference -> (128, N)


def _load_raw_mumtaz(path) -> np.ndarray:
    """Load a Mumtaz .edf -> (19, n_samples) of the canonical 10-20 channels (µV)."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    name_to_idx = {ch: i for i, ch in enumerate(raw.ch_names)}
    missing = [ch for ch in MUMTAZ_CHANNELS if ch not in name_to_idx]
    if missing:
        raise ValueError(f"{path.name}: missing EEG channels {missing}")
    idx = [name_to_idx[ch] for ch in MUMTAZ_CHANNELS]
    x = raw.get_data()[idx] * 1e6                         # V -> µV (scale only; corr is invariant)
    return np.asarray(x, dtype=np.float64)


def _load_raw_opennuero(path) -> np.ndarray:
    """Load a ds003478 .set -> (64, n_samples) EEG channels (HEOG/VEOG/EKG dropped, µV)."""
    import mne
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    keep = [ch for ch in raw.ch_names if ch not in OPENNUERO_DROP]
    if len(keep) != C.N_CHANNELS:
        raise ValueError(f"{path.name}: expected {C.N_CHANNELS} EEG channels, got "
                         f"{len(keep)} ({keep})")
    raw.pick(keep)
    x = raw.get_data() * 1e6                              # V -> µV
    return np.asarray(x, dtype=np.float64)


def _resample_to_working(x: np.ndarray, fs_orig: int) -> np.ndarray:
    """Band-pass {1..40 Hz} then resample fs_orig -> working FS.  Returns float32.

    Integer ratios (250->125, 500->125) use scipy ``decimate`` (zero-phase IIR) — this
    reproduces the original MODMA path exactly (factor 2). Non-integer ratios (256->125)
    use polyphase ``resample_poly``. Either way every dataset ends at FS=125 Hz so the
    epoching and model architecture are shared.
    """
    nyq = fs_orig / 2.0
    b, a = butter(C.FILTER_ORDER, [C.BANDPASS[0] / nyq, C.BANDPASS[1] / nyq], btype="band")
    x = filtfilt(b, a, x, axis=1)
    if fs_orig % C.FS == 0:
        x = decimate(x, fs_orig // C.FS, axis=1, ftype="iir", zero_phase=True)
    else:
        g = gcd(int(fs_orig), int(C.FS))
        x = resample_poly(x, int(C.FS) // g, int(fs_orig) // g, axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


def detect_bad_channels(data: np.ndarray, frac: float = C.BAD_CH_FRAC) -> np.ndarray:
    """Indices of channels that trip the artifact threshold in >``frac`` of windows.

    A persistently noisy/dead electrode (e.g. ch107 in subject 2030018) would otherwise
    cause nearly every window to be rejected. We flag such channels per recording; the
    caller drops the *union* across subjects so all covariances keep the same size.
    """
    med = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - med), axis=1, keepdims=True)
    scale = 1.4826 * mad + 1e-6
    n = (data.shape[1] - C.WIN_LEN) // C.STRIDE + 1
    idx = np.arange(C.WIN_LEN)[None, :] + C.STRIDE * np.arange(n)[:, None]
    wins = np.transpose(data[:, idx], (1, 0, 2))          # (n_win, C, WIN_LEN)
    z = np.abs(wins - med[None]) / scale[None]
    trip_frac = (z.max(axis=2) > C.REJECT_Z).mean(axis=0)  # per channel
    return np.where(trip_frac > frac)[0]


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
def build_cache(force: bool = False, verbose: bool = True) -> None:
    """Preprocess every recording once and cache to the active dataset's ``cache/{sid}.npz``.

    Two passes: (1) band-pass/resample every recording and detect its bad channels;
    (2) drop the *union* of bad channels from all subjects, re-reference (common-average
    over the surviving channels), and save. The kept-channel list (0-based indices into
    the dataset's EEG channels) is stored alongside each subject.
    """
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    files = _recording_files()
    # keep only recordings we have a label for (drops e.g. ds003478's NaN-BDI subject)
    files = [f for f in files if _subject_id_from_path(f) in labels]
    if not files:
        raise FileNotFoundError(f"no labelled recordings under {C.DATA_DIR}")

    # ---- pass 1: band-pass/resample + bad-channel detection ----
    pre: dict[int, np.ndarray] = {}
    bad_union: set[int] = set()
    bad_by_subj: dict[int, list[int]] = {}
    for i, f in enumerate(files, 1):
        sid = _subject_id_from_path(f)
        x = _resample_to_working(_load_raw(f), C.FS_ORIG)
        bad = detect_bad_channels(x)
        pre[sid] = x
        bad_by_subj[sid] = bad.tolist()
        bad_union |= set(bad.tolist())
        if verbose:
            tag = f"BAD={bad.tolist()}" if len(bad) else "ok"
            print(f"  [pass1 {i:>2}/{len(files)}] sid={sid}  {tag}")

    # MODMA drops the union of per-subject bad channels (a few dead electrodes among the
    # dense 128-array). For the fixed clinical montages (mumtaz 19, opennuero 64) the union
    # over 100+ subjects with scattered transient artifacts would swallow every channel, so
    # we keep the full montage and let the per-window robust-z rejection in epoch() handle
    # transients (verified to leave a healthy clean-window count per subject).
    if C.DATASET_OPTS.get("bad_channel_union", True):
        good = np.array(sorted(set(range(C.N_CHANNELS)) - bad_union), dtype=int)
        if verbose:
            print(f"\n  dropping {len(bad_union)} bad channel(s) from all subjects: "
                  f"{sorted(bad_union)}  ->  keeping {len(good)}/{C.N_CHANNELS} channels\n")
    else:
        good = np.arange(C.N_CHANNELS, dtype=int)
        if verbose:
            print(f"\n  keeping full {C.N_CHANNELS}-channel montage (per-window artifact "
                  f"rejection handles transients; flagged-by-subject for info only)\n")

    # ---- pass 2: drop union, re-reference over survivors, save ----
    for i, (sid, x) in enumerate(pre.items(), 1):
        x = x[good]                                       # drop bad channels
        x = x - x.mean(axis=0, keepdims=True)             # common-average over good chans
        x = np.ascontiguousarray(x, dtype=np.float32)
        np.savez(C.CACHE_DIR / f"{sid}.npz", data=x, label=labels[sid], sid=sid,
                 good_channels=good, bad_channels=np.array(bad_by_subj[sid], dtype=int))
        if verbose:
            print(f"  [pass2 {i:>2}/{len(files)}] saved   sid={sid}  "
                  f"label={C.LABEL_NAMES[labels[sid]]}  shape={x.shape}")


def load_subjects(limit: int | None = None,
                  harmonize: bool = False) -> list[Subject]:
    """Load all cached subjects (building the cache first if needed).

    ``harmonize=True`` projects every subject onto the shared 17-channel 10-20 montage
    (see ``triad.harmonize``) so correlation matrices are dimension- and electrode-
    compatible across datasets — required for any cross-dataset (LODO) run.
    """
    if not C.CACHE_DIR.exists() or not any(C.CACHE_DIR.glob("*.npz")):
        print("Cache not found — building (one-off preprocessing) ...")
        build_cache()
    sel = None  # harmonisation row selector (identical across a dataset's subjects)
    subs: list[Subject] = []
    for npz in sorted(C.CACHE_DIR.glob("*.npz"), key=lambda p: int(p.stem)):
        z = np.load(npz)
        data = z["data"].astype(np.float32)
        if harmonize:
            from . import harmonize as H
            if sel is None:
                sel = H.build_selector(C.DATASET_KIND, z["good_channels"])
            data = np.ascontiguousarray(data[sel])
        subs.append(Subject(sid=int(z["sid"]), label=int(z["label"]), data=data,
                            dataset=C.DATASET))
    if limit is not None:
        # keep a class-balanced subset for quick smoke tests
        mdd = [s for s in subs if s.label == 1][: max(1, limit // 2)]
        hc = [s for s in subs if s.label == 0][: max(1, limit - len(mdd))]
        subs = sorted(mdd + hc, key=lambda s: s.sid)
    return subs


# ---------------------------------------------------------------------------
# Epoching + artifact rejection (frozen)
# ---------------------------------------------------------------------------
def epoch(data: np.ndarray, max_windows: int | None = None) -> np.ndarray:
    """Slice continuous (C, N) into overlapping windows -> (n_win, C, WIN_LEN).

    Windows whose peak deviation exceeds ``REJECT_Z`` robust-z (per channel, scale
    estimated over the whole recording) are dropped as gross artifacts. The robust
    scale uses only this subject's own data, so the operation is leakage-free.
    """
    n = (data.shape[1] - C.WIN_LEN) // C.STRIDE + 1
    if n <= 0:
        raise ValueError("recording shorter than one window")
    idx = np.arange(C.WIN_LEN)[None, :] + C.STRIDE * np.arange(n)[:, None]
    wins = np.transpose(data[:, idx], (1, 0, 2))          # (n_win, C, WIN_LEN)

    med = np.median(data, axis=1, keepdims=True)          # (C, 1)
    mad = np.median(np.abs(data - med), axis=1, keepdims=True)
    scale = 1.4826 * mad + 1e-6                            # robust std per channel
    z = np.abs(wins - med[None]) / scale[None]            # (n_win, C, WIN_LEN)
    keep = z.reshape(n, -1).max(axis=1) < C.REJECT_Z
    wins = wins[keep]
    if wins.shape[0] == 0:                                 # never return empty
        wins = np.transpose(data[:, idx], (1, 0, 2))
    if max_windows is not None and wins.shape[0] > max_windows:
        sel = np.linspace(0, wins.shape[0] - 1, max_windows).round().astype(int)
        wins = wins[sel]
    return np.ascontiguousarray(wins, dtype=np.float32)


def summary(subs: list[Subject]) -> str:
    n_mdd = sum(s.label == 1 for s in subs)
    n_hc = sum(s.label == 0 for s in subs)
    wins = [epoch(s.data).shape[0] for s in subs]
    n_ch = subs[0].data.shape[0]
    return (f"{len(subs)} subjects  ({n_mdd} MDD / {n_hc} HC)  |  "
            f"fs={C.FS}Hz  ch={n_ch}  win={C.WIN_SEC}s/{C.WIN_LEN}smp  "
            f"windows/subj: min={min(wins)} med={int(np.median(wins))} max={max(wins)}")
