"""Channel harmonisation to a shared 10-20 montage — the mandatory first step for LODO.

The three datasets use incompatible montages (MODMA 128-ch EGI HydroCel GSN,
Mumtaz 19-ch 10-20, OpenNeuro 64-ch 10-10). Riemannian/tangent-space methods and the
``nr`` identity-removal step need correlation matrices of *identical dimension* and
*identical electrode meaning* across datasets, so before any cross-dataset (LODO) step
every dataset is projected onto the shared 10-20 intersection.

Shared montage = **17 electrodes**. We start from the classic 19-channel 10-20 set and
drop the two that are not physically present in MODMA (verified against the cache and the
BIDS sidecars — see the audit in the LODO build):

  * **Cz**  — in the EGI HydroCel GSN-128 layout the vertex Cz is the recording
              REFERENCE (E129); ``data._load_raw_modma`` keeps only E1..E128, so MODMA
              has no Cz channel at all.
  * **T4 (T8)** — EGI E108 is a dead electrode in MODMA (e.g. subject 2030018, the
              ``ch107`` noted in ``data.detect_bad_channels``) and is removed by the
              bad-channel union in ``data.build_cache``, so it is absent from every MODMA
              subject's cached signal.

Interpolating either back would inject a dataset-specific *synthetic* channel into the
correlation structure — exactly the artefact LODO must avoid — so they are dropped from
the shared set instead. The remaining 17 are present and clean in all three datasets.

The selector maps each canonical channel to a ROW INDEX into the (already
bad-channel-dropped) cached ``data`` array, in ``HARMONIZED_CHANNELS`` order, so the
caller does ``data = data[selector]`` and everything downstream sees a fixed 17xN signal.
"""
from __future__ import annotations

import numpy as np

from . import config as C

# Canonical shared montage (classic 10-20 names), Cz and T4 excluded (see module docstring).
HARMONIZED_CHANNELS = [
    "Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "Fz",
    "Fp2", "F4", "C4", "P4", "O2", "F8", "T6", "Pz",
]
N_HARMONIZED = len(HARMONIZED_CHANNELS)  # 17

# MODMA: canonical -> EGI HydroCel GSN-128 electrode number (1-based "E" label).
# Standard HydroCel 10-20 correspondence; Cz=E129 (reference) and T4=E108 (dead) omitted.
_MODMA_EGI1 = {
    "Fp1": 22, "F3": 24, "C3": 36, "P3": 52, "O1": 70, "F7": 33, "T3": 45,
    "T5": 58, "Fz": 11, "Fp2": 9, "F4": 124, "C4": 104, "P4": 92, "O2": 83,
    "F8": 122, "T6": 96, "Pz": 62,
}

# OpenNeuro (ds003478): canonical (old 10-20) -> modern 10-10 name used in channels.tsv.
_OPENNUERO_NAME = {
    "Fp1": "FP1", "F3": "F3", "C3": "C3", "P3": "P3", "O1": "O1", "F7": "F7",
    "T3": "T7", "T5": "P7", "Fz": "FZ", "Fp2": "FP2", "F4": "F4", "C4": "C4",
    "P4": "P4", "O2": "O2", "F8": "F8", "T6": "P8", "Pz": "PZ",
}


def _opennuero_channel_order() -> list[str]:
    """The 64 EEG channel names in cache-row order (= BIDS channels.tsv minus aux channels).

    Verified: MNE's ``raw.ch_names`` for the .set files matches channels.tsv order exactly,
    so the cache row index of a channel equals its position in this list.
    """
    import pandas as pd
    from .data import OPENNUERO_DROP

    run = C.DATASET_OPTS.get("run", "01")
    tsvs = sorted(C.DATA_DIR.glob(f"sub-*/eeg/sub-*_task-Rest_run-{run}_channels.tsv"))
    if not tsvs:
        raise FileNotFoundError(f"no channels.tsv under {C.DATA_DIR} (run-{run})")
    names = pd.read_csv(tsvs[0], sep="\t")["name"].tolist()
    return [n for n in names if n not in OPENNUERO_DROP]


def build_selector(dataset_kind: str, good_channels: np.ndarray) -> np.ndarray:
    """Row indices into the cached (bad-dropped) data array for HARMONIZED_CHANNELS.

    ``good_channels`` is the per-dataset array stored in each cached ``.npz`` mapping
    cache row -> original 0-based channel index (identical across subjects of a dataset).
    Raises if any of the 17 montage channels is missing (a hard error — never silently
    return a smaller montage, which would make cross-dataset matrices incompatible).
    """
    gc = np.asarray(good_channels).ravel()

    if dataset_kind == "modma":
        # original 0-based GSN index -> cache row
        pos = {int(v): i for i, v in enumerate(gc)}
        rows = []
        for ch in HARMONIZED_CHANNELS:
            e0 = _MODMA_EGI1[ch] - 1                      # E-number is 1-based
            if e0 not in pos:
                raise ValueError(f"MODMA: channel {ch} (E{e0 + 1}) not in survivors")
            rows.append(pos[e0])

    elif dataset_kind == "mumtaz":
        from .data import MUMTAZ_CHANNELS
        # cache rows are exactly MUMTAZ_CHANNELS order; map short name -> row
        short_to_row = {full.split()[1].split("-")[0]: i
                        for i, full in enumerate(MUMTAZ_CHANNELS)}
        rows = []
        for ch in HARMONIZED_CHANNELS:
            if ch not in short_to_row:
                raise ValueError(f"Mumtaz: channel {ch} not found in montage")
            rows.append(short_to_row[ch])

    elif dataset_kind == "opennuero":
        order = _opennuero_channel_order()
        name_to_row = {n: i for i, n in enumerate(order)}
        rows = []
        for ch in HARMONIZED_CHANNELS:
            modern = _OPENNUERO_NAME[ch]
            if modern not in name_to_row:
                raise ValueError(f"OpenNeuro: channel {ch} ({modern}) not found")
            rows.append(name_to_row[modern])

    else:
        raise ValueError(f"unknown dataset kind {dataset_kind!r}")

    sel = np.asarray(rows, dtype=int)
    if len(set(rows)) != N_HARMONIZED:
        raise ValueError(f"{dataset_kind}: selector has duplicates: {rows}")
    if sel.max() >= len(gc):
        raise ValueError(f"{dataset_kind}: selector row {sel.max()} out of range {len(gc)}")
    return sel


if __name__ == "__main__":
    # Sanity print: resolve the selector for each dataset against its built cache.
    import glob
    for ds, cdir in [("modma", C.CACHE_ROOT), ("mumtaz", C.CACHE_ROOT / "mumtaz"),
                     ("opennuero", C.CACHE_ROOT / "opennuero")]:
        C.set_active_dataset(ds)
        fs = sorted(glob.glob(str(cdir / "*.npz")))
        if not fs:
            print(f"{ds:10s}: no cache")
            continue
        z = np.load(fs[0])
        sel = build_selector(ds, z["good_channels"])
        print(f"{ds:10s}: data {z['data'].shape} -> 17 rows {sel.tolist()}")
    print(f"\nHARMONIZED_CHANNELS ({N_HARMONIZED}): {HARMONIZED_CHANNELS}")
