from __future__ import annotations

from pathlib import Path
import mne


def load_raw_with_mne(file_path: str | Path, preload: bool = True) -> mne.io.BaseRaw:
    p = Path(file_path)
    sfx = p.suffix.lower()

    if sfx == ".edf":
        return mne.io.read_raw_edf(p, preload=preload, verbose="ERROR")

    if sfx == ".set":
        return mne.io.read_raw_eeglab(p, preload=preload, verbose="ERROR")

    if sfx == ".bdf":
        return mne.io.read_raw_bdf(p, preload=preload, verbose="ERROR")

    if sfx == ".fif":
        return mne.io.read_raw_fif(p, preload=preload, verbose="ERROR")

    if sfx == ".raw":
        # MODMA original NetStation `.raw` is not directly supported by mne.
        # Prefer the MODMA BIDS export (EDF/SET) when available.
        raise ValueError(
            f"MNE cannot directly parse NetStation .raw file: {p}. "
            "Use MODMA BIDS format files (.edf/.set) for loading."
        )

    raise ValueError(f"Unsupported EEG file extension for MNE loader: {p.suffix}")
