from __future__ import annotations

from pathlib import Path
import gzip
import shutil
import tempfile

import mne


class UnsupportedModalityError(ValueError):
    pass


def _open_mne_by_suffix(path: Path, preload: bool) -> mne.io.BaseRaw:
    sfx = path.suffix.lower()
    if sfx == ".edf":
        return mne.io.read_raw_edf(path, preload=preload, verbose="ERROR")
    if sfx == ".set":
        return mne.io.read_raw_eeglab(path, preload=preload, verbose="ERROR")
    if sfx == ".bdf":
        return mne.io.read_raw_bdf(path, preload=preload, verbose="ERROR")
    if sfx == ".fif":
        return mne.io.read_raw_fif(path, preload=preload, verbose="ERROR")
    raise ValueError(f"Unsupported EEG file extension for MNE loader: {path.suffix}")


def load_raw_with_mne(file_path: str | Path, preload: bool = True) -> mne.io.BaseRaw:
    p = Path(file_path)
    suffixes = [s.lower() for s in p.suffixes]

    if suffixes[-2:] == [".nii", ".gz"] or (suffixes and suffixes[-1] == ".nii"):
        raise UnsupportedModalityError(f"NIfTI fMRI file is not EEG and cannot be loaded by this EEG pipeline: {p}")

    if suffixes and suffixes[-1] == ".raw":
        raise ValueError(
            f"MNE cannot directly parse NetStation .raw file: {p}. "
            "Convert to EDF/SET/BDF/FIF first."
        )

    if suffixes and suffixes[-1] == ".gz":
        base_suffix = suffixes[-2] if len(suffixes) >= 2 else ""
        if base_suffix not in {".edf", ".set", ".bdf", ".fif"}:
            raise ValueError(f"Unsupported gzip EEG file type: {p}")
        with tempfile.TemporaryDirectory(prefix="wdanet_eeg_") as td:
            tmp_path = Path(td) / p.name[:-3]
            with gzip.open(p, "rb") as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            raw = _open_mne_by_suffix(tmp_path, preload=preload)
            return raw.copy().load_data()

    return _open_mne_by_suffix(p, preload=preload)
