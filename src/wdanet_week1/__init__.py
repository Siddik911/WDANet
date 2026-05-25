"""Week 1 WDANet pipeline: dataset loading, preprocessing, and DE features."""

from .config import DatasetConfig, PipelineConfig
from .datasets import build_dataset_index
from .preprocess import preprocess_signal, preprocess_file_to_windows, preprocess_raw_mne
from .de_features import extract_de_features

__all__ = [
    "DatasetConfig",
    "PipelineConfig",
    "build_dataset_index",
    "preprocess_signal",
    "preprocess_file_to_windows",
    "preprocess_raw_mne",
    "extract_de_features",
]
