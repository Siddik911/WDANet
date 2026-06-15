"""TRIAD — Transductive Riemannian Adaptation for cross-subject LODO depression detection.

Staged, ablation-driven build-up on the MODMA 128-channel resting-state EEG dataset.
The data pipeline (`triad.data`) is FROZEN at Stage 0 and shared unchanged by every
stage; only the model (`triad.models`) changes between stages. See
TRIAD_Progressive_Research_Roadmap.md.
"""

__all__ = ["config", "data", "metrics", "models", "harness"]
