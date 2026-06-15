#!/usr/bin/env python3
"""
build_bib.py — Verified BibTeX bibliography builder for IEEE TNSRE paper.

GOLDEN RULE: Never hand-write a BibTeX entry. Every entry must come from
CrossRef or OpenAlex API. Unverified papers go to to-verify.txt.

Usage:
    cd "/home/hasan/fable's rEEG/Paper/scripts"
    python3 build_bib.py

Outputs:
    ../refs.bib          — verified BibTeX entries
    to-verify.txt        — papers requiring manual entry
"""

import urllib.request
import urllib.parse
import json
import time
import re
import os

MAILTO = "sihabhasan4567@gmail.com"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
REFS_BIB = os.path.join(PAPER_DIR, "refs.bib")
TO_VERIFY = os.path.join(SCRIPT_DIR, "to-verify.txt")

# ---------------------------------------------------------------------------
# Paper database — extracted from docs/Paper Summary.md
# format: {"id", "title", "first_author", "year", "venue", "doi", "notes"}
# ---------------------------------------------------------------------------

PAPERS = [
    {"id": "P1",  "title": "Alternatives to Sine Carrier in Auditory BCI: Exploring Machine Learning Strategies for Assessing Modulation Detectability in EEG",
     "first_author": "Gueho", "year": "2026", "venue": "IEEE OJSP", "doi": None},
    {"id": "P2",  "title": "A New Subject-Specific Discriminative and Multi-Scale Filter Bank Tangent Space Mapping Method for Recognition of Multiclass Motor Imagery",
     "first_author": "Wu", "year": "2021", "venue": "Frontiers in Human Neuroscience", "doi": None},
    {"id": "P3",  "title": "Riemannian High-Order Pooling for Brain Foundation Models",
     "first_author": "Hu", "year": "2026", "venue": "ICLR 2026", "doi": None},
    {"id": "P4",  "title": "A Sliced-Wasserstein Framework on Correlation Matrices for EEG Decoding",
     "first_author": "Hu", "year": "2026", "venue": "KDD 2026", "doi": None,
     "notes": "CorSW — PRIORITY paper"},
    {"id": "P5",  "title": "A study on the combination of functional connection features and Riemannian manifold in EEG emotion recognition",
     "first_author": "Wu", "year": "2024", "venue": "Frontiers in Neuroscience", "doi": None},
    {"id": "P6",  "title": "Classification of covariance matrices using a Riemannian-based kernel for BCI applications",
     "first_author": "Barachant", "year": "2013", "venue": "Neurocomputing", "doi": None},
    {"id": "P7",  "title": "Combining detrended cross-correlation analysis with Riemannian geometry-based classification for improved brain-computer interface performance",
     "first_author": "Racz", "year": "2024", "venue": "Frontiers in Neuroscience", "doi": None},
    {"id": "P8",  "title": "Decoding Cognitive States via Riemannian Geometry-Informed Channel Clustering for EEG Transformers",
     "first_author": "Feng", "year": "2026", "venue": "Mathematics MDPI", "doi": None,
     "notes": "EEG-RCFormer — PRIORITY paper"},
    {"id": "P9",  "title": "Multiclass Brain-Computer Interface Classification by Riemannian Geometry",
     "first_author": "Barachant", "year": "2012", "venue": "IEEE Transactions on Biomedical Engineering", "doi": None},
    {"id": "P10", "title": "CTSSP: A Temporal-Spectral-Spatial Joint Optimization Algorithm for Motor Imagery EEG Decoding",
     "first_author": "Unknown", "year": "2026", "venue": "Journal of Neural Engineering", "doi": None},
    {"id": "P11", "title": "Pseudo Affine-Invariant Riemannian Metrics for Efficient Brain-Computer Interfaces",
     "first_author": "Unknown", "year": "2026", "venue": "HAL preprint", "doi": None,
     "notes": "HAL preprint — no CrossRef DOI likely"},
    {"id": "P12", "title": "Real-time EEG-based Emotion Recognition Using Riemannian Quantification Learning",
     "first_author": "Unknown", "year": "2026", "venue": "Biomedical Signal Processing and Control", "doi": None},
    {"id": "P13", "title": "RepSPD: Enhancing SPD Manifold Representation in EEGs via Dynamic Graphs",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv preprint", "doi": None,
     "notes": "arXiv preprint — check arxiv.org for ID"},
    {"id": "P14", "title": "Riemannian Geometry-Preserving Variational Autoencoder for MI-BCI Data Augmentation",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv preprint", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P15", "title": "Riemannian Geometry Meets fMRI: The Advantages of Modeling Correlation Manifolds and Eigenvector Subspaces",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv preprint", "doi": None,
     "notes": "arXiv preprint — King's College London"},
    {"id": "P16", "title": "Riemannian Spatio-Temporal Graph Neural Network for Enhanced Cognitive Load Detection Using EEG",
     "first_author": "Unknown", "year": "2026", "venue": "Neurocomputing", "doi": None},
    {"id": "P17", "title": "Stiefel-SPD Manifold Graph Convolution for End-to-End EEG Learning",
     "first_author": "Unknown", "year": "2026", "venue": "IEEE Transactions on Neural Systems and Rehabilitation Engineering", "doi": None},
    {"id": "P18", "title": "The Riemannian Geometry of User Learning in MI-BCI: A Cybathlon Longitudinal Study",
     "first_author": "Unknown", "year": "2026", "venue": "Research Square preprint", "doi": None,
     "notes": "Research Square preprint — no stable DOI"},
    {"id": "P19", "title": "Filter Bank CSP with Riemannian Weighting for Disability-Centric Motor Imagery BCI",
     "first_author": "Unknown", "year": "2026", "venue": "Brain Informatics", "doi": None},
    {"id": "P20", "title": "Objective Assessment of Familiarity in Music Using Imagery and EEG-Based Machine Learning",
     "first_author": "Unknown", "year": "2026", "venue": "Scientific Reports", "doi": None},
    {"id": "P21", "title": "Robust Brainprint Recognition via Session-wise Riemannian Alignment: Overcoming Non-stationary Brain Dynamics",
     "first_author": "Unknown", "year": "2026", "venue": "Research Square preprint", "doi": None,
     "notes": "Research Square preprint — S-RCT"},
    {"id": "P22", "title": "SPD Domain-Specific Batch Normalization to Crack Interpretable Unsupervised Domain Adaptation in EEG",
     "first_author": "Kobler", "year": "2022", "venue": "NeurIPS 2022", "doi": None,
     "notes": "TSMNet/SPDDSMBN — PRIORITY paper"},
    {"id": "P23", "title": "Domain Adaptation Using Riemannian Geometry of SPD Matrices",
     "first_author": "Yair", "year": "2019", "venue": "ICASSP 2019", "doi": None},
    {"id": "P24", "title": "Tangent Space Alignment: Transfer Learning for Brain-Computer Interface",
     "first_author": "Rodrigues", "year": "2022", "venue": "Frontiers in Human Neuroscience", "doi": None,
     "notes": "TSA — PRIORITY paper"},
    {"id": "P25", "title": "Cross-Subject EEG Emotion Recognition Using Riemannian Graph Transformers with Geodesic Adversarial Adaptation",
     "first_author": "Unknown", "year": "2026", "venue": "Alexandria Engineering Journal", "doi": None},
    {"id": "P26", "title": "Routing on the Stiefel Manifold: When Does Adaptive Subspace Selection Help for Cross-Domain EEG Decoding",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv preprint", "doi": None,
     "notes": "arXiv preprint — GIPSA-lab"},
    {"id": "P27", "title": "Geometric Moment Alignment for Domain Adaptation via Siegel Embeddings",
     "first_author": "Unknown", "year": "2026", "venue": "ICLR 2026", "doi": None},
    {"id": "P28", "title": "Transformer-Based SFDA by Class-Balanced Multicentric Dynamic Pseudo-Labeling for Privacy-Preserving EEG-Based BCI Systems",
     "first_author": "Unknown", "year": "2026", "venue": "MDPI", "doi": None},
    {"id": "P29", "title": "Cross-Subject EEG Decoding with Spatiotemporal Transformers for Zero-Calibration Brain-Computer Interfaces",
     "first_author": "Unknown", "year": "2026", "venue": "Alexandria Engineering Journal", "doi": None},
    {"id": "P30", "title": "Transfer Learning for the Riemannian Tangent Space: Applications to Brain-Computer Interfaces",
     "first_author": "Bleuze", "year": "2021", "venue": "ICEET 2021", "doi": None},
    {"id": "P31", "title": "Selective Cross-Subject Transfer Learning Based on Riemannian Tangent Space for Motor Imagery Brain-Computer Interface",
     "first_author": "Unknown", "year": "2021", "venue": "Frontiers in Neuroscience", "doi": None},
    {"id": "P32", "title": "A Self-Attention Domain Adaptation Network Based on Riemannian Representation and Contrastive Learning for EEG Driver Drowsiness Detection",
     "first_author": "Unknown", "year": "2026", "venue": "Engineering Science and Technology", "doi": None},
    {"id": "P33", "title": "A Band-Aware Riemannian Network with Domain Adaptation for Motor Imagery EEG Signal Decoding",
     "first_author": "Unknown", "year": "2026", "venue": "MDPI", "doi": None},
    {"id": "P34", "title": "FB-RCSP-RA: A Filter-Bank Regularized CSP Framework with Per-Band Riemannian Alignment for Cross-Subject Motor Imagery EEG Decoding",
     "first_author": "Unknown", "year": "2026", "venue": "HAL preprint", "doi": None,
     "notes": "HAL preprint — no CrossRef DOI likely"},
    {"id": "P35", "title": "Rotation-Based Metric on the Riemannian Manifold of SPD Matrices with Applications to Source Data Selection for BCI Transfer Learning",
     "first_author": "Unknown", "year": "2026", "venue": "Frontiers in Human Neuroscience", "doi": None},
    {"id": "P36", "title": "Domain Generalized Feature Embedded Learning for Calibration-Free Event-Related Potentials Recognition",
     "first_author": "Unknown", "year": "2026", "venue": "Cognitive Neurodynamics", "doi": None},
    {"id": "P37", "title": "Cross-Dataset Variability Problem in EEG Decoding with Deep Learning",
     "first_author": "Kostas", "year": "2020", "venue": "Frontiers in Human Neuroscience", "doi": None},
    {"id": "P38", "title": "Latent Alignment in Deep Learning Models for EEG Decoding",
     "first_author": "Unknown", "year": "2025", "venue": "Journal of Neural Engineering", "doi": None},
    {"id": "P39", "title": "Source Data Selection for Brain-Computer Interfaces Based on Simple Features",
     "first_author": "Unknown", "year": "2026", "venue": "IEEE Access", "doi": None},
    {"id": "P40", "title": "Dual Selections Based Knowledge Transfer Learning for Cross-Subject Motor Imagery EEG Classification",
     "first_author": "Unknown", "year": "2023", "venue": "Frontiers in Neuroscience", "doi": None},
    {"id": "P41", "title": "Maximizing Single-Feature Separability for Improving Transfer Learning in Motor Imagery EEG Decoding",
     "first_author": "Unknown", "year": "2026", "venue": "MDPI", "doi": None},
    {"id": "P42", "title": "Seizure Prediction With HIVE-CODAs: The Hierarchical Vote Collective of Domain Adaptation Methods",
     "first_author": "Unknown", "year": "2022", "venue": "Frontiers in Physics", "doi": None},
    {"id": "P43", "title": "A Time-Frequency Transform and Riemannian Manifold-Based Domain Adaptation Method for Motor Imagery in Brain Source Space",
     "first_author": "Unknown", "year": "2026", "venue": "Chinese biomedical journal", "doi": None,
     "notes": "Chinese journal — may not be in CrossRef"},
    {"id": "P44", "title": "WDANet: Wasserstein Distribution Inspired Dynamic Adversarial Network for EEG-Based Cross-Domain Depression Recognition",
     "first_author": "Unknown", "year": "2026", "venue": "IEEE Transactions on Affective Computing", "doi": None,
     "notes": "WDANet — PRIORITY paper, IEEE TAC 2026"},
    {"id": "P45", "title": "ReTa-Diffusion: Exploring Task-State from Resting-State in EEG Signals via Bidirectional Decoupling and Latent Guiding for Early Detection of Subclinical Depression",
     "first_author": "Unknown", "year": "2026", "venue": "ICLR 2026 under review", "doi": None,
     "notes": "ICLR 2026 under review — not yet published"},
    {"id": "P46", "title": "HybridRDG: Zero-Shot ASD Biomarker Detection from Multi-Paradigm EEG via Hybrid Deep-Riemannian Domain Generalization",
     "first_author": "Unknown", "year": "2026", "venue": "CVPR Workshop 2026", "doi": None,
     "notes": "CVPR Workshop 2026 — may not be indexed yet"},
    {"id": "P47", "title": "REDDI: A Riemannian Ensemble Learning Framework for Interpretable Differential Diagnosis of Neurodegenerative Diseases",
     "first_author": "Unknown", "year": "2026", "venue": "medRxiv", "doi": None,
     "notes": "medRxiv preprint — no stable DOI"},
    {"id": "P48", "title": "Riemannian Geometry of Functional Connectivity Matrices for Multi-Site ADHD Data Harmonization",
     "first_author": "Colclough", "year": "2022", "venue": "Frontiers in Neuroinformatics", "doi": None,
     "notes": "ADHD harmonization — PRIORITY paper"},
    {"id": "P49", "title": "Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry",
     "first_author": "Unknown", "year": "2025", "venue": "NeurIPS 2025", "doi": None},
    {"id": "P50", "title": "PSDNorm: Temporal Normalization for Deep Learning in Sleep Staging",
     "first_author": "Unknown", "year": "2026", "venue": "ICLR 2026", "doi": None,
     "notes": "PSDNorm — PRIORITY paper"},
    {"id": "P51", "title": "NeuroBRIDGE: Behavior-Conditioned Koopman Dynamics with Riemannian Alignment for Early Substance Use Initiation Prediction",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P52", "title": "Real-Time Decoding of Movement Onset and Offset for Brain-Controlled Rehabilitation Exoskeleton",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P53", "title": "EEG-Based Classification of Alzheimer's Disease and Frontotemporal Dementia Using Functional Connectivity",
     "first_author": "Unknown", "year": "2026", "venue": "Scientific Reports", "doi": None},
    {"id": "P54", "title": "Cross-Subject Generalization for EEG Decoding: A Survey of Deep Learning Methods",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint / IOP journal — check for published version"},
    {"id": "P55", "title": "A Survey on Deep Learning-Based Short Zero-Calibration Approaches for EEG-Based Brain-Computer Interfaces",
     "first_author": "Fahimi", "year": "2021", "venue": "Frontiers in Human Neuroscience", "doi": None},
    {"id": "P56", "title": "A Review on Signal Processing Approaches to Reduce Calibration Time in EEG-Based Brain-Computer Interface",
     "first_author": "Unknown", "year": "2021", "venue": "Frontiers in Neuroscience", "doi": None},
    {"id": "P57", "title": "Harnessing Few-Shot Learning for EEG Signal Classification: A Survey of State-of-the-Art Techniques",
     "first_author": "Unknown", "year": "2024", "venue": "Frontiers in Human Neuroscience", "doi": None},
    {"id": "P58", "title": "Generalizability of Machine Learning Models Applied to Electroencephalographic Recordings",
     "first_author": "Unknown", "year": "2026", "venue": "MSc Thesis Université Laval", "doi": None,
     "notes": "MSc Thesis — no published DOI"},
    {"id": "P59", "title": "Geometric-aware Interpretability and Understanding of EEG Spectral Dynamics",
     "first_author": "Unknown", "year": "2026", "venue": "PhD Thesis Università degli Studi di Verona", "doi": None,
     "notes": "PhD Thesis — obfuscated title, no DOI"},
    {"id": "P60", "title": "Geometric Representation Learning on Riemannian Manifolds for Multidimensional Signal Decoding and Modeling",
     "first_author": "Unknown", "year": "2026", "venue": "PhD Thesis Sapienza University of Rome", "doi": None,
     "notes": "PhD Thesis — no DOI"},
    {"id": "P61", "title": "Non-Invasive Brain-Computer Interfaces: Converging Frontiers in Neural Signal Decoding and Flexible Bioelectronics Integration",
     "first_author": "Unknown", "year": "2026", "venue": "Nano-Micro Letters", "doi": None},
    {"id": "P62", "title": "Feature and Classification Analyses of the EEG Recorded During the Perception and Production of Emotional Vocalisations",
     "first_author": "Unknown", "year": "2026", "venue": "PhD Thesis University of Auckland", "doi": None,
     "notes": "PhD Thesis — no DOI"},
    {"id": "P63", "title": "Classification of MI EEG Signal Using an Advanced Deep Learning Architecture for a Lower-Limb Rehabilitation Exoskeleton",
     "first_author": "Unknown", "year": "2025", "venue": "Technical Report University of West Bohemia", "doi": None,
     "notes": "Technical Report — no DOI"},
    {"id": "P64", "title": "Accessible and Explainable AI for EEG Decoding in Brain-Computer Interfaces",
     "first_author": "Unknown", "year": "2025", "venue": "PhD Thesis Politecnico di Bari", "doi": None,
     "notes": "PhD Thesis — no DOI"},
    {"id": "P65", "title": "A Riemannian Network for SPD Matrix Learning",
     "first_author": "Huang", "year": "2017", "venue": "AAAI 2017", "doi": None,
     "notes": "SPDNet — foundational paper"},
    {"id": "P66", "title": "ARMAGNAC: A New Parametric Batch Normalization Layer for SPDNet Architecture",
     "first_author": "Unknown", "year": "2026", "venue": "HAL preprint", "doi": None,
     "notes": "HAL preprint — no CrossRef DOI likely"},
    {"id": "P67", "title": "FedSPDnet: Geometry-Aware Federated Deep Learning with SPDnet",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P68", "title": "Riemannian Networks over Full-Rank Correlation Matrices",
     "first_author": "Chen", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint — Jiangnan University / Tübingen"},
    {"id": "P69", "title": "SPD Matrix Learning for Neuroimaging Analysis: Perspectives, Methods, and Challenges",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv survey", "doi": None,
     "notes": "arXiv survey — Inria/RIKEN/NTU"},
    {"id": "P70", "title": "Res2SPDNet: Multi-Granularity SPD Matrix Residual Learning for Signal Classification",
     "first_author": "Unknown", "year": "2026", "venue": "CVPR 2026", "doi": None},
    {"id": "P71", "title": "Geo-Mamba: Dual-Path Riemannian State Space Models for Functional Dynamics",
     "first_author": "Unknown", "year": "2026", "venue": "ICLR 2026 under review", "doi": None,
     "notes": "ICLR 2026 under review — not yet published"},
    {"id": "P72", "title": "A Unified SPD Token Transformer Framework for EEG Classification: Systematic Comparison of Geometric Embeddings",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P73", "title": "Towards a General Attention Framework on Gyrovector Spaces for Matrix Manifolds",
     "first_author": "Unknown", "year": "2025", "venue": "NeurIPS 2025", "doi": None},
    {"id": "P74", "title": "A Robust Multi-Branch CNN-LSTM Architecture for Cross-Subject Motor Imagery Classification",
     "first_author": "Unknown", "year": "2026", "venue": "MDPI", "doi": None},
    {"id": "P75", "title": "SPDLearn: A Geometric Deep Learning Python Library for Neural Decoding Through Trivialization",
     "first_author": "Unknown", "year": "2026", "venue": "Unknown software paper", "doi": None,
     "notes": "Software paper — venue unknown, search GitHub/arXiv"},
    {"id": "P76", "title": "Channel Adaptation for EEG Foundation Models: A Systematic Benchmark",
     "first_author": "Unknown", "year": "2026", "venue": "Unknown benchmark paper", "doi": None},
    {"id": "P77", "title": "EEG-Deformer: A Dense Convolutional Transformer for Brain-Computer Interfaces",
     "first_author": "Unknown", "year": "2026", "venue": "Unknown", "doi": None},
    {"id": "P78", "title": "NeuroPhysNet: A FitzHugh-Nagumo-Based Physics-Informed Neural Network for EEG Analysis and Motor Imagery Classification",
     "first_author": "Unknown", "year": "2026", "venue": "Unknown", "doi": None},
    {"id": "P79", "title": "ReManNet: A Riemannian Manifold Network for Monocular 3D Lane Detection",
     "first_author": "Unknown", "year": "2026", "venue": "Unknown", "doi": None},
    {"id": "P80", "title": "Latte: Hyperbolic Lorentz Attention for Joint-Subject EEG Classification",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P81", "title": "HEEGNet: Hyperbolic Embeddings for EEG",
     "first_author": "Unknown", "year": "2026", "venue": "ICLR 2026 under review", "doi": None,
     "notes": "ICLR 2026 under review — not yet published"},
    {"id": "P82", "title": "EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P83", "title": "Robust Hyperbolic Learning with Curvature-Aware Optimization",
     "first_author": "Unknown", "year": "2025", "venue": "NeurIPS 2025", "doi": None},
    {"id": "P84", "title": "Beyond Rigid Geometries: The Spline-Pullback Metric for Universal Diffeomorphic SPD Representation Learning",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P85", "title": "Riemannian Adversarial Attacks on Symmetric Positive Definite Matrices",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P86", "title": "Riemannian Block SPD Coupling Manifold and Its Application to Optimal Transport",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P87", "title": "Sheaf Neural Networks on SPD Manifolds: Second-Order Geometric Representation Learning",
     "first_author": "Unknown", "year": "2026", "venue": "Unknown", "doi": None},
    {"id": "P88", "title": "Riemannian Optimization over Symmetric Positive Definite Matrices with the Alpha-Procrustes Geometry",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P89", "title": "Fast and Stable Riemannian SPD Methods",
     "first_author": "Unknown", "year": "2026", "venue": "Unknown", "doi": None},
    {"id": "P90", "title": "Building Transformation Layers for SPDNet",
     "first_author": "Unknown", "year": "2026", "venue": "Unknown", "doi": None},
    {"id": "P91", "title": "Riemannian Diffusion Models on General Manifolds via Physics-Informed Neural Networks",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P92", "title": "A Riemannian Quasi-Newton Algorithm for Optimization with Euclidean Bounds",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P93", "title": "Multivariate Intrinsic Local Polynomial Regression on Isometric Riemannian Manifolds",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P94", "title": "Batch Normalization for Neural Networks on Complex Domains",
     "first_author": "Unknown", "year": "2026", "venue": "arXiv", "doi": None,
     "notes": "arXiv preprint"},
    {"id": "P95", "title": "EEG Cross-Subject Taste Classification Method: A Meta-Learning Wavelet Graph Convolutional Neural Network Under Sweet and Bitter Stimuli",
     "first_author": "Unknown", "year": "2026", "venue": "MDPI", "doi": None},
    {"id": "P96", "title": "NEED: Cross-Subject and Cross-Task Generalization for Video and Image Reconstruction from EEG Signals",
     "first_author": "Unknown", "year": "2025", "venue": "NeurIPS 2025", "doi": None},
    {"id": "P97", "title": "Outlier Detection for Riemannian Manifold-Valued Functional Data",
     "first_author": "Unknown", "year": "2026", "venue": "Applied Intelligence", "doi": None},
    {"id": "P98", "title": "SPD-DANN: An SPD manifold unsupervised domain adaptation method for cross subject motor imagery EEG decoding",
     "first_author": "Cheng", "year": "2026", "venue": "Neural Networks Elsevier", "doi": None,
     "notes": "SPD-DANN — PRIORITY paper"},
]

# Papers to go straight to to-verify (theses, technical reports, obfuscated titles)
THESIS_REPORT_IDS = {"P58", "P59", "P60", "P62", "P63", "P64"}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def jaccard(a: str, b: str) -> float:
    """Jaccard similarity on word sets."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def normalize_title(title: str) -> str:
    """Strip punctuation for similarity comparison."""
    return re.sub(r"[^\w\s]", " ", title.lower())


def make_key(first_author: str, year: str, title: str) -> str:
    """Generate BibTeX key: firstauthorYEARfirstword."""
    stopwords = {
        "a", "an", "the", "on", "of", "for", "in", "to", "and", "with",
        "via", "by", "from", "using", "based", "towards", "toward", "beyond",
        "over", "across", "through", "into", "onto", "at", "its",
    }
    author = re.sub(r"[^a-z]", "", first_author.lower())
    if not author or author == "unknown":
        author = "anon"
    words = re.sub(r"[^\w\s]", " ", title.lower()).split()
    word = next((w for w in words if w not in stopwords and len(w) > 2), words[0] if words else "paper")
    word = re.sub(r"[^a-z]", "", word)
    return f"{author}{year}{word}"


def http_get(url: str, headers: dict = None, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_crossref_doi(title: str, first_author: str, year: str) -> str | None:
    """Search CrossRef and return best-matching DOI."""
    q = urllib.parse.quote(title[:200])
    url = (
        f"https://api.crossref.org/works"
        f"?query.bibliographic={q}"
        f"&rows=5"
        f"&mailto={MAILTO}"
    )
    try:
        data = json.loads(http_get(url))
        items = data.get("message", {}).get("items", [])
        for item in items:
            item_title = " ".join(item.get("title", [""]))
            score = jaccard(normalize_title(title), normalize_title(item_title))
            if score >= 0.6:
                doi = item.get("DOI")
                if doi:
                    print(f"    CrossRef match ({score:.2f}): {item_title[:65]}")
                    return doi
    except Exception as e:
        print(f"    CrossRef error: {e}")
    return None


def fetch_openalex_doi(title: str) -> str | None:
    """Fallback: search OpenAlex for DOI."""
    q = urllib.parse.quote(title[:150])
    url = f"https://api.openalex.org/works?filter=title.search:{q}&per-page=5"
    try:
        data = json.loads(http_get(url))
        works = data.get("results", [])
        for w in works:
            wt = w.get("title") or w.get("display_name") or ""
            score = jaccard(normalize_title(title), normalize_title(wt))
            if score >= 0.6:
                doi = (w.get("doi") or "").replace("https://doi.org/", "")
                if doi:
                    print(f"    OpenAlex match ({score:.2f}): {wt[:65]}")
                    return doi
    except Exception as e:
        print(f"    OpenAlex error: {e}")
    return None


def fetch_bibtex_from_doi(doi: str) -> str | None:
    """Fetch BibTeX from doi.org content negotiation."""
    url = f"https://doi.org/{doi}"
    try:
        data = http_get(url, headers={"Accept": "application/x-bibtex"})
        bib = data.decode("utf-8", errors="replace").strip()
        if bib.startswith("@"):
            return bib
    except Exception as e:
        print(f"    BibTeX fetch error for {doi}: {e}")
    return None


def replace_bibtex_key(bibtex: str, new_key: str) -> str:
    """Replace the existing BibTeX key with our standardized key."""
    return re.sub(r"^(@\w+\{)[^,]+", rf"\g<1>{new_key}", bibtex, count=1)


def extract_first_author_from_bibtex(bibtex: str) -> str | None:
    """Pull first author surname from BibTeX author field."""
    m = re.search(r"\bauthor\s*=\s*\{([^}]+)\}", bibtex, re.IGNORECASE)
    if not m:
        return None
    authors = m.group(1)
    # "Last, First and ..." or "First Last and ..."
    first = authors.split(" and ")[0].strip()
    if "," in first:
        surname = first.split(",")[0].strip()
    else:
        parts = first.split()
        surname = parts[-1] if parts else ""
    # Clean
    surname = re.sub(r"[^a-zA-Z]", "", surname)
    return surname.lower() if surname else None


def is_bad_doi(doi: str) -> bool:
    """Reject DOIs from preprint servers, review systems, and data repos."""
    bad_prefixes = [
        "10.5281/zenodo",          # Zenodo — usually preprints/data
        "10.17504/protocols.io",   # protocols.io
        "10.1101/",                # bioRxiv/medRxiv
        "10.21203/",               # Research Square
        "10.48550/arxiv",          # arXiv via DataCite
        # Note: 10.52202/ is NeurIPS proceedings — OK to keep
    ]
    # Also reject review DOIs (contain /review or /v1/review)
    if "/review" in doi.lower():
        return True
    doi_lower = doi.lower()
    for prefix in bad_prefixes:
        if doi_lower.startswith(prefix.lower()):
            return True
    return False


def is_skippable(paper: dict) -> tuple:
    """Return (True, reason) if this paper should go straight to to-verify."""
    pid = paper["id"]
    notes = paper.get("notes", "")
    venue = paper.get("venue", "")

    if pid in THESIS_REPORT_IDS:
        return True, "Thesis/technical report — no published DOI"
    lower_venue = venue.lower()
    lower_notes = notes.lower()
    if "hal preprint" in lower_notes or ("hal" in lower_venue and "preprint" in lower_venue):
        return True, "HAL preprint — search hal.science manually"
    if "under review" in lower_venue or "under review" in lower_notes:
        return True, "Under review — not yet published; no citable DOI"
    if "medrxiv" in lower_venue.replace("-", "").lower():
        return True, "medRxiv preprint — no stable DOI; cite with URL"
    if "research square preprint" in lower_venue:
        return True, "Research Square preprint — no stable DOI"
    if "cvpr workshop" in lower_venue:
        return True, "CVPR Workshop 2026 — may not be indexed in CrossRef yet"
    if "arxiv" in lower_notes or lower_venue.startswith("arxiv"):
        return True, "arXiv preprint — use arXiv ID; not in CrossRef"
    if "unknown" in lower_venue and "unknown" in (paper.get("first_author", "")).lower():
        return True, "Venue and author unknown — manual lookup required"
    return False, ""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    verified = []      # list of (paper, key, bibtex, doi)
    unverified = []    # list of (paper, reason)

    total = len(PAPERS)
    print(f"Processing {total} papers from Paper Summary.md\n")
    print("=" * 70)

    for i, paper in enumerate(PAPERS):
        pid = paper["id"]
        title = paper["title"]
        author = paper["first_author"]
        year = paper["year"]

        print(f"\n[{i+1:02d}/{total}] {pid}: {title[:65]}...")

        # Quick-skip: preprints, theses, under-review papers
        skip, reason = is_skippable(paper)
        if skip:
            print(f"  -> SKIP ({reason})")
            unverified.append((paper, reason))
            continue

        # Use pre-existing DOI if provided in the database
        doi = paper.get("doi")

        # Step 1: CrossRef query
        if not doi:
            doi = fetch_crossref_doi(title, author, year)
            time.sleep(0.4)

        # Step 2: OpenAlex fallback
        if not doi:
            print(f"  -> CrossRef miss, trying OpenAlex...")
            doi = fetch_openalex_doi(title)
            time.sleep(0.3)

        if not doi:
            reason = "No DOI found via CrossRef or OpenAlex (best Jaccard < 0.6)"
            print(f"  -> UNVERIFIED: {reason}")
            unverified.append((paper, reason))
            continue

        # Reject bad DOIs (preprint servers, review systems, data repos)
        if is_bad_doi(doi):
            reason = f"DOI ({doi}) points to preprint/data repo — needs published DOI"
            print(f"  -> UNVERIFIED: {reason}")
            unverified.append((paper, reason))
            continue

        # Step 3: Fetch BibTeX
        bibtex = fetch_bibtex_from_doi(doi)
        time.sleep(0.2)

        if not bibtex:
            reason = f"DOI found ({doi}) but BibTeX content negotiation failed"
            print(f"  -> UNVERIFIED: {reason}")
            unverified.append((paper, f"{reason}; DOI={doi}"))
            continue

        # Step 4: Standardize BibTeX key using actual author name from BibTeX
        real_author = extract_first_author_from_bibtex(bibtex)
        key_author = real_author if real_author else re.sub(r"[^a-z]", "", author.lower())
        key = make_key(key_author, year, title)
        bibtex = replace_bibtex_key(bibtex, key)

        print(f"  -> VERIFIED: key={key}, DOI={doi}")
        verified.append((paper, key, bibtex, doi))

    print("\n" + "=" * 70)

    # Deduplicate keys — if two entries share a key, append _b, _c, etc.
    key_counts: dict = {}
    deduped = []
    for paper, key, bibtex, doi in verified:
        if key in key_counts:
            key_counts[key] += 1
            suffix = chr(ord("a") + key_counts[key])  # b, c, d...
            new_key = f"{key}{suffix}"
            bibtex = replace_bibtex_key(bibtex, new_key)
            deduped.append((paper, new_key, bibtex, doi))
        else:
            key_counts[key] = 0
            deduped.append((paper, key, bibtex, doi))
    verified = deduped

    print(f"Results: {len(verified)} verified, {len(unverified)} pending")

    # ---------------------------------------------------------------------------
    # Write refs.bib
    # ---------------------------------------------------------------------------
    date_str = "2026-06-15"
    header = f"""% ============================================================
%  IEEE TNSRE Paper — Verified Bibliography
%  Generated by scripts/build_bib.py from docs/Paper Summary.md
%  DO NOT EDIT BY HAND — re-run build_bib.py to update
%  Last updated: {date_str}
%  Verified entries: {len(verified)} | Pending verification: {len(unverified)}
% ============================================================

"""

    with open(REFS_BIB, "w", encoding="utf-8") as f:
        f.write(header)
        for paper, key, bibtex, doi in verified:
            f.write(f"% {paper['id']} — {paper['title'][:80]}\n")
            f.write(bibtex.strip())
            f.write("\n\n")

    print(f"\nWrote refs.bib: {REFS_BIB}")

    # ---------------------------------------------------------------------------
    # Write to-verify.txt
    # ---------------------------------------------------------------------------
    with open(TO_VERIFY, "w", encoding="utf-8") as f:
        f.write("# Papers requiring manual BibTeX entry\n")
        f.write("# For each: search Google Scholar or arXiv, get DOI/URL, then:\n")
        f.write('#   curl -LH "Accept: application/x-bibtex" https://doi.org/<DOI>\n')
        f.write(f"# Total unverified: {len(unverified)}\n")
        f.write("#\n\n")

        for paper, reason in unverified:
            pid = paper["id"]
            title = paper["title"]
            year = paper["year"]
            venue = paper["venue"]
            notes = paper.get("notes", "")
            f.write(f"{pid} — {title[:80]}\n")
            f.write(f"  Year: {year} | Venue: {venue}\n")
            if notes:
                f.write(f"  Notes: {notes}\n")
            f.write(f"  Reason: {reason}\n\n")

    print(f"Wrote to-verify.txt: {TO_VERIFY}")

    # ---------------------------------------------------------------------------
    # Summary of priority papers
    # ---------------------------------------------------------------------------
    priority_ids = {"P8", "P44", "P22", "P24", "P50", "P4", "P98", "P45", "P46", "P48"}
    print("\n--- Priority paper status ---")
    for paper, key, bibtex, doi in verified:
        if paper["id"] in priority_ids:
            print(f"  VERIFIED  {paper['id']}: key={key}")
    for paper, reason in unverified:
        if paper["id"] in priority_ids:
            print(f"  PENDING   {paper['id']}: {reason}")


if __name__ == "__main__":
    main()
