"""Week 3 WDANet: Wasserstein Transform Calculator (Sinkhorn) + integration losses."""

from .wtc import pairwise_cost_matrix, sinkhorn_wtc, transport_matrices
from .wasserstein import wasserstein_distribution_loss
from .integration import compute_wdanet_objective

__all__ = [
    "pairwise_cost_matrix",
    "sinkhorn_wtc",
    "transport_matrices",
    "wasserstein_distribution_loss",
    "compute_wdanet_objective",
]
