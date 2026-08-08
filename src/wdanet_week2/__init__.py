"""Week 2 WDANet modules: model blocks, GRL, and core losses."""

from .grl import GradientReversal, grad_reverse
from .models import (
    FeatureExtractor,
    LabelClassifier,
    GlobalDomainDiscriminator,
    LocalDomainDiscriminator,
    LocalDomainDiscriminatorBank,
)
from .losses import (
    label_classification_loss,
    global_domain_loss,
    local_domain_loss,
    dynamic_adversarial_factor,
)

__all__ = [
    "GradientReversal",
    "grad_reverse",
    "FeatureExtractor",
    "LabelClassifier",
    "GlobalDomainDiscriminator",
    "LocalDomainDiscriminator",
    "LocalDomainDiscriminatorBank",
    "label_classification_loss",
    "global_domain_loss",
    "local_domain_loss",
    "dynamic_adversarial_factor",
]
