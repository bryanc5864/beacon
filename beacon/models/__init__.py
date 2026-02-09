"""
BEACON Model Components

Binding Event Attention-based Compositional Object Network
"""

from .backbone import ConvBackbone, DilatedConvBackbone
from .positional import HelicalPositionalEncoding, SinusoidalPositionalEncoding
from .slot_attention import SlotAttention
from .heads import (
    PositionHead,
    TFIdentityHead,
    OccupancyHead,
    ProfileHead,
    AttributionHead,
    MotifEmbeddingHead,
)
from .beacon import BEACON, BEACONSmall, BEACONBase, BEACONLarge
from .losses import (
    BEACONLoss,
    ProfileReconstructionLoss,
    SlotDiversityLoss,
    SlotOrthogonalityLoss,
    BindingSiteSupervisionLoss,
    AnchorLoss,
    ImportanceSupervisionLoss,
    PrototypeDiversityLoss,
    ContrastiveTFEmbeddingLoss,
    PerTFDifficultyLoss,
    MultiScaleImportanceLoss,
)

__all__ = [
    "ConvBackbone",
    "DilatedConvBackbone",
    "HelicalPositionalEncoding",
    "SinusoidalPositionalEncoding",
    "SlotAttention",
    "PositionHead",
    "TFIdentityHead",
    "OccupancyHead",
    "ProfileHead",
    "AttributionHead",
    "MotifEmbeddingHead",
    "BEACON",
    "BEACONSmall",
    "BEACONBase",
    "BEACONLarge",
    "BEACONLoss",
    "ProfileReconstructionLoss",
    "SlotDiversityLoss",
    "SlotOrthogonalityLoss",
    "BindingSiteSupervisionLoss",
    "AnchorLoss",
    "ImportanceSupervisionLoss",
    "PrototypeDiversityLoss",
    "ContrastiveTFEmbeddingLoss",
    "PerTFDifficultyLoss",
    "MultiScaleImportanceLoss",
]
