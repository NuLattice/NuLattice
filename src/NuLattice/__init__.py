import sys
# from . import lattice_soa
from ._types import (
    OneBodyOperator,
    TwoBodyOperator,
    ThreeBodyOperator,
)

# sys.modules["NuLattice.lattice"] = lattice_soa
__all__ = ["OneBodyOperator", "TwoBodyOperator", "ThreeBodyOperator"]


