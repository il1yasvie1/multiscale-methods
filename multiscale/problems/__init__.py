"""Problem data used by current multiscale examples."""

from .elliptic import (
    ConstantCoefficientProblem,
    EllipticProblemData,
    NonConstantCoefficientProblem,
    PeriodicCoefficientProblem,
    RandomCoefficientProblem,
)

__all__ = [
    "EllipticProblemData",
    "ConstantCoefficientProblem",
    "NonConstantCoefficientProblem",
    "PeriodicCoefficientProblem",
    "RandomCoefficientProblem",
]
