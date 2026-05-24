"""Spec ingestion module — extracts proof obligations from decorated Python source."""

from .parser import SpecParser
from .obligations import (
    ProofObligation,
    SpecCollection,
    ObligationKind,
    ObligationStatus,
)

__all__ = [
    "SpecParser", "ProofObligation", "SpecCollection",
    "ObligationKind", "ObligationStatus", "extract_specs",
]


def extract_specs(ir, abstract_state=None) -> SpecCollection:
    """Extract proof obligations from normalized IR."""
    return SpecParser().extract_specs(ir, abstract_state)
