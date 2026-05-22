"""
Axiom Zero - Spec Ingestion Module

Parses formal specifications from decorated Python source and extracts
proof obligations. Supports @requires, @ensures, @invariant decorators
and inline assertions.

Public API:
    - extract_specs(ir, state) -> SpecCollection: Extract all proof obligations
    - SpecCollection: Collection of proof obligations
    - ProofObligation: A single thing to prove
"""

from .parser import SpecParser
from .obligations import (
    ProofObligation,
    SpecCollection,
    ObligationKind,
    ObligationStatus,
)

__all__ = [
    "SpecParser",
    "ProofObligation",
    "SpecCollection",
    "ObligationKind",
    "ObligationStatus",
    "extract_specs",
]


def extract_specs(ir, abstract_state=None) -> SpecCollection:
    """Extract proof obligations from normalized IR.

    Args:
        ir: NormalizedIR from AST extraction
        abstract_state: Optional AbstractState for context

    Returns:
        SpecCollection with all extracted proof obligations
    """
    parser = SpecParser()
    return parser.extract_specs(ir, abstract_state)
