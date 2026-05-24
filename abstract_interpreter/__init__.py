"""Axiom Zero - Abstract Interpreter Module

Performs symbolic type inference and tensor shape analysis over normalized IR.
Tracks data flow and generates background facts for the proof state.
"""

from .interpreter import AbstractInterpreter
from .abstract_domain import (
    AbstractState,
    AbstractValue,
    TypeDomain,
    TensorShape,
    ShapeDimension,
)
from .type_inference import TypeInferenceEngine
from .shape_analysis import TensorShapeAnalyzer

__all__ = [
    "AbstractInterpreter",
    "AbstractState",
    "AbstractValue",
    "TypeDomain",
    "TensorShape",
    "ShapeDimension",
    "TypeInferenceEngine",
    "TensorShapeAnalyzer",
    "analyze",
]


def analyze(ir) -> AbstractState:
    """Run abstract interpretation over normalized IR."""
    return AbstractInterpreter().analyze(ir)
