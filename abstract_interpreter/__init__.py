"""
Axiom Zero - Abstract Interpreter Module

Performs symbolic type inference and tensor shape analysis over normalized IR.
Tracks data flow and generates background facts for the proof state.

Public API:
    - analyze(ir) -> AbstractState: Run full abstract interpretation
    - AbstractState: Complete analysis results
    - AbstractValue: Type + shape at a program point
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
    """Run abstract interpretation over normalized IR.

    Args:
        ir: NormalizedIR from AST extraction

    Returns:
        AbstractState with inferred types, shapes, and data flow facts
    """
    interpreter = AbstractInterpreter()
    return interpreter.analyze(ir)
