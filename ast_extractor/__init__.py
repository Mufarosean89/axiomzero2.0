"""Axiom Zero - AST Extractor Module

Parses Python source code into a normalized intermediate representation (IR).
Handles function signatures, class definitions, tensor operations, and type annotations.

Public API:
    - parse_file(path) -> NormalizedIR: Parse a Python file
    - parse_source(source, name) -> NormalizedIR: Parse a source string
    - normalize(ir) -> NormalizedIR: Normalize/desugar the IR
    - to_ssa(ir) -> NormalizedIR: Convert to SSA form
"""

from __future__ import annotations

from typing import Optional

from .ir import (
    NormalizedIR,
    FunctionIR,
    FunctionSignature,
    ParameterIR,
    ClassIR,
    LoopIR,
    ConditionalIR,
    StatementIR,
    ExpressionIR,
    TypeAnnotation,
    TensorOpKind,
    ParserWarning,
    ParserConfig,
)
from .parser import Parser, ParserError
from .normalizer import Normalizer, SSAConverter

__all__ = [
    "NormalizedIR",
    "FunctionIR",
    "FunctionSignature",
    "ParameterIR",
    "ClassIR",
    "LoopIR",
    "ConditionalIR",
    "StatementIR",
    "ExpressionIR",
    "TypeAnnotation",
    "TensorOpKind",
    "ParserWarning",
    "ParserConfig",
    "Parser",
    "ParserError",
    "Normalizer",
    "SSAConverter",
    "parse_file",
    "parse_source",
    "normalize",
    "to_ssa",
]


def parse_file(path: str, config: Optional[ParserConfig] = None) -> NormalizedIR:
    """Parse a Python source file into NormalizedIR."""
    return Parser(config=config).parse(path)


def parse_source(source: str, module_name: str = "<string>", config: Optional[ParserConfig] = None) -> NormalizedIR:
    """Parse Python source code string into NormalizedIR."""
    return Parser(config=config).parse_source(source, module_name)


def normalize(ir: NormalizedIR) -> NormalizedIR:
    """Normalize/desugar the IR."""
    return Normalizer().normalize(ir)


def to_ssa(ir: NormalizedIR) -> NormalizedIR:
    """Convert IR to Static Single Assignment form."""
    return SSAConverter().to_ssa(ir)
