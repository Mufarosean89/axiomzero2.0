"""
Axiom Zero - AST Extractor Module

Parses Python source code into a normalized intermediate representation (IR).
Handles function signatures, class definitions, tensor operations, and type annotations.

Public API:
    - parse_file(path) -> NormalizedIR: Parse a Python file
    - parse_source(source, name) -> NormalizedIR: Parse a source string
    - normalize(ir) -> NormalizedIR: Normalize/desugar the IR
    - to_ssa(ir) -> NormalizedIR: Convert to SSA form

Parsing can be configured with ParserConfig:
    - strict_mode: If True, unsupported constructs raise ParserError
    - collect_warnings: If True, warnings are collected in ir.warnings
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
    # Data structures
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
    # Parser
    "Parser",
    "ParserError",
    # Normalizer
    "Normalizer",
    "SSAConverter",
    # Convenience functions
    "parse_file",
    "parse_source",
    "normalize",
    "to_ssa",
]


def parse_file(path: str, config: Optional[ParserConfig] = None) -> NormalizedIR:
    """Parse a Python source file into NormalizedIR.

    Args:
        path: Path to the .py file
        config: Optional ParserConfig to control strict mode and warning collection

    Returns:
        NormalizedIR representation of the module
    """
    parser = Parser(config=config)
    return parser.parse(path)


def parse_source(source: str, module_name: str = "<string>", config: Optional[ParserConfig] = None) -> NormalizedIR:
    """Parse Python source code string into NormalizedIR.

    Args:
        source: Python source code as a string
        module_name: Name for the module (used in error messages)
        config: Optional ParserConfig to control strict mode and warning collection

    Returns:
        NormalizedIR representation
    """
    parser = Parser(config=config)
    return parser.parse_source(source, module_name)


def normalize(ir: NormalizedIR) -> NormalizedIR:
    """Normalize/desugar the IR."""
    normalizer = Normalizer()
    return normalizer.normalize(ir)


def to_ssa(ir: NormalizedIR) -> NormalizedIR:
    """Convert IR to Static Single Assignment form."""
    converter = SSAConverter()
    return converter.to_ssa(ir)
