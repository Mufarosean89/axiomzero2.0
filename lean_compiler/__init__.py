"""Axiom Zero — Phase 4: Python -> Lean Compiler

Translates Python source code (via the existing IR pipeline) into Lean 4
theorems and proofs. Bridges Phases 1-3 into a working compiler that
produces verifiable Lean 4 output.
"""

from __future__ import annotations

from typing import List, Optional

from ast_extractor.ir import NormalizedIR
from abstract_interpreter.abstract_domain import AbstractState
from spec_ingestion.obligations import SpecCollection

from .ir_to_lean import IRToLeanCompiler
from .hole_filler import HoleFiller, HoleDifficulty
from .benchmark_suite import BenchmarkSuite, BenchmarkProblem


__all__ = [
    "compile",
    "IRToLeanCompiler",
    "HoleFiller",
    "HoleDifficulty",
    "BenchmarkSuite",
    "BenchmarkProblem",
]


def compile(
    source: str,
    module_name: str = "target",
    ir: Optional[NormalizedIR] = None,
    abstract_state: Optional[AbstractState] = None,
    specs: Optional[SpecCollection] = None,
    fill_holes: bool = True,
    module_doc: str = "",
) -> str:
    """End-to-end compilation: Python source -> Lean 4 module string.

    Uses the existing Phase 1-2 pipeline (AST extractor, abstract interpreter,
    spec ingestion) to produce a complete Lean 4 .lean file from annotated
    Python source.

    Args:
        source: Python source code with @requires/@ensures decorators.
        module_name: Name for the generated Lean module.
        ir: Pre-parsed IR (if already available). Parsed from source if None.
        abstract_state: Pre-analysed abstract state (computed from ir if None).
        specs: Pre-extracted spec collection (computed from ir if None).
        fill_holes: If True, attempt to fill proof holes; otherwise leave as sorry.
        module_doc: Optional doc comment at the top of the generated file.
    """
    from ast_extractor import parse_source, normalize
    from abstract_interpreter import analyze
    from spec_ingestion import extract_specs

    if ir is None:
        ir = parse_source(source, module_name)
        ir = normalize(ir)
    if abstract_state is None:
        abstract_state = analyze(ir)
    if specs is None:
        specs = extract_specs(ir, abstract_state)

    compiler = IRToLeanCompiler()
    result = compiler.compile_module(ir, specs, abstract_state, module_name, module_doc)

    if fill_holes:
        filler = HoleFiller()
        result = filler.fill_all_holes(result, specs)

    return result
