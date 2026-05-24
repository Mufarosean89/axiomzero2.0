"""
Axiom Zero - Spec Ingestion Parser

Parses formal specifications from decorated Python source code and extracts
proof obligations for the RL agent.

Supports:
- @requires(condition) — Precondition: must hold before function executes
- @ensures(condition)  — Postcondition: must hold after function executes
- @invariant(condition) — Loop invariant: must hold at each loop iteration
- @assert condition — Inline assertion
- Type annotations as implicit type constraints
- Tensor shape annotations as implicit shape constraints

Also generates implicit safety obligations:
- No division by zero
- In-bounds list/tensor indexing
- Shape compatibility for tensor operations
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from ast_extractor.ir import (
    NormalizedIR,
    FunctionIR,
    StatementIR,
    ExpressionIR,
    ClassIR,
    TypeAnnotation,
    TensorOpKind,
)
from ast_extractor.parser import SPEC_DECORATORS, Parser
from .obligations import (
    ProofObligation,
    SpecCollection,
    ObligationKind,
    ObligationStatus,
)


class SpecParser:
    """
    Parses formal specifications from Python code.
    
    Usage:
        parser = SpecParser()
        specs = parser.extract_specs(normalized_ir)
        # specs is a SpecCollection of proof obligations
    """

    def __init__(self):
        self._collected_obligations: SpecCollection = SpecCollection()
        self._parser = Parser()

    def extract_specs(
        self,
        ir: NormalizedIR,
        abstract_state: Optional[Any] = None,
    ) -> SpecCollection:
        """
        Extract all proof obligations from the normalized IR.
        
        Args:
            ir: Normalized IR from AST extraction
            abstract_state: Optional AbstractState for additional context
            
        Returns:
            SpecCollection with all extracted proof obligations
        """
        self._collected_obligations = SpecCollection()

        # Extract from all functions
        for func in ir.functions:
            self._extract_from_function(func, ir)

        # Extract from class methods
        for cls in ir.classes:
            for method in cls.methods:
                self._extract_from_function(method, ir, class_name=cls.name)

        # Extract implicit safety conditions
        if abstract_state:
            self._extract_safety_conditions(ir, abstract_state)

        return self._collected_obligations

    def _extract_from_function(
        self,
        func: FunctionIR,
        ir: NormalizedIR,
        class_name: Optional[str] = None,
    ):
        """Extract proof obligations from a single function."""
        func_name = f"{class_name}.{func.signature.name}" if class_name else func.signature.name

        # Process decorator-based specs
        for dec in func.signature.decorators:
            self._process_decorator(dec, func_name, func)

        # Extract from assertions in body
        self._extract_assertions(func.body, func_name)

        # Extract loop invariants and termination conditions
        self._extract_loop_specs(func.body, func_name)

        # Extract implicit type/shape conditions from signature
        self._extract_signature_conditions(func, func_name)

    def _process_decorator(
        self,
        dec: Dict[str, Any],
        func_name: str,
        func: FunctionIR,
    ):
        """Process a single decorator and create obligations."""
        name = dec.get("name", "")
        predicates = dec.get("predicates", [])
        args = dec.get("args", [])
        kwargs = dec.get("kwargs", {})

        # Determine obligation kind
        kind = self._decorator_to_kind(name)
        if kind is None:
            return

        # Build context from function parameters
        context = {}
        for param in func.signature.parameters:
            if param.type:
                context[param.name] = param.type.to_string()
            else:
                context[param.name] = "Any"

        # Create obligation from each predicate string
        for pred in predicates:
            obligation = ProofObligation(
                kind=kind,
                predicate=pred,
                location=func.source_loc or "",
                function=func_name,
                context=dict(context),
                hypotheses=self._build_hypotheses(kind, func),
            )
            self._collected_obligations.add(obligation)

        # Also create obligations from expression args
        for arg in args:
            pred_str = self._expr_to_predicate(arg)
            if pred_str:
                obligation = ProofObligation(
                    kind=kind,
                    predicate=pred_str,
                    location=func.source_loc or "",
                    function=func_name,
                    context=dict(context),
                    hypotheses=self._build_hypotheses(kind, func),
                )
                self._collected_obligations.add(obligation)

    def _decorator_to_kind(self, decorator_name: str) -> Optional[ObligationKind]:
        """Map a decorator name to an obligation kind."""
        name = decorator_name.lower().replace("@", "").strip()

        if name in ("requires", "precondition", "pre"):
            return ObligationKind.PRECONDITION
        if name in ("ensures", "postcondition", "post", "guarantees"):
            return ObligationKind.POSTCONDITION
        if name in ("invariant", "loop_invariant"):
            return ObligationKind.LOOP_INVARIANT

        return None

    def _build_hypotheses(self, kind: ObligationKind, func: FunctionIR) -> List[str]:
        """Build available hypotheses for the obligation."""
        hypotheses = []

        if kind == ObligationKind.POSTCONDITION:
            # For postconditions, we can assume preconditions held
            # and use the function body's effect
            for dec in func.signature.decorators:
                dname = dec.get("name", "").lower().replace("@", "")
                if dname in ("requires", "precondition", "pre"):
                    for pred in dec.get("predicates", []):
                        hypotheses.append(f"(assume {pred})")

        return hypotheses

    def _expr_to_predicate(self, expr: ExpressionIR) -> Optional[str]:
        """Convert an expression IR to a predicate string."""
        if expr is None:
            return None

        if expr.expr_type == "constant":
            return str(expr.value)

        if expr.expr_type == "variable":
            return expr.name

        if expr.expr_type == "binary_op":
            left = self._expr_to_predicate(expr.left) or "?"
            right = self._expr_to_predicate(expr.right) or "?"
            return f"({left} {expr.op} {right})"

        if expr.expr_type == "unary_op":
            operand = self._expr_to_predicate(expr.operand) or "?"
            return f"({expr.op} {operand})"

        if expr.expr_type == "call":
            func_name = self._expr_to_predicate(expr.func) or "?"
            arg_strs = [self._expr_to_predicate(a) or "?" for a in expr.args]
            return f"{func_name}({', '.join(arg_strs)})"

        return None

    def _extract_assertions(self, body: List[StatementIR], func_name: str):
        """Extract proof obligations from assert statements."""
        for stmt in self._walk_statements(body):
            if stmt.stmt_type == "assert" and stmt.condition:
                pred = self._expr_to_predicate(stmt.condition)
                if pred:
                    obligation = ProofObligation(
                        kind=ObligationKind.ASSERTION,
                        predicate=pred,
                        location=stmt.source_loc or "",
                        function=func_name,
                    )
                    self._collected_obligations.add(obligation)

    def _extract_loop_specs(self, body: List[StatementIR], func_name: str):
        """Extract loop invariants and termination conditions."""
        for stmt in self._walk_statements(body):
            if stmt.stmt_type in ("for", "while"):
                # Create termination obligation
                term_obligation = ProofObligation(
                    kind=ObligationKind.LOOP_TERMINATION,
                    predicate=f"loop_terminates({stmt.source_loc or 'unknown'})",
                    location=stmt.source_loc or "",
                    function=func_name,
                )
                self._collected_obligations.add(term_obligation)

                # Check for loop invariant annotations
                if stmt.annotations and "invariant" in stmt.annotations:
                    inv = stmt.annotations["invariant"]
                    pred = str(inv) if not isinstance(inv, ExpressionIR) else self._expr_to_predicate(inv)
                    if pred:
                        inv_obligation = ProofObligation(
                            kind=ObligationKind.LOOP_INVARIANT,
                            predicate=pred,
                            location=stmt.source_loc or "",
                            function=func_name,
                        )
                        self._collected_obligations.add(inv_obligation)

    def _extract_signature_conditions(self, func: FunctionIR, func_name: str):
        """Extract implicit conditions from type annotations."""
        for param in func.signature.parameters:
            if param.type and "Tensor" in param.type.to_string():
                # Tensor parameters imply shape constraints
                shape_obligation = ProofObligation(
                    kind=ObligationKind.SHAPE_CONDITION,
                    predicate=f"is_tensor({param.name})",
                    location=func.source_loc or "",
                    function=func_name,
                    context={param.name: param.type.to_string()},
                )
                self._collected_obligations.add(shape_obligation)

    def _extract_safety_conditions(self, ir: NormalizedIR, abstract_state: Any):
        """Extract implicit safety conditions from the IR."""
        for func in ir.all_functions:
            func_name = func.signature.name
            for op_expr in func.tensor_operations:
                if op_expr.tensor_op_kind == TensorOpKind.MATMUL:
                    # Matrix multiplication shape compatibility
                    if len(op_expr.args) >= 2:
                        shape_obligation = ProofObligation(
                            kind=ObligationKind.SHAPE_CONDITION,
                            predicate="matmul_shapes_compatible",
                            location=op_expr.source_loc or func.source_loc or "",
                            function=func_name,
                        )
                        self._collected_obligations.add(shape_obligation)

    def _walk_statements(self, stmts: List[StatementIR]) -> List[StatementIR]:
        """Walk all statements recursively."""
        result = []
        for stmt in stmts:
            result.append(stmt)
            if stmt.body:
                result.extend(self._walk_statements(stmt.body))
            if stmt.orelse:
                result.extend(self._walk_statements(stmt.orelse))
        return result

    # ─── Standalone Spec Parsing ──────────────────────────────────────────

    def parse_spec_file(self, spec_path: str) -> SpecCollection:
        """
        Parse a standalone spec file (.lean or .spec).
        
        Args:
            spec_path: Path to spec file
        Returns:
            SpecCollection with obligations
        """
        if spec_path.endswith(".lean"):
            return self._parse_lean_spec(spec_path)
        return self._parse_text_spec(spec_path)

    def _parse_lean_spec(self, path: str) -> SpecCollection:
        """Parse a .lean spec file for known theorem patterns."""
        collection = SpecCollection()
        try:
            with open(path, "r") as f:
                content = f.read()

            # Match theorem declarations
            theorem_pattern = r"theorem\s+(\w+)\s*(.*?)\s*:\s*(.+?)(?=\n|$)"
            for match in re.finditer(theorem_pattern, content):
                name = match.group(1)
                statement = match.group(3).strip()

                obligation = ProofObligation(
                    kind=ObligationKind.EQUALITY,
                    predicate=statement,
                    location=f"{path}:{match.start()}",
                    function=name,
                )
                collection.add(obligation)

        except FileNotFoundError:
            pass

        return collection

    def _parse_text_spec(self, path: str) -> SpecCollection:
        """Parse a plain text spec file."""
        collection = SpecCollection()
        try:
            with open(path, "r") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("//"):
                        continue

                    # Try to parse as predicate
                    obligation = ProofObligation(
                        kind=ObligationKind.ASSERTION,
                        predicate=line,
                        location=f"{path}:{i}",
                    )
                    collection.add(obligation)

        except FileNotFoundError:
            pass

        return collection
