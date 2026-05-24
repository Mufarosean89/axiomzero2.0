"""Parses formal specifications from decorated Python source into proof obligations."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from ast_extractor.ir import (
    NormalizedIR, FunctionIR, StatementIR, ExpressionIR, TensorOpKind,
)
from .obligations import (
    ProofObligation, SpecCollection, ObligationKind,
)


class SpecParser:
    """Parses formal specifications from Python code into proof obligations."""

    def __init__(self):
        self._collected_obligations: SpecCollection = SpecCollection()

    def extract_specs(
        self, ir: NormalizedIR, abstract_state: Optional[Any] = None,
    ) -> SpecCollection:
        self._collected_obligations = SpecCollection()
        for func in ir.functions:
            self._extract_from_function(func, ir)
        for cls in ir.classes:
            for method in cls.methods:
                self._extract_from_function(method, ir, class_name=cls.name)
        if abstract_state:
            self._extract_safety_conditions(ir, abstract_state)
        return self._collected_obligations

    def _extract_from_function(
        self, func: FunctionIR, ir: NormalizedIR, class_name: Optional[str] = None,
    ):
        func_name = f"{class_name}.{func.signature.name}" if class_name else func.signature.name
        for dec in func.signature.decorators:
            self._process_decorator(dec, func_name, func)
        self._extract_assertions(func.body, func_name)
        self._extract_loop_specs(func.body, func_name)
        self._extract_signature_conditions(func, func_name)

    def _process_decorator(self, dec: Dict[str, Any], func_name: str, func: FunctionIR):
        kind = self._decorator_to_kind(dec.get("name", ""))
        if kind is None:
            return

        context = {}
        for param in func.signature.parameters:
            context[param.name] = param.type.to_string() if param.type else "Any"

        if kind == ObligationKind.POSTCONDITION:
            context["result"] = func.signature.return_type.to_string() if func.signature.return_type else "Any"

        for pred in dec.get("predicates", []):
            self._collected_obligations.add(ProofObligation(
                kind=kind, predicate=pred, location=func.source_loc or "",
                function=func_name, context=dict(context),
                hypotheses=self._build_hypotheses(kind, func),
            ))
        # Only process expression args when there are no string predicates,
        # to avoid duplicates (string-based decorators populate both fields).
        if not dec.get("predicates"):
            for arg in dec.get("args", []):
                pred_str = self._expr_to_predicate(arg)
                if pred_str:
                    self._collected_obligations.add(ProofObligation(
                        kind=kind, predicate=pred_str, location=func.source_loc or "",
                        function=func_name, context=dict(context),
                        hypotheses=self._build_hypotheses(kind, func),
                    ))

    @staticmethod
    def _decorator_to_kind(decorator_name: str) -> Optional[ObligationKind]:
        name = decorator_name.lower().replace("@", "").strip()
        if name in ("requires", "precondition", "pre"):
            return ObligationKind.PRECONDITION
        if name in ("ensures", "postcondition", "post", "guarantees"):
            return ObligationKind.POSTCONDITION
        if name in ("invariant", "loop_invariant"):
            return ObligationKind.LOOP_INVARIANT
        return None

    @staticmethod
    def _build_hypotheses(kind: ObligationKind, func: FunctionIR) -> List[str]:
        hypotheses = []
        if kind == ObligationKind.POSTCONDITION:
            for dec in func.signature.decorators:
                dname = dec.get("name", "").lower().replace("@", "")
                if dname in ("requires", "precondition", "pre"):
                    for pred in dec.get("predicates", []):
                        hypotheses.append(f"(assume {pred})")
                    # Only fall through to args if no string predicates,
                    # to avoid duplicates (string decorators populate both fields)
                    if not dec.get("predicates"):
                        for arg in dec.get("args", []):
                            pred_str = SpecParser._expr_to_predicate(arg)
                            if pred_str:
                                hypotheses.append(f"(assume {pred_str})")
        return hypotheses

    @staticmethod
    def _expr_to_predicate(expr: Optional[ExpressionIR]) -> Optional[str]:
        if expr is None:
            return None
        if expr.expr_type == "constant":
            return str(expr.value)
        if expr.expr_type == "variable":
            return expr.name
        if expr.expr_type == "binary_op":
            left = SpecParser._expr_to_predicate(expr.left) or "?"
            right = SpecParser._expr_to_predicate(expr.right) or "?"
            return f"({left} {expr.op} {right})"
        if expr.expr_type == "unary_op":
            operand = SpecParser._expr_to_predicate(expr.operand) or "?"
            return f"({expr.op} {operand})"
        if expr.expr_type == "lambda":
            return SpecParser._expr_to_predicate(expr.value)
        if expr.expr_type == "call":
            func_name = SpecParser._expr_to_predicate(expr.func) or "?"
            arg_strs = [SpecParser._expr_to_predicate(a) or "?" for a in expr.args]
            return f"{func_name}({', '.join(arg_strs)})"
        return None

    def _extract_assertions(self, body: List[StatementIR], func_name: str):
        for stmt in self._walk_statements(body):
            if stmt.stmt_type == "assert" and stmt.condition:
                pred = self._expr_to_predicate(stmt.condition)
                if pred:
                    self._collected_obligations.add(ProofObligation(
                        kind=ObligationKind.ASSERTION, predicate=pred,
                        location=stmt.source_loc or "", function=func_name,
                    ))

    def _extract_loop_specs(self, body: List[StatementIR], func_name: str):
        for stmt in self._walk_statements(body):
            if stmt.stmt_type in ("for", "while"):
                self._collected_obligations.add(ProofObligation(
                    kind=ObligationKind.LOOP_TERMINATION,
                    predicate=f"loop_terminates({stmt.source_loc or 'unknown'})",
                    location=stmt.source_loc or "", function=func_name,
                ))
                if stmt.annotations and "invariant" in stmt.annotations:
                    inv = stmt.annotations["invariant"]
                    pred = str(inv) if not isinstance(inv, ExpressionIR) else self._expr_to_predicate(inv)
                    if pred:
                        self._collected_obligations.add(ProofObligation(
                            kind=ObligationKind.LOOP_INVARIANT, predicate=pred,
                            location=stmt.source_loc or "", function=func_name,
                        ))

    def _extract_signature_conditions(self, func: FunctionIR, func_name: str):
        for param in func.signature.parameters:
            if param.type and "Tensor" in param.type.to_string():
                self._collected_obligations.add(ProofObligation(
                    kind=ObligationKind.SHAPE_CONDITION,
                    predicate=f"is_tensor({param.name})",
                    location=func.source_loc or "", function=func_name,
                    context={param.name: param.type.to_string()},
                ))

    def _extract_safety_conditions(self, ir: NormalizedIR, abstract_state: Any):
        for func in ir.all_functions:
            func_name = func.signature.name
            for op_expr in func.tensor_operations:
                if op_expr.tensor_op_kind == TensorOpKind.MATMUL and len(op_expr.args) >= 2:
                    self._collected_obligations.add(ProofObligation(
                        kind=ObligationKind.SHAPE_CONDITION,
                        predicate="matmul_shapes_compatible",
                        location=op_expr.source_loc or func.source_loc or "",
                        function=func_name,
                    ))

    @staticmethod
    def _walk_statements(stmts: List[StatementIR]) -> List[StatementIR]:
        result = []
        for stmt in stmts:
            result.append(stmt)
            if stmt.body:
                result.extend(SpecParser._walk_statements(stmt.body))
            if stmt.orelse:
                result.extend(SpecParser._walk_statements(stmt.orelse))
        return result
    # ─── Standalone Spec File Parsing ────────────────────────────────────

    def parse_spec_file(self, spec_path: str) -> SpecCollection:
        if spec_path.endswith(".lean"):
            return self._parse_lean_spec(spec_path)
        return self._parse_text_spec(spec_path)

    def _parse_lean_spec(self, path: str) -> SpecCollection:
        collection = SpecCollection()
        try:
            with open(path) as f:
                content = f.read()
            for match in re.finditer(r"theorem\s+(\w+)\s*(.*?)\s*:\s*(.+?)(?=\n|$)", content):
                collection.add(ProofObligation(
                    kind=ObligationKind.EQUALITY,
                    predicate=match.group(3).strip(),
                    location=f"{path}:{match.start()}",
                    function=match.group(1),
                ))
        except FileNotFoundError:
            pass
        return collection

    def _parse_text_spec(self, path: str) -> SpecCollection:
        collection = SpecCollection()
        try:
            with open(path) as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith(("#", "//")):
                        continue
                    collection.add(ProofObligation(
                        kind=ObligationKind.ASSERTION,
                        predicate=line, location=f"{path}:{i}",
                    ))
        except FileNotFoundError:
            pass
        return collection
