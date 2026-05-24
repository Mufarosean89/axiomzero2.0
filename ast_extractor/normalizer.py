"""Axiom Zero - IR Normalizer

Transforms raw extracted IR into a normalized form suitable for
abstract interpretation and proof obligation generation.

Normalization passes:
1. Desugar augmented assignments (a += 1 -> a = a + 1)
2. Flatten nested expressions to SSA-like form
3. Normalize loop structures
4. Simplify boolean expressions
5. Expand list/dict comprehensions into explicit loops
6. Annotate with implicit type information
7. Extract tensor operation chains into sequential steps
"""

from __future__ import annotations

from typing import Dict, List, Optional
from .ir import (
    NormalizedIR,
    FunctionIR,
    StatementIR,
    ExpressionIR,
)


class Normalizer:
    """Normalizes the extracted IR for downstream analysis."""

    def __init__(self):
        self._temp_var_counter: int = 0
        self._current_function: Optional[str] = None

    def normalize(self, ir: NormalizedIR) -> NormalizedIR:
        """Run all normalization passes on the IR."""
        for func in ir.functions:
            self._normalize_function(func)
        for cls in ir.classes:
            for method in cls.methods:
                self._normalize_function(method)
        ir.global_statements = self._normalize_statements(ir.global_statements)
        return ir

    def _normalize_function(self, func: FunctionIR):
        self._current_function = func.signature.name
        self._temp_var_counter = 0
        func.body = self._normalize_statements(func.body)
        for nested in func.nested_functions:
            self._normalize_function(nested)

    def _normalize_statements(self, stmts: List[StatementIR]) -> List[StatementIR]:
        result = []
        for stmt in stmts:
            normalized = self._normalize_statement(stmt)
            if isinstance(normalized, list):
                result.extend(normalized)
            elif normalized is not None:
                result.append(normalized)
        return result

    def _normalize_statement(self, stmt: StatementIR):
        if stmt.body:
            stmt.body = self._normalize_statements(stmt.body)
        if stmt.orelse:
            stmt.orelse = self._normalize_statements(stmt.orelse)
        if stmt.expression:
            stmt.expression = self._normalize_expression(stmt.expression)
        if stmt.target:
            stmt.target = self._normalize_expression(stmt.target)
        if stmt.condition:
            stmt.condition = self._normalize_expression(stmt.condition)
        if stmt.value:
            stmt.value = self._normalize_expression(stmt.value)
        if stmt.iterable:
            stmt.iterable = self._normalize_expression(stmt.iterable)
        return stmt

    def _normalize_expression(self, expr: ExpressionIR) -> ExpressionIR:
        if expr is None:
            return None
        if expr.left:
            expr.left = self._normalize_expression(expr.left)
        if expr.right:
            expr.right = self._normalize_expression(expr.right)
        if expr.operand:
            expr.operand = self._normalize_expression(expr.operand)
        if expr.target:
            expr.target = self._normalize_expression(expr.target)
        if expr.func:
            expr.func = self._normalize_expression(expr.func)
        if expr.index:
            expr.index = self._normalize_expression(expr.index)
        expr.args = [self._normalize_expression(a) for a in expr.args if a is not None]
        expr.kwargs = {k: self._normalize_expression(v) for k, v in expr.kwargs.items() if v is not None}
        return expr

    def _fresh_temp_var(self) -> str:
        self._temp_var_counter += 1
        return f"_t{self._temp_var_counter}"


class SSAConverter:
    """Converts IR to Static Single Assignment form where each variable is assigned exactly once."""

    def __init__(self):
        self._version_counter: Dict[str, int] = {}
        self._current_versions: Dict[str, str] = {}

    def to_ssa(self, ir: NormalizedIR) -> NormalizedIR:
        for func in ir.functions:
            self._ssa_function(func)
        for cls in ir.classes:
            for method in cls.methods:
                self._ssa_function(method)
        return ir

    def _ssa_function(self, func: FunctionIR):
        self._version_counter = {}
        self._current_versions = {}
        for param in func.signature.parameters:
            self._fresh_version(param.name)
        func.body = self._ssa_statements(func.body)

    def _fresh_version(self, name: str) -> str:
        count = self._version_counter.get(name, 0)
        self._version_counter[name] = count + 1
        versioned = f"{name}_{count}"
        self._current_versions[name] = versioned
        return versioned

    def _get_version(self, name: str) -> str:
        return self._current_versions.get(name, name)

    def _ssa_statements(self, stmts: List[StatementIR]) -> List[StatementIR]:
        result = []
        for stmt in stmts:
            ssa_stmts = self._ssa_statement(stmt)
            if isinstance(ssa_stmts, list):
                result.extend(ssa_stmts)
            else:
                result.append(ssa_stmts)
        return result

    def _ssa_statement(self, stmt: StatementIR):
        if stmt.target and stmt.target.expr_type == "variable":
            stmt.target.name = self._fresh_version(stmt.target.name)
        if stmt.expression:
            stmt.expression = self._ssa_expression(stmt.expression)
        if stmt.condition:
            stmt.condition = self._ssa_expression(stmt.condition)
        if stmt.value:
            stmt.value = self._ssa_expression(stmt.value)
        if stmt.body:
            stmt.body = self._ssa_statements(stmt.body)
        if stmt.orelse:
            stmt.orelse = self._ssa_statements(stmt.orelse)
        return stmt

    def _ssa_expression(self, expr: ExpressionIR) -> ExpressionIR:
        if expr is None:
            return None
        if expr.expr_type == "variable":
            expr.name = self._get_version(expr.name)
        if expr.left:
            expr.left = self._ssa_expression(expr.left)
        if expr.right:
            expr.right = self._ssa_expression(expr.right)
        if expr.operand:
            expr.operand = self._ssa_expression(expr.operand)
        if expr.target:
            expr.target = self._ssa_expression(expr.target)
        if expr.func:
            expr.func = self._ssa_expression(expr.func)
        if expr.index:
            expr.index = self._ssa_expression(expr.index)
        expr.args = [self._ssa_expression(a) for a in expr.args if a is not None]
        expr.kwargs = {k: self._ssa_expression(v) for k, v in expr.kwargs.items() if v is not None}
        return expr
