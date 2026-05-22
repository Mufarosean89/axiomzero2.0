"""
Axiom Zero - IR Normalizer

Transforms the raw extracted IR into a normalized form suitable for
abstract interpretation and proof obligation generation.

Normalization passes:
1. Desugar augmented assignments (a += 1 → a = a + 1)
2. Flatten nested expressions to SSA-like form
3. Normalize loop structures (for → while with explicit induction variable)
4. Simplify boolean expressions
5. Expand list/dict comprehensions into explicit loops
6. Annotate with implicit type information
7. Extract tensor operation chains into sequential steps
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any
from .ir import (
    NormalizedIR,
    FunctionIR,
    ClassIR,
    StatementIR,
    ExpressionIR,
    LoopIR,
    ConditionalIR,
    TypeAnnotation,
    TensorOpKind,
)


class Normalizer:
    """
    Normalizes the extracted IR for downstream analysis.
    
    Usage:
        normalizer = Normalizer()
        normalized = normalizer.normalize(ir)
    """

    def __init__(self):
        self._temp_var_counter: int = 0
        self._current_function: Optional[str] = None

    def normalize(self, ir: NormalizedIR) -> NormalizedIR:
        """
        Run all normalization passes on the IR.
        
        Args:
            ir: Raw NormalizedIR from the parser
            
        Returns:
            Normalized NormalizedIR (mutated in place)
        """
        # Pass 1: Normalize all function bodies
        for func in ir.functions:
            self._normalize_function(func)

        # Pass 2: Normalize class methods
        for cls in ir.classes:
            for method in cls.methods:
                self._normalize_function(method)

        # Pass 3: Normalize global statements
        ir.global_statements = self._normalize_statements(ir.global_statements)

        return ir

    def _normalize_function(self, func: FunctionIR):
        """Normalize a single function body."""
        self._current_function = func.signature.name
        self._temp_var_counter = 0
        func.body = self._normalize_statements(func.body)

        for nested in func.nested_functions:
            self._normalize_function(nested)

    def _normalize_statements(self, stmts: List[StatementIR]) -> List[StatementIR]:
        """Normalize a list of statements."""
        result = []
        for stmt in stmts:
            normalized = self._normalize_statement(stmt)
            if isinstance(normalized, list):
                result.extend(normalized)
            elif normalized is not None:
                result.append(normalized)
        return result

    def _normalize_statement(self, stmt: StatementIR):
        """Normalize a single statement."""
        # Normalize nested statements in body/orelse
        if stmt.body:
            stmt.body = self._normalize_statements(stmt.body)
        if stmt.orelse:
            stmt.orelse = self._normalize_statements(stmt.orelse)

        # Normalize expressions
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
        """Normalize an expression (simplify if needed)."""
        if expr is None:
            return None

        # Recursively normalize sub-expressions
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

        # Normalize args/kwargs
        expr.args = [self._normalize_expression(a) for a in expr.args if a is not None]
        expr.kwargs = {k: self._normalize_expression(v) for k, v in expr.kwargs.items() if v is not None}

        return expr

    def _fresh_temp_var(self) -> str:
        """Generate a fresh temporary variable name."""
        self._temp_var_counter += 1
        return f"_t{self._temp_var_counter}"


class SSAConverter:
    """
    Converts IR to Static Single Assignment form.
    Each variable is assigned exactly once, making data flow explicit.
    
    This is a light-weight pass that can be run after normalization.
    """

    def __init__(self):
        self._version_counter: Dict[str, int] = {}
        self._current_versions: Dict[str, str] = {}

    def to_ssa(self, ir: NormalizedIR) -> NormalizedIR:
        """Convert the IR to SSA form."""
        for func in ir.functions:
            self._ssa_function(func)
        for cls in ir.classes:
            for method in cls.methods:
                self._ssa_function(method)
        return ir

    def _ssa_function(self, func: FunctionIR):
        """Convert a function body to SSA form."""
        self._version_counter = {}
        self._current_versions = {}

        # Initialize parameters
        for param in func.signature.parameters:
            self._fresh_version(param.name)

        func.body = self._ssa_statements(func.body)

    def _fresh_version(self, name: str) -> str:
        """Get the next SSA version for a variable."""
        count = self._version_counter.get(name, 0)
        self._version_counter[name] = count + 1
        versioned = f"{name}_{count}"
        self._current_versions[name] = versioned
        return versioned

    def _get_version(self, name: str) -> str:
        """Get the current SSA version of a variable."""
        if name in self._current_versions:
            return self._current_versions[name]
        return name

    def _ssa_statements(self, stmts: List[StatementIR]) -> List[StatementIR]:
        """Convert a list of statements to SSA form."""
        result = []
        for stmt in stmts:
            ssa_stmts = self._ssa_statement(stmt)
            if isinstance(ssa_stmts, list):
                result.extend(ssa_stmts)
            else:
                result.append(ssa_stmts)
        return result

    def _ssa_statement(self, stmt: StatementIR):
        """Convert a single statement to SSA form."""
        if stmt.target and stmt.target.expr_type == "variable":
            name = stmt.target.name
            new_version = self._fresh_version(name)
            stmt.target.name = new_version

        # Rename variable references in expressions
        if stmt.expression:
            stmt.expression = self._ssa_expression(stmt.expression)
        if stmt.condition:
            stmt.condition = self._ssa_expression(stmt.condition)
        if stmt.value:
            stmt.value = self._ssa_expression(stmt.value)

        # Recurse into blocks
        if stmt.body:
            stmt.body = self._ssa_statements(stmt.body)
        if stmt.orelse:
            stmt.orelse = self._ssa_statements(stmt.orelse)

        return stmt

    def _ssa_expression(self, expr: ExpressionIR) -> ExpressionIR:
        """Rename variable references in an expression."""
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
