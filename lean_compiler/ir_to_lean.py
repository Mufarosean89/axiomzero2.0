"""
Axiom Zero - Phase 4: IR → Lean 4 Compiler

Translates the pipeline's NormalizedIR, SpecCollection, and AbstractState into
a complete Lean 4 module.  Each function with formal specifications (from Phase 1
@requires/@ensures decorators) generates:

  1. A theorem statement for each proof obligation.
  2. A proof block — either auto-filled (simple arithmetic, equalities) or left
     as a ``sorry`` placeholder for the RL agent (Phase 3).

The translation is entirely deterministic: given the same IR + specs, the same
Lean 4 output is always produced.

Key design points
-----------------
- Python ``int`` → Lean ``ℤ`` (signed integers).
- Python ``bool`` → Lean ``Bool``.
- Python ``List[T]`` → Lean ``List T``.
- Python binary operators are mapped to their Lean equivalents (``+``, ``-``,
  ``*``, ``/``, ``<``, ``≤``, ``>``, ``≥``, ``=``, ``≠``, ``∧``, ``∨``, ``¬``).
- ``@requires(p)`` becomes a theorem whose goal is ``p``, with the function
  parameters as binder hypotheses.
- ``@ensures(p)`` becomes a theorem whose goal is ``p``, with both parameters
  and preconditions as binder hypotheses.
- Loop invariants are expressed as ``∀`` statements over the induction variable.
- Every proof obligation that is provable by ``omega``, ``simp``, or ``rfl`` is
  filled immediately; everything else stays as ``sorry`` (delegated to Phase 3).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ast_extractor.ir import (
    NormalizedIR,
    FunctionIR,
    StatementIR,
    ExpressionIR,
    TypeAnnotation,
    TensorOpKind,
    ClassIR,
)
from abstract_interpreter.abstract_domain import AbstractState, TypeDomain
from spec_ingestion.obligations import (
    ProofObligation,
    SpecCollection,
    ObligationKind,
    ObligationStatus,
)
from proof_engine.builder import obligation_to_lean_theorem


# ── Python → Lean type mapping ──────────────────────────────────────────────

_PY_TYPE_TO_LEAN: Dict[str, str] = {
    "int": "ℤ",
    "Int": "ℤ",
    "float": "ℝ",
    "Float": "ℝ",
    "bool": "Bool",
    "Bool": "Bool",
    "str": "String",
    "String": "String",
    "None": "Unit",
    "NoneType": "Unit",
    "List": "List",
    "list": "List",
    "Dict": "HashMap",
    "dict": "HashMap",
    "Optional": "Option",
    "Set": "Set",
    "set": "Set",
    "Tuple": "Prod",
    "tuple": "Prod",
}


# ── Python → Lean operator mapping ──────────────────────────────────────────

_BINOP_TO_LEAN: Dict[str, str] = {
    "+": " + ",
    "-": " - ",
    "*": " * ",
    "/": " / ",
    "//": " / ",         # ℕ division is the same symbol
    "%": " % ",
    "**": " ^ ",
    "==": " = ",
    "!=": " ≠ ",
    "<": " < ",
    "<=": " ≤ ",
    ">": " > ",
    ">=": " ≥ ",
    "and": " ∧ ",
    "or": " ∨ ",
    "in": " ∈ ",
    "not in": " ∉ ",
}

_UNOP_TO_LEAN: Dict[str, str] = {
    "-": "-",
    "not": "¬",
    "~": "~~~",
}


# ── Difficulty heuristics ───────────────────────────────────────────────────

_SIMPLE_PATTERNS: List[str] = [
    r"^\s*\d+\s*[+\-*/]\s*\d+\s*[=<>!]=\s*\d+\s*$",        # 1 + 2 == 3
    r"^\s*x\s*[+\-*/]\s*\d+\s*[=<>!]=\s*\d+\s*$",           # x + 1 == 2
    r"^\s*\d+\s*[+\-*/]\s*x\s*[=<>!]=\s*\d+\s*$",           # 1 + x == 2
    r"^\s*x\s*[+\-*/]\s*y\s*[=<>!]=\s*[a-z]+\s*$",          # x + y == result
    r"^\s*x\s*[=<>!]=\s*\d+\s*$",                            # x == 5
    r"^\s*x\s*[+\-*/]\s*0\s*[=<>!]=\s*x\s*$",               # x + 0 == x
    r"^\s*0\s*[+\-*/]\s*x\s*[=<>!]=\s*x\s*$",               # 0 + x == x
    r"^\s*x\s*-\s*x\s*[=<>!]=\s*0\s*$",                     # x - x == 0
    r"^\s*x\s*[*/]\s*1\s*[=<>!]=\s*x\s*$",                  # x * 1 == x
    r"^\s*1\s*[*]\s*x\s*[=<>!]=\s*x\s*$",                   # 1 * x == x
    r"^\s*x\s*[=<>!]=\s*x\s*$",                              # x == x
    r"^\s*x\s*>\s*0\s*$",                                    # x > 0
    r"^\s*x\s*<\s*0\s*$",                                    # x < 0
    r"^\s*x\s*>=\s*0\s*$",                                   # x >= 0
    r"^\s*x\s*<=\s*0\s*$",                                   # x <= 0
    r"^\s*result\s*[=<>!]=\s*.*$",                           # result == ...
    r"^\s*loop_terminates",                                  # loop termination
    r"^\s*is_tensor",                                        # tensor existence
]

_OMEGA_PATTERNS: List[str] = [
    r"^\s*x\s*[+]\s*\d+\s*>\s*x\s*$",                       # x + n > x
    r"^\s*x\s*[+]\s*\d+\s*>=\s*x\s*$",                       # x + n >= x
    r"^\s*x\s*>\s*\d+\s*→\s*x\s*>\s*0\s*$",                 # x > 5 → x > 0
    r"^\s*x\s*[+]\s*y\s*[=<>!]=\s*y\s*[+]\s*x\s*$",        # x + y == y + x (comm)
    r"^\s*x\s*>\s*0\s*→\s*-x\s*<\s*0\s*$",                  # x > 0 → -x < 0
    r"^\s*x\s*<\s*0\s*→\s*-x\s*>\s*0\s*$",                  # x < 0 → -x > 0
]


# ═══════════════════════════════════════════════════════════════════════════
#  IR → Lean Compiler
# ═══════════════════════════════════════════════════════════════════════════

class IRToLeanCompiler:
    """
    Translates Axiom Zero's NormalizedIR + SpecCollection into a Lean 4 module.

    Usage
    -----
        compiler = IRToLeanCompiler()
        module = compiler.compile_module(ir, specs, abstract_state, "my_module")
        with open("output.lean", "w") as f:
            f.write(module)
    """

    def __init__(self) -> None:
        self._module_name: str = ""
        self._ir: Optional[NormalizedIR] = None
        self._specs: Optional[SpecCollection] = None
        self._abstract_state: Optional[AbstractState] = None

    # ── Public API ────────────────────────────────────────────────────────

    def compile_module(
        self,
        ir: NormalizedIR,
        specs: SpecCollection,
        abstract_state: AbstractState,
        module_name: str = "target",
        module_doc: str = "",
    ) -> str:
        """
        Compile a full module into Lean 4.

        Args:
            ir              : Normalized IR from the AST pipeline.
            specs           : Spec collection extracted from the IR.
            abstract_state  : Abstract state from type/shape analysis.
            module_name     : Name for the generated Lean module.
            module_doc      : Optional documentation string (placed in a comment).

        Returns:
            Complete Lean 4 module source as a string.
        """
        self._ir = ir
        self._specs = specs
        self._abstract_state = abstract_state
        self._module_name = module_name

        lines: List[str] = []

        # Module header
        if module_doc:
            lines.append(f"/- {module_doc} -/")
        else:
            lines.append(f"/- Auto-generated by Axiom Zero from module: {module_name} -/")
        lines.append("")

        # Imports
        lines.append("import Mathlib")
        lines.append("open Classical")
        lines.append("")

        # Optional: set_option pp.all true for debugging
        # lines.append("set_option pp.all true")
        # lines.append("")

        # Type aliases (from abstract state)
        type_lines = self._generate_type_aliases()
        if type_lines:
            lines.extend(type_lines)
            lines.append("")

        # Function specifications as Lean theorems
        theorem_lines = self._generate_theorems()
        lines.extend(theorem_lines)

        return "\n".join(lines)

    # ── Theorem generation ────────────────────────────────────────────────

    def _generate_type_aliases(self) -> List[str]:
        """Generate Lean type aliases for custom Python types."""
        # Currently a no-op; custom type aliases can be added here.
        return []

    def _generate_theorems(self) -> List[str]:
        """Generate Lean theorem blocks for all proof obligations."""
        lines: List[str] = []
        seen_ids: Set[str] = set()

        for ob in self._specs.all if self._specs else []:
            if ob.id in seen_ids:
                continue
            seen_ids.add(ob.id)

            # Generate the theorem header and skeleton
            theorem_text = self._obligation_to_lean(ob)
            if theorem_text:
                lines.append(theorem_text)
                lines.append("")

        return lines

    def _obligation_to_lean(self, ob: ProofObligation) -> str:
        """
        Convert a single proof obligation into a Lean 4 theorem + proof block.

        Uses the existing ``obligation_to_lean_theorem`` from the proof_engine
        bridge to generate the skeleton, then optionally replaces ``sorry`` with
        a heuristic proof.
        """
        # Use the existing bridge to get the skeleton
        theorem_skeleton = obligation_to_lean_theorem(ob)

        # Determine the proof strategy based on obligation kind and predicate
        proof_tactic = self._select_proof_tactic(ob)

        if proof_tactic and proof_tactic != "sorry":
            # Replace the last line ("  sorry") with the proof block
            lines = theorem_skeleton.rsplit("\n", 1)
            if len(lines) == 2 and lines[1].strip() == "sorry":
                theorem_skeleton = lines[0] + "\n" + self._format_proof_block(proof_tactic)

        return theorem_skeleton

    def _select_proof_tactic(self, ob: ProofObligation) -> str:
        """Select a proof tactic for the obligation based on its predicate."""
        pred = ob.predicate.strip()
        kind = ob.kind

        # Handle specific obligation kinds
        if kind == ObligationKind.LOOP_TERMINATION:
            return "sorry"  # Loop termination often needs manual reasoning

        if kind == ObligationKind.SHAPE_CONDITION:
            if pred.startswith("is_tensor"):
                return "sorry"  # Tensor existence — needs the specific context
            if "matmul_shapes_compatible" in pred:
                return "sorry"

        if kind == ObligationKind.PRECONDITION:
            # Preconditions often involve simple arithmetic
            return self._classify_predicate(pred)

        if kind == ObligationKind.POSTCONDITION:
            return self._classify_predicate(pred)

        if kind == ObligationKind.ASSERTION:
            return self._classify_predicate(pred)

        return "sorry"

    def _classify_predicate(self, pred: str) -> str:
        """Classify a predicate string and return the best proof tactic."""
        stripped = pred.strip()

        # Check omega patterns first (arithmetic)
        for pattern in _OMEGA_PATTERNS:
            if re.match(pattern, stripped):
                return "omega"

        # Check simple patterns (simp / rfl)
        for pattern in _SIMPLE_PATTERNS:
            if re.match(pattern, stripped):
                return "simp"

        # Equality of simple expressions
        if " = " in stripped and not any(c in stripped for c in ("∀", "∃", "→", "∧", "∨")):
            left, right = stripped.split(" = ", 1)
            left = left.strip()
            right = right.strip()
            # x = x  → rfl
            if left == right:
                return "rfl"
            # Simple numeric identities
            if left.isdigit() and right.isdigit():
                if int(left) == int(right):
                    return "rfl"

        # Boolean / trivial goals
        if stripped in ("True", "true"):
            return "trivial"

        # Default: defer to RL agent — only simp what we're confident about
        return "sorry"

    def _format_proof_block(self, tactic: str) -> str:
        """Format a proof block for a given tactic."""
        return f"  by\n    {tactic}"

    # ── Expression translation ────────────────────────────────────────────

    def translate_expression(self, expr: ExpressionIR) -> str:
        """
        Translate an Axiom Zero expression IR node to a Lean 4 expression string.

        This is used for translating loop invariants, assertions, and conditions
        into Lean syntax.
        """
        if expr is None:
            return "True"

        expr_type = expr.expr_type

        if expr_type == "constant":
            return self._translate_constant(expr.value)

        if expr_type == "variable":
            # Rename common Python variables to Lean conventions
            name = expr.name
            if name == "result":
                return "result"
            return name

        if expr_type == "binary_op":
            return self._translate_binary_op(expr)

        if expr_type == "unary_op":
            return self._translate_unary_op(expr)

        if expr_type == "call":
            return self._translate_call(expr)

        if expr_type == "attribute":
            return self._translate_attribute(expr)

        if expr_type == "subscript":
            return self._translate_subscript(expr)

        if expr_type == "list":
            return self._translate_list(expr)

        if expr_type == "if_exp":
            return self._translate_if_exp(expr)

        return "True"  # safe fallback

    def _translate_constant(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            # Use rational notation if integer-valued
            if value == int(value):
                return f"({int(value)} : ℝ)"
            return f"({value} : ℝ)"
        if isinstance(value, str):
            return f'"{value}"'
        if value is None:
            return "none"
        return str(value)

    def _translate_binary_op(self, expr: ExpressionIR) -> str:
        left = self.translate_expression(expr.left) if expr.left else "?"
        right = self.translate_expression(expr.right) if expr.right else "?"
        op = expr.op

        lean_op = _BINOP_TO_LEAN.get(op, f" {op} ")
        return f"({left}{lean_op}{right})"

    def _translate_unary_op(self, expr: ExpressionIR) -> str:
        operand = self.translate_expression(expr.operand) if expr.operand else "?"
        op = expr.op
        lean_op = _UNOP_TO_LEAN.get(op, op)
        return f"({lean_op}{operand})"

    def _translate_call(self, expr: ExpressionIR) -> str:
        func_name = ""
        if expr.func:
            func_name = self.translate_expression(expr.func)

        args = [self.translate_expression(a) for a in expr.args]

        # Map known Python functions to Lean equivalents
        name = func_name.lower()

        if name in ("abs",):
            return f"(abs {args[0]})" if args else f"(abs ?_)"
        if name in ("len",):
            # Lean: List.length, Finset.card, etc.
            return f"({args[0]}.length)" if args else "(?_.length)"
        if name in ("max", "min"):
            return f"({name} {args[0]} {args[1]})" if len(args) >= 2 else f"({name} {args[0]})"
        if name in ("range",):
            return f"(Finset.range {args[0]})" if args else "(Finset.range ?_)"
        if name in ("int", "float", "str", "bool"):
            return args[0] if args else "?"

        # Generic function application
        if func_name:
            return f"({func_name} {' '.join(args)})"

        return f"(? {', '.join(args)})" if args else "?"

    def _translate_attribute(self, expr: ExpressionIR) -> str:
        target = self.translate_expression(expr.target) if expr.target else "?"
        attr = expr.attr

        if attr in ("shape",):
            return f"({target}.shape)"
        if attr in ("length",):
            return f"({target}.length)"
        if attr == "T":
            return f"({target}.transpose)"

        return f"({target}.{attr})"

    def _translate_subscript(self, expr: ExpressionIR) -> str:
        target = self.translate_expression(expr.target) if expr.target else "?"
        index = self.translate_expression(expr.index) if expr.index else "?"
        return f"({target}[{index}])"

    def _translate_list(self, expr: ExpressionIR) -> str:
        elements = [self.translate_expression(e) for e in expr.elements]
        return f"[{', '.join(elements)}]"

    def _translate_if_exp(self, expr: ExpressionIR) -> str:
        condition = self.translate_expression(expr.left) if expr.left else "?"
        then_expr = self.translate_expression(expr.right) if expr.right else "?"
        else_expr = self.translate_expression(expr.operand) if expr.operand else "?"
        return f"(if {condition} then {then_expr} else {else_expr})"

    # ── Type translation ──────────────────────────────────────────────────

    def translate_type(self, py_type: str) -> str:
        """
        Translate a Python type string to a Lean 4 type string.

        Args:
            py_type: A Python type name (e.g., "int", "List[int]", "Optional[str]").

        Returns:
            Lean 4 type string (e.g., "ℤ", "List ℤ", "Option String").
        """
        py_type = py_type.strip()

        # Handle generic types: List[int], Optional[str], etc.
        if "[" in py_type and py_type.endswith("]"):
            base = py_type[: py_type.index("[")]
            inner = py_type[py_type.index("[") + 1 : -1]
            base_lean = _PY_TYPE_TO_LEAN.get(base, base)
            inner_lean = self.translate_type(inner)
            return f"{base_lean} {inner_lean}"

        # Simple types
        return _PY_TYPE_TO_LEAN.get(py_type, py_type)

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def obligation_to_predicate_lean(ob: ProofObligation) -> str:
        """
        Convert a proof obligation's predicate to a Lean 4 proposition string.

        This handles common Python idioms in predicates:
        - ``result`` → ``result`` (the Lean binder)
        - ``==`` → ``=``
        - ``!=`` → ``≠``
        - ``and``/``or``/``not`` → ``∧``/``∨``/``¬``
        - ``True``/``False`` → ``true``/``false``

        Args:
            ob: The proof obligation.

        Returns:
            Lean 4 proposition string.
        """
        pred = ob.predicate.strip()

        # Use string replacements for common patterns
        replacements = [
            ("==", "="),
            ("!=", " ≠ "),
            (" and ", " ∧ "),
            (" or ", " ∨ "),
            ("not ", "¬ "),
            ("True", "true"),
            ("False", "false"),
        ]
        for old, new in replacements:
            pred = pred.replace(old, new)

        # Normalize whitespace: collapse multiple spaces
        pred = re.sub(r'\s+', ' ', pred).strip()
        return pred
