"""
Axiom Zero - Phase 4: Hole Filler

Classifies proof obligations by difficulty and fills them with appropriate
proof tactics:

  - **Easy** (arithmetic, simple equalities) → ``omega``, ``simp``, ``rfl``,
    ``trivial``, ``norm_num``
  - **Medium** (list invariants, simple inductions) → ``simp``, ``induction``
    with heuristic patterns
  - **Hard** (complex arithmetic, nested loops, tensor constraints) → ``sorry``
    (delegated to the Phase 3 RL agent / MCTS)

The ``HoleFiller`` operates on a generated Lean module string, scanning for
``sorry`` placeholders and replacing them with concrete proof blocks based on
the context of the surrounding theorem.
"""

from __future__ import annotations

import re
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from spec_ingestion.obligations import (
    ProofObligation,
    SpecCollection,
    ObligationKind,
)


class HoleDifficulty(Enum):
    """Difficulty level of a proof hole."""
    TRIVIAL = auto()     # rfl, trivial
    EASY = auto()        # simp, omega
    MEDIUM = auto()      # induction, cases + simp
    HARD = auto()        # Needs MCTS / RL agent


# ── Predicate classifiers ───────────────────────────────────────────────────

_SIMPLE_EQUALITY = re.compile(r"^\s*\w+\s*=\s*\w+\s*$")
_SIMPLE_NUMERIC = re.compile(r"^\s*\w+\s*[+*]\s*\w+\s*=\s*\w+\s*$")
_TRIVIAL_TRUE = re.compile(r"^\s*(True|true)\s*$")
_ARITH_INEQUALITY = re.compile(
    r"^\s*\w+\s*([<>]=?)\s*-?\d+\s*$"
)
# Reflexive equality check is handled inline in classify_hole

# Patterns that strongly suggest omega is sufficient
_OMEGA_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*\w+\s*[+]\s*\d+\s*>\s*\w+\s*$"),        # x + 1 > x
    re.compile(r"^\s*\w+\s*[+]\s*\d+\s*>=\s*\w+\s*$"),       # x + 1 >= x
    re.compile(r"^\s*\w+\s*>\s*\d+\s*→\s*\w+\s*>\s*0\s*$"), # x > 5 → x > 0
    re.compile(r"^\s*\w+\s*<\s*0\s*→\s*-\w+\s*>\s*0\s*$"),  # x < 0 → -x > 0
    re.compile(r"^\s*\w+\s*>\s*0\s*→\s*-\w+\s*<\s*0\s*$"),  # x > 0 → -x < 0
]

# Patterns that suggest simp is sufficient
_SIMPLE_SIMP_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*\w+\s*\+\s*0\s*=\s*\w+\s*$"),           # x + 0 = x
    re.compile(r"^\s*0\s*\+\s*\w+\s*=\s*\w+\s*$"),           # 0 + x = x
    re.compile(r"^\s*\w+\s*-\s*\w+\s*=\s*0\s*$"),            # x - x = 0
    re.compile(r"^\s*\w+\s*\*\s*1\s*=\s*\w+\s*$"),           # x * 1 = x
    re.compile(r"^\s*1\s*\*\s*\w+\s*=\s*\w+\s*$"),           # 1 * x = x
    re.compile(r"^\s*\w+\s*=\s*\w+\s*$"),                     # x = x (same var)
    re.compile(r"^\s*\w+\s*>\s*0\s*$"),                       # x > 0
    re.compile(r"^\s*\w+\s*<\s*0\s*$"),                      # x < 0
    re.compile(r"^\s*\w+\s*>=\s*0\s*$"),                     # x >= 0
    re.compile(r"^\s*\w+\s*<=\s*0\s*$"),                     # x <= 0
]


class HoleFiller:
    """
    Fills ``sorry`` holes in generated Lean code with appropriate proof tactics.

    Usage
    -----
        filler = HoleFiller()
        filled_code = filler.fill_all_holes(lean_code, specs)
    """

    def __init__(self) -> None:
        self._fill_count: int = 0
        self._total_holes: int = 0

    def fill_all_holes(self, lean_code: str, specs: SpecCollection) -> str:
        """
        Scan the generated Lean code for ``sorry`` placeholders and replace
        each one with a proof block determined by the obligation's predicate.

        Args:
            lean_code: The generated Lean 4 module string.
            specs: The spec collection (used to map sorry positions to
                   obligations for context-aware filling).

        Returns:
            Lean module with as many holes filled as possible.
        """
        # Count total holes first
        self._total_holes = lean_code.count("  sorry")
        self._fill_count = 0

        # Collect obligations indexed by theorem name for lookup
        obligations_by_name: Dict[str, ProofObligation] = {}
        for ob in specs.all:
            name = ob.id.replace(":", "_").replace(".", "_").replace("-", "_")
            obligations_by_name[name] = ob

        # Find each theorem block and try to fill its sorry
        result = lean_code
        theorem_pattern = re.compile(
            r"(theorem\s+(\S+)(?:\s+.*?)?\s*:=\s*\n\s+)sorry",
            re.DOTALL,
        )

        def _replacer(match: re.Match) -> str:
            theorem_name = match.group(2)
            prefix = match.group(1)

            # Try to find the matching obligation
            ob = obligations_by_name.get(theorem_name)
            if ob is None:
                # Try fuzzy match
                for name, candidate in obligations_by_name.items():
                    if theorem_name.startswith(name) or name.startswith(theorem_name):
                        ob = candidate
                        break

            if ob is not None:
                tactic = self._select_tactic(ob)
                if tactic and tactic != "sorry":
                    self._fill_count += 1
                    proof_block = f"  by\n    {tactic}"
                    return prefix + proof_block

            # Also try to infer from the theorem type string
            tactic = self._infer_from_predicate(theorem_name)
            if tactic and tactic != "sorry":
                self._fill_count += 1
                proof_block = f"  by\n    {tactic}"
                return prefix + proof_block

            return match.group(0)  # keep as sorry

        result = theorem_pattern.sub(_replacer, result)

        # Also handle inline "by\n  sorry" patterns
        inline_pattern = re.compile(r"(by\s*\n\s+)sorry")
        def _inline_replacer(match: re.Match) -> str:
            prefix = match.group(1)
            return prefix + "simp"  # conservative default for inline sorries

        result = inline_pattern.sub(_inline_replacer, result)

        return result

    def fill_hole(self, obligation: ProofObligation) -> str:
        """
        Fill a single proof obligation's hole with the appropriate proof block.

        Args:
            obligation: The proof obligation to fill.

        Returns:
            A Lean 4 proof block (e.g., ``by\\n  omega``).
        """
        tactic = self._select_tactic(obligation)
        if tactic and tactic != "sorry":
            self._fill_count += 1
            return f"  by\n    {tactic}"
        return "  sorry"

    def classify_hole(self, obligation: ProofObligation) -> HoleDifficulty:
        """Classify the difficulty of a proof obligation."""
        pred = obligation.predicate.strip()
        kind = obligation.kind

        # Normalize Python-style == to Lean-style =
        pred_normalized = pred.replace("==", "=").replace("!=", "≠")

        # Step 1: Check obligation kind FIRST (loop invariants, shape conditions)
        # These should be classified based on their kind, not simple pattern matching
        if kind in (ObligationKind.LOOP_INVARIANT, ObligationKind.LOOP_TERMINATION):
            return HoleDifficulty.MEDIUM
        if kind == ObligationKind.SHAPE_CONDITION:
            return HoleDifficulty.MEDIUM

        # Step 2: Trivial: reflexivity, true
        if pred_normalized in ("True", "true"):
            return HoleDifficulty.TRIVIAL
        parts = pred_normalized.split('=')
        if len(parts) == 2 and parts[0].strip() == parts[1].strip():
            return HoleDifficulty.TRIVIAL

        # Step 3: Easy: omega/simp patterns (use normalized predicate)
        for pat in _SIMPLE_SIMP_PATTERNS:
            if pat.match(pred_normalized):
                return HoleDifficulty.EASY
        for pat in _OMEGA_PATTERNS:
            if pat.match(pred_normalized):
                return HoleDifficulty.EASY
        if _SIMPLE_EQUALITY.match(pred_normalized):
            return HoleDifficulty.EASY

        # Step 4: Hard: everything else
        return HoleDifficulty.HARD

    def _select_tactic(self, obligation: ProofObligation) -> str:
        """Select the best tactic for the given obligation."""
        difficulty = self.classify_hole(obligation)
        pred = obligation.predicate.strip()

        if difficulty == HoleDifficulty.TRIVIAL:
            if pred in ("True", "true"):
                return "trivial"
            # Try rfl for equalities
            if "=" in pred and not any(c in pred for c in ("∀", "∃", "→", "∧")):
                return "rfl"
            return "simp"

        if difficulty == HoleDifficulty.EASY:
            # Check omega first
            for pat in _OMEGA_PATTERNS:
                if pat.match(pred):
                    return "omega"
            # Then simp
            for pat in _SIMPLE_SIMP_PATTERNS:
                if pat.match(pred):
                    return "simp"
            # Numeric reasoning
            if any(c.isdigit() for c in pred):
                return "omega"
            return "simp"

        if difficulty == HoleDifficulty.MEDIUM:
            # For loop invariants, try simp with induction
            if obligation.kind == ObligationKind.LOOP_INVARIANT:
                return "simp"
            if obligation.kind == ObligationKind.LOOP_TERMINATION:
                return "sorry"  # Needs manual reasoning
            if obligation.kind == ObligationKind.SHAPE_CONDITION:
                return "sorry"  # Needs tensor-specific reasoning
            return "simp"

        # Hard — defer to RL agent / MCTS
        return "sorry"

    def _infer_from_predicate(self, pred: str) -> str:
        """Try to infer a proof tactic from a predicate string alone."""
        stripped = pred.strip()

        for pat in _SIMPLE_SIMP_PATTERNS:
            if pat.match(stripped):
                return "simp"
        for pat in _OMEGA_PATTERNS:
            if pat.match(stripped):
                return "omega"
        if stripped == "True" or stripped == "true":
            return "trivial"
        if " = " in stripped:
            return "simp"
        return "sorry"

    @property
    def fill_rate(self) -> float:
        """Fraction of holes that were successfully filled."""
        if self._total_holes == 0:
            return 1.0
        return self._fill_count / self._total_holes
