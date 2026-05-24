"""Axiom Zero — Phase 4: Hole Filler

Classifies proof obligations by difficulty and fills them with appropriate
proof tactics:

- Easy (arithmetic, simple equalities) -> ``omega``, ``simp``, ``rfl``, ``trivial``, ``norm_num``
- Medium (list invariants, simple inductions) -> ``simp``, ``induction`` with heuristic patterns
- Hard (complex arithmetic, nested loops, tensor constraints) -> ``sorry`` (delegated to RL agent / MCTS)
"""

from __future__ import annotations

import re
from enum import Enum, auto
from typing import Dict, List, Optional

from spec_ingestion.obligations import ProofObligation, SpecCollection, ObligationKind


class HoleDifficulty(Enum):
    """Difficulty level of a proof hole."""
    TRIVIAL = auto()
    EASY = auto()
    MEDIUM = auto()
    HARD = auto()


# ── Predicate classifiers ───────────────────────────────────────────────────

_TRIVIAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*(True|true)\s*$"),
]

_SIMP_SIMP_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*\w+\s*\+\s*0\s*=\s*\w+\s*$"),
    re.compile(r"^\s*0\s*\+\s*\w+\s*=\s*\w+\s*$"),
    re.compile(r"^\s*\w+\s*-\s*\w+\s*=\s*0\s*$"),
    re.compile(r"^\s*\w+\s*\*\s*1\s*=\s*\w+\s*$"),
    re.compile(r"^\s*1\s*\*\s*\w+\s*=\s*\w+\s*$"),
    re.compile(r"^\s*\w+\s*>\s*0\s*$"),
    re.compile(r"^\s*\w+\s*<\s*0\s*$"),
    re.compile(r"^\s*\w+\s*>=\s*0\s*$"),
    re.compile(r"^\s*\w+\s*<=\s*0\s*$"),
]

_OMEGA_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*\w+\s*\+\s*\d+\s*>\s*\w+\s*$"),
    re.compile(r"^\s*\w+\s*\+\s*\d+\s*>=\s*\w+\s*$"),
    re.compile(r"^\s*\w+\s*>\s*\d+\s*→\s*\w+\s*>\s*0\s*$"),
    re.compile(r"^\s*\w+\s*<\s*0\s*→\s*-\w+\s*>\s*0\s*$"),
    re.compile(r"^\s*\w+\s*>\s*0\s*→\s*-\w+\s*<\s*0\s*$"),
]

_SIMPLE_EQUALITY = re.compile(r"^\s*\w+\s*=\s*\w+\s*$")


class HoleFiller:
    """Fills ``sorry`` holes in generated Lean code with appropriate proof tactics."""

    def __init__(self) -> None:
        self._fill_count = 0
        self._total_holes = 0

    def fill_all_holes(self, lean_code: str, specs: SpecCollection) -> str:
        """Scan the generated Lean code for ``sorry`` placeholders and replace
        each one with a proof block determined by the obligation's predicate."""
        self._total_holes = lean_code.count("  sorry")
        self._fill_count = 0

        obligations_by_name: Dict[str, ProofObligation] = {}
        for ob in specs.all:
            name = ob.id.replace(":", "_").replace(".", "_").replace("-", "_")
            obligations_by_name[name] = ob

        result = lean_code
        theorem_pattern = re.compile(
            r"(theorem\s+(\S+)(?:\s+.*?)?\s*:=\s*\n\s+)sorry",
            re.DOTALL,
        )

        def _replacer(match: re.Match) -> str:
            theorem_name = match.group(2)
            prefix = match.group(1)

            ob = obligations_by_name.get(theorem_name)
            if ob is None:
                for name, candidate in obligations_by_name.items():
                    if theorem_name.startswith(name) or name.startswith(theorem_name):
                        ob = candidate
                        break

            # Fallback: extract function name from theorem name pattern (funcname_correct[_N])
            # and match against the obligation's function field + kind + index.
            if ob is None:
                func_match = re.match(r'^(\w+?)(?:_correct(?:_(\d+))?)?$', theorem_name)
                if func_match:
                    th_func_name = func_match.group(1)
                    th_idx = int(func_match.group(2)) if func_match.group(2) else 0
                    # Collect postcondition obligations for this function
                    func_obs = [
                        ob for ob in specs.all
                        if ob.function == th_func_name and ob.kind == ObligationKind.POSTCONDITION
                    ]
                    if th_idx < len(func_obs):
                        ob = func_obs[th_idx]

            if ob is not None:
                tactic = self._select_tactic(ob)
                if tactic and tactic != "sorry":
                    self._fill_count += 1
                    return prefix + f"  by\n    {tactic}"

            tactic = self._infer_from_predicate(theorem_name)
            if tactic and tactic != "sorry":
                self._fill_count += 1
                return prefix + f"  by\n    {tactic}"

            return match.group(0)

        result = theorem_pattern.sub(_replacer, result)

        inline_pattern = re.compile(r"(by\s*\n\s+)sorry")
        result = inline_pattern.sub(lambda m: m.group(1) + "simp", result)

        return result

    def fill_hole(self, obligation: ProofObligation) -> str:
        """Fill a single proof obligation's hole with the appropriate proof block."""
        tactic = self._select_tactic(obligation)
        if tactic and tactic != "sorry":
            self._fill_count += 1
            return f"  by\n    {tactic}"
        return "  sorry"

    @staticmethod
    def _strip_parens(s: str) -> str:
        """Strip balanced outer parentheses iteratively."""
        result = s.strip()
        while result.startswith("(") and result.endswith(")"):
            inner = result[1:-1].strip()
            if inner.count("(") == inner.count(")"):
                result = inner
            else:
                break
        return result

    def classify_hole(self, obligation: ProofObligation) -> HoleDifficulty:
        """Classify the difficulty of a proof obligation."""
        pred = self._strip_parens(obligation.predicate)
        kind = obligation.kind
        pred_norm = pred.replace("==", "=").replace("!=", "≠")

        # Kind-based classification
        if kind in (ObligationKind.LOOP_INVARIANT, ObligationKind.LOOP_TERMINATION):
            return HoleDifficulty.MEDIUM
        if kind == ObligationKind.SHAPE_CONDITION:
            return HoleDifficulty.MEDIUM

        # Trivial: reflexivity, True
        if pred_norm in ("True", "true"):
            return HoleDifficulty.TRIVIAL
        for pat in _TRIVIAL_PATTERNS:
            if pat.match(pred_norm):
                return HoleDifficulty.TRIVIAL
        parts = pred_norm.split("=")
        if len(parts) == 2 and parts[0].strip() == parts[1].strip():
            return HoleDifficulty.TRIVIAL

        # Easy: omega/simp patterns
        for pat in _SIMP_SIMP_PATTERNS:
            if pat.match(pred_norm):
                return HoleDifficulty.EASY
        for pat in _OMEGA_PATTERNS:
            if pat.match(pred_norm):
                return HoleDifficulty.EASY
        if _SIMPLE_EQUALITY.match(pred_norm):
            return HoleDifficulty.EASY

        return HoleDifficulty.HARD

    def _select_tactic(self, obligation: ProofObligation) -> str:
        """Select the best tactic for the given obligation."""
        difficulty = self.classify_hole(obligation)
        pred = self._strip_parens(obligation.predicate)

        if difficulty == HoleDifficulty.TRIVIAL:
            if pred in ("True", "true"):
                return "trivial"
            if "=" in pred and not any(c in pred for c in ("∀", "∃", "→", "∧")):
                return "rfl"
            return "simp"

        if difficulty == HoleDifficulty.EASY:
            for pat in _OMEGA_PATTERNS:
                if pat.match(pred):
                    return "omega"
            for pat in _SIMP_SIMP_PATTERNS:
                if pat.match(pred):
                    return "simp"
            if any(c.isdigit() for c in pred):
                return "omega"
            return "simp"

        if difficulty == HoleDifficulty.MEDIUM:
            if obligation.kind == ObligationKind.LOOP_INVARIANT:
                return "simp"
            if obligation.kind in (ObligationKind.LOOP_TERMINATION, ObligationKind.SHAPE_CONDITION):
                return "sorry"
            return "simp"

        return "sorry"

    def _infer_from_predicate(self, pred: str) -> str:
        """Try to infer a proof tactic from a predicate string alone."""
        stripped = pred.strip()
        for pat in _SIMP_SIMP_PATTERNS:
            if pat.match(stripped):
                return "simp"
        for pat in _OMEGA_PATTERNS:
            if pat.match(stripped):
                return "omega"
        if stripped in ("True", "true"):
            return "trivial"
        if " = " in stripped:
            return "simp"
        return "sorry"

    @property
    def fill_rate(self) -> float:
        if self._total_holes == 0:
            return 1.0
        return self._fill_count / self._total_holes
