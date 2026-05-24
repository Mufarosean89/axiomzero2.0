"""Axiom Zero — ProofState Builder

Bridges Phase 1 (AST extraction, abstract interpretation, spec ingestion)
and Phase 2 (proof engine). Converts proof obligations and abstract state
into ProofState objects that the RL agent can act on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from proof_engine.proof_state import ProofState, Goal, GoalStatus, Hypothesis, TacticStep
from spec_ingestion.obligations import ProofObligation, SpecCollection, ObligationKind, ObligationStatus


def obligation_to_goal(obligation: ProofObligation, goal_id: Optional[str] = None) -> Goal:
    """Convert a ProofObligation from Phase 1 to a Goal for Phase 2.

    Each obligation becomes a single open goal:
    - The predicate as the goal type (what needs to be proved)
    - Context variables converted to hypotheses
    - Existing obligation hypotheses preserved
    """
    gid = goal_id or obligation.id

    hypotheses: List[Hypothesis] = [
        Hypothesis(name=name, type=typ, is_parameter=True)
        for name, typ in obligation.context.items()
    ]

    for idx, hyp_str in enumerate(obligation.hypotheses):
        if " : " in hyp_str:
            name, typ = hyp_str.split(" : ", 1)
            hypotheses.append(Hypothesis(name=name.strip(), type=typ.strip()))
        else:
            hypotheses.append(Hypothesis(name=f"h{idx}", type=hyp_str))

    return Goal(
        id=gid,
        type=obligation.predicate,
        hypotheses=hypotheses,
        status=GoalStatus.OPEN,
        depth=0,
    )


def build_proof_state(obligation: ProofObligation, theorem_name: Optional[str] = None) -> ProofState:
    """Build a complete ProofState from a single proof obligation.

    Creates a fresh proof state with one open goal representing the obligation's predicate.
    """
    name = theorem_name or f"{obligation.function}_{obligation.kind.name.lower()}"
    goal = obligation_to_goal(obligation)
    return ProofState(
        theorem_name=name,
        theorem_type=obligation.predicate,
        goals=[goal],
        closed_goals=[],
        depth=0,
        num_tactics_applied=0,
    )


def build_proof_state_collection(specs: SpecCollection, function_filter: Optional[str] = None) -> List[ProofState]:
    """Build a list of ProofStates from all obligations in a SpecCollection."""
    obligations = specs.all
    if function_filter:
        obligations = [o for o in obligations if o.function == function_filter]
    return [build_proof_state(ob) for ob in obligations]


def build_proof_state_grouped_by_function(specs: SpecCollection) -> Dict[str, List[ProofState]]:
    """Build ProofStates grouped by function name."""
    grouped: Dict[str, List[ProofState]] = {}
    for ob in specs.all:
        grouped.setdefault(ob.function, []).append(build_proof_state(ob))
    return grouped


_PYTHON_TO_LEAN_TYPE: Dict[str, str] = {
    "int": "ℤ",
    "float": "ℝ",
    "bool": "Bool",
    "str": "String",
    "torch.Tensor": "Tensor",
    "Tensor": "Tensor",
    "List": "List",
    "list": "List",
}


def _python_type_to_lean(py_type: str) -> str:
    """Convert a Python type string to a Lean 4 type."""
    return _PYTHON_TO_LEAN_TYPE.get(py_type, py_type)


def obligation_to_lean_theorem(obligation: ProofObligation) -> str:
    """Convert a proof obligation to a Lean 4 theorem skeleton with a ``sorry`` placeholder."""
    theorem_name = (
        obligation.id
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
    )

    binder_lines: List[str] = []
    for var_name, var_type in obligation.context.items():
        binder_lines.append(f"  ({var_name} : {_python_type_to_lean(var_type)})")
    for hyp_str in obligation.hypotheses:
        binder_lines.append(f"  (h : {hyp_str})")

    lines: List[str] = []
    if binder_lines:
        lines.append(f"theorem {theorem_name}")
        lines.extend(binder_lines)
        lines.append(f"  : {obligation.predicate} :=")
    else:
        lines.append(f"theorem {theorem_name} : {obligation.predicate} :=")

    lines.append("  sorry")
    return "\n".join(lines)


def to_proof_state(ir, abstract_state, specs) -> List[ProofState]:
    """Full end-to-end bridge: NormalizedIR -> list of ProofStates.

    Convenience function that takes the outputs of Phase 1 and produces
    the inputs for Phase 2 (proof engine).
    """
    return build_proof_state_collection(specs)
