"""
Axiom Zero - ProofState Builder

Bridges Phase 1 (AST extraction, abstract interpretation, spec ingestion)
and Phase 2 (proof engine). Converts proof obligations and abstract state
into ProofState objects that the RL agent can act on.

Exposes:
    - build_proof_state: Build a ProofState from a ProofObligation
    - build_proof_state_collection: Build multiple ProofStates from a SpecCollection
    - obligation_to_goal: Convert a ProofObligation to a Goal
    - to_proof_state: Full end-to-end from NormalizedIR → ProofState list
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from proof_engine.proof_state import (
    ProofState,
    Goal,
    GoalStatus,
    Hypothesis,
    TacticStep,
)
from spec_ingestion.obligations import (
    ProofObligation,
    SpecCollection,
    ObligationKind,
    ObligationStatus,
)


def obligation_to_goal(obligation: ProofObligation, goal_id: Optional[str] = None) -> Goal:
    """
    Convert a ProofObligation from Phase 1 to a Goal for Phase 2.

    Each obligation becomes a single open goal with:
    - The predicate as the goal type (what needs to be proved)
    - Context variables converted to hypotheses
    - Existing obligation hypotheses preserved

    Args:
        obligation: The proof obligation to convert
        goal_id: Optional custom goal ID (auto-generated if omitted)

    Returns:
        Goal suitable for use in the proof engine
    """
    gid = goal_id or obligation.id

    # Build hypotheses from context variables
    hypotheses: List[Hypothesis] = []
    for var_name, var_type in obligation.context.items():
        hypotheses.append(
            Hypothesis(
                name=var_name,
                type=var_type,
                is_parameter=True,
            )
        )

    # Add obligation-level hypotheses
    for hyp_str in obligation.hypotheses:
        # Parse "h : type" format or use as-is
        if " : " in hyp_str:
            name, typ = hyp_str.split(" : ", 1)
            hypotheses.append(
                Hypothesis(
                    name=name.strip(),
                    type=typ.strip(),
                )
            )
        else:
            hypotheses.append(
                Hypothesis(
                    name=f"h{len(hypotheses)}",
                    type=hyp_str,
                )
            )

    return Goal(
        id=gid,
        type=obligation.predicate,
        hypotheses=hypotheses,
        status=GoalStatus.OPEN,
        depth=0,
    )


def build_proof_state(
    obligation: ProofObligation,
    theorem_name: Optional[str] = None,
) -> ProofState:
    """
    Build a complete ProofState from a single proof obligation.

    This creates a fresh proof state with one open goal representing
    the obligation's predicate.

    Args:
        obligation: The proof obligation to turn into a proof state
        theorem_name: Optional name for the theorem (defaults to obligation function)

    Returns:
        ProofState ready for the tactic executor
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


def build_proof_state_collection(
    specs: SpecCollection,
    function_filter: Optional[str] = None,
) -> List[ProofState]:
    """
    Build a list of ProofStates from all obligations in a SpecCollection.

    Args:
        specs: The SpecCollection from Phase 1 spec ingestion
        function_filter: Optional function name to filter by

    Returns:
        List of ProofStates, one per obligation
    """
    obligations = specs.all
    if function_filter:
        obligations = [o for o in obligations if o.function == function_filter]

    return [build_proof_state(ob) for ob in obligations]


def build_proof_state_grouped_by_function(
    specs: SpecCollection,
) -> Dict[str, List[ProofState]]:
    """
    Build ProofStates grouped by function name.

    Useful for processing all obligations for a function together.
    Each function maps to a list of proof states (one per obligation).

    Args:
        specs: The SpecCollection from Phase 1

    Returns:
        Dict mapping function names to lists of ProofStates
    """
    grouped: Dict[str, List[ProofState]] = {}
    for ob in specs.all:
        func_name = ob.function
        if func_name not in grouped:
            grouped[func_name] = []
        grouped[func_name].append(build_proof_state(ob))
    return grouped


def obligation_to_lean_theorem(obligation: ProofObligation) -> str:
    """
    Convert a proof obligation to a Lean 4 theorem skeleton.

    Produces a Lean 4 theorem statement with a 'sorry' placeholder
    that can be written to a .lean file.

    Args:
        obligation: The proof obligation

    Returns:
        Lean 4 theorem string with 'sorry' body
    """
    theorem_name = obligation.id.replace(":", "_").replace(".", "_").replace("-", "_")

    # Build binder lines from context variables and hypotheses
    binder_lines: List[str] = []
    for var_name, var_type in obligation.context.items():
        lean_type = _python_type_to_lean(var_type)
        binder_lines.append(f"  ({var_name} : {lean_type})")
    for hyp_str in obligation.hypotheses:
        binder_lines.append(f"  (h : {hyp_str})")

    # Assemble the theorem declaration
    lines: List[str] = []
    if binder_lines:
        lines.append(f"theorem {theorem_name}")
        lines.extend(binder_lines)
        lines.append(f"  : {obligation.predicate} :=")
    else:
        lines.append(f"theorem {theorem_name} : {obligation.predicate} :=")

    lines.append("  sorry")
    return "\n".join(lines)


def _python_type_to_lean(py_type: str) -> str:
    """Convert a Python type string to a Lean 4 type."""
    mapping = {
        "int": "ℤ",
        "float": "ℝ",
        "bool": "Bool",
        "str": "String",
        "torch.Tensor": "Tensor",
        "Tensor": "Tensor",
        "List": "List",
        "list": "List",
    }
    if py_type in mapping:
        return mapping[py_type]
    return py_type


def to_proof_state(ir, abstract_state, specs) -> List[ProofState]:
    """
    Full end-to-end bridge: NormalizedIR → list of ProofStates.

    Convenience function that takes the outputs of Phase 1 and produces
    the inputs for Phase 2 (proof engine).

    Args:
        ir: NormalizedIR from ast_extractor.parse_source
        abstract_state: AbstractState from abstract_interpreter.analyze
        specs: SpecCollection from spec_ingestion.extract_specs

    Returns:
        List of ProofStates ready for the tactic executor
    """
    return build_proof_state_collection(specs)
