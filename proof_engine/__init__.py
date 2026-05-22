"""
Axiom Zero - Proof Engine Module

The proof environment connects the RL agent to the Lean 4 theorem prover.
It manages a Lean 4 server via JSON-RPC, executes tactics, and tracks
the proof state (open goals, hypotheses, tactic history).

This is the "game engine" — the agent takes tactics as actions and receives
updated proof states as observations.

Public API:
    - LeanEnv: Lean 4 subprocess manager (start/stop/tactic execution)
    - TacticExecutor: Tactic action space (apply tactics to goals)
    - ProofState: Complete proof state (goals, history, summary)
    - Goal: A single proof goal with hypotheses
    - Hypothesis: A hypothesis in the goal context
    - TacticTemplate: A tactic pattern with holes
    - CORE_TACTICS: Dict of all available tactics
    - suggest_tactics_for_goal: Heuristic tactic suggestions
"""

from __future__ import annotations
from typing import Any, Dict, List

from .lean_env import LeanEnv, LeanServerError, JSONRPCError
from .proof_state import (
    ProofState,
    Goal,
    GoalStatus,
    Hypothesis,
    TacticStep,
)
from .tactics import (
    TacticExecutor,
    TacticTemplate,
    TacticResult,
    TacticCategory,
    CORE_TACTICS,
    ZERO_ARG_TACTICS,
    suggest_tactics_for_goal,
    compute_tactic_embedding,
    tactic_from_embedding,
)
from .builder import (
    obligation_to_goal,
    build_proof_state,
    build_proof_state_collection,
    build_proof_state_grouped_by_function,
    obligation_to_lean_theorem,
    to_proof_state,
)

__all__ = [
    # Environment
    "LeanEnv",
    "LeanServerError",
    "JSONRPCError",
    # Proof State
    "ProofState",
    "Goal",
    "GoalStatus",
    "Hypothesis",
    "TacticStep",
    # Tactics
    "TacticExecutor",
    "TacticTemplate",
    "TacticResult",
    "TacticCategory",
    "CORE_TACTICS",
    "ZERO_ARG_TACTICS",
    "suggest_tactics_for_goal",
    "compute_tactic_embedding",
    "tactic_from_embedding",
    # Builder / Bridge
    "obligation_to_goal",
    "build_proof_state",
    "build_proof_state_collection",
    "build_proof_state_grouped_by_function",
    "obligation_to_lean_theorem",
    "to_proof_state",
    # Convenience
    "create_proof_env",
    "prove_theorem",
]


def create_proof_env(lean_path: str = "lean", timeout: float = 30.0, verbose: bool = False) -> LeanEnv:
    """
    Create and start a Lean 4 proof environment.
    
    Args:
        lean_path: Path to the Lean 4 executable
        timeout: Timeout for server responses
        verbose: Enable debug output
        
    Returns:
        Started LeanEnv instance
    """
    env = LeanEnv(lean_path=lean_path, timeout=timeout, verbose=verbose)
    env.start()
    return env


def prove_theorem(
    env: LeanEnv,
    theorem_name: str,
    statement: str,
    tactic_sequence: List[str],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Attempt to prove a theorem using a sequence of tactics.
    
    This is a convenience function for testing and debugging.
    
    Args:
        env: Started LeanEnv instance
        theorem_name: Name of the theorem
        statement: Theorem statement in Lean syntax
        tactic_sequence: List of tactics to apply in sequence
        verbose: Print progress information
        
    Returns:
        Dict with keys: success, proof, error, steps_applied
    """
    # Create the theorem file
    file_path = env.create_temp_theorem(theorem_name, statement)

    executor = TacticExecutor(env)
    steps_applied = 0
    error = None

    for tactic in tactic_sequence:
        if verbose:
            print(f"  Applying: {tactic}")

        result = executor.apply(tactic)

        if not result.success:
            error = result.error
            if verbose:
                print(f"  Failed: {error}")
            break

        steps_applied += 1

        if result.is_proof_complete:
            if verbose:
                print(f"  Proof complete after {steps_applied} steps!")
            break

    # Read the proof file
    proof = ""
    try:
        with open(file_path, "r") as f:
            proof = f.read()
    except IOError:
        pass

    return {
        "success": steps_applied > 0 and error is None,
        "proof": proof,
        "error": error,
        "steps_applied": steps_applied,
    }
