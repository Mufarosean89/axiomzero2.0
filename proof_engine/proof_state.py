"""
Axiom Zero - Proof State

Represents the state of a proof being constructed in Lean 4.
The proof state is the "game board" that the RL agent observes.

A proof state consists of:
- A list of open goals (things remaining to prove)
- For each goal: the target type + local context (hypotheses/variables)
- The tactic history (how we got here)
- Metadata about the environment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


class GoalStatus(Enum):
    """Status of an individual goal."""
    OPEN = auto()          # Not yet proven
    PROVEN = auto()        # Successfully closed
    FAILED = auto()        # Tactic application failed
    SUSPENDED = auto()     # Deferred (e.g., have/let)

    @property
    def is_closed(self) -> bool:
        return self in (GoalStatus.PROVEN, GoalStatus.FAILED)


@dataclass
class Hypothesis:
    """
    A hypothesis available in the proof context.
    
    Attributes:
        name: The binder name (e.g., "h", "h1", "x", "H")
        type: The type of the hypothesis as a string
        is_goal: Whether this is the goal itself
        is_parameter: Whether this is a function parameter
    """
    name: str
    type: str
    is_goal: bool = False
    is_parameter: bool = False
    is_inductive: bool = False  # Whether this is an inductive hypothesis

    def to_lean(self) -> str:
        """Convert to Lean 4 syntax."""
        return f"{self.name} : {self.type}"

    def __repr__(self) -> str:
        return f"{self.name}: {self.type}"


@dataclass
class Goal:
    """
    A single proof goal — something that needs to be proven.
    
    Attributes:
        id: Unique identifier for this goal
        type: The type of the goal (what we need to prove)
        hypotheses: Available hypotheses in the local context
        status: Current status of this goal
        tactic_applied: The last tactic applied to this goal
        depth: How deep in the goal tree this is
    """
    id: str
    type: str
    hypotheses: List[Hypothesis] = field(default_factory=list)
    status: GoalStatus = GoalStatus.OPEN
    tactic_applied: Optional[str] = None
    depth: int = 0
    parent_goal_id: Optional[str] = None

    @property
    def is_closed(self) -> bool:
        return self.status.is_closed

    def to_lean(self) -> str:
        """Render this goal as a Lean 4 `show` block."""
        lines = [f"  ⊢ {self.type}"]
        hyps = [h.to_lean() for h in self.hypotheses if not h.is_goal]
        if hyps:
            lines = hyps + lines
        return "\n".join(lines)

    def __repr__(self) -> str:
        status_str = self.status.name
        return f"Goal({self.id[:8]}... ⊢ {self.type[:50]} [{status_str}])"


@dataclass
class TacticStep:
    """
    A single step in the proof, recording what tactic was applied.
    
    Attributes:
        tactic: The tactic string that was executed
        before_state: Goal IDs before applying the tactic
        after_state: New goal IDs after applying the tactic
        success: Whether the tactic succeeded
        error: Error message if it failed
        reward: The reward for this step (for RL)
    """
    tactic: str
    before_state: List[str]  # Goal IDs
    after_state: List[str]   # New goal IDs
    success: bool = True
    error: Optional[str] = None
    reward: float = 0.0


@dataclass
class ProofState:
    """
    The complete proof state at a point in the proof tree.
    
    This is the "game board" the RL agent observes to decide the next tactic.
    
    Attributes:
        theorem_name: Name of the theorem being proved
        theorem_type: The type/statement of the theorem
        goals: List of open goals
        closed_goals: List of already-proven goals
        tactic_history: Sequence of tactics applied so far
        depth: Current depth in the proof tree
        num_tactics_applied: Total tactics applied
        local_decls: Other local declarations available
    """
    theorem_name: str = ""
    theorem_type: str = ""
    goals: List[Goal] = field(default_factory=list)
    closed_goals: List[Goal] = field(default_factory=list)
    tactic_history: List[TacticStep] = field(default_factory=list)
    depth: int = 0
    num_tactics_applied: int = 0
    local_decls: Dict[str, str] = field(default_factory=dict)
    source_file: Optional[str] = None
    is_complete: bool = False
    error: Optional[str] = None

    @property
    def open_goals(self) -> List[Goal]:
        """Get only the open (unproven) goals."""
        return [g for g in self.goals if g.status == GoalStatus.OPEN]

    @property
    def num_open_goals(self) -> int:
        """Number of remaining open goals."""
        return len(self.open_goals)

    @property
    def is_finished(self) -> bool:
        """Check if the proof is complete (no open goals)."""
        return self.num_open_goals == 0 and not self.error

    def to_observation(self) -> Dict[str, Any]:
        """
        Convert the proof state to an observation dict for the RL agent.
        
        Returns:
            Dict containing structured observation of the proof state
        """
        return {
            "theorem": self.theorem_name,
            "num_open_goals": self.num_open_goals,
            "num_total_goals": len(self.goals),
            "num_tactics_applied": self.num_tactics_applied,
            "depth": self.depth,
            "is_complete": self.is_complete,
            "goals": [
                {
                    "id": g.id,
                    "type": g.type,
                    "num_hypotheses": len(g.hypotheses),
                    "hypotheses": [
                        {"name": h.name, "type": h.type, "is_inductive": h.is_inductive}
                        for h in g.hypotheses
                    ],
                }
                for g in self.open_goals
            ],
            "tactic_history": [
                {
                    "tactic": step.tactic,
                    "success": step.success,
                }
                for step in self.tactic_history[-10:]  # Last 10 steps
            ],
        }

    def apply_tactic(self, tactic: str, success: bool, new_goals: List[Goal], error: Optional[str] = None):
        """
        Record a tactic application and update the proof state.
        
        Args:
            tactic: The tactic string that was applied
            success: Whether it succeeded
            new_goals: New goals resulting from the tactic
            error: Error message if failed
        """
        before_ids = [g.id for g in self.goals]
        after_ids = [g.id for g in new_goals]

        step = TacticStep(
            tactic=tactic,
            before_state=before_ids,
            after_state=after_ids,
            success=success,
            error=error,
        )

        self.tactic_history.append(step)
        self.num_tactics_applied += 1

        if success:
            # Move closed goals to closed list
            for g in self.goals:
                if g.id not in after_ids:
                    g.status = GoalStatus.PROVEN
                    self.closed_goals.append(g)

            # Update open goals
            self.goals = new_goals
            self.depth += 1

            if self.num_open_goals == 0:
                self.is_complete = True
        else:
            self.error = error

    def _get_tactic_reward(self, step: TacticStep) -> float:
        """
        Compute the reward for a tactic step.
        
        Returns:
            +1.0 if goal was closed
            +0.1 per goal reduced
            -0.1 per tactic used (to encourage shorter proofs)
            -1.0 if tactic failed
        """
        if not step.success:
            return -1.0

        goals_before = len(step.before_state)
        goals_after = len(step.after_state)
        goals_reduced = goals_before - goals_after

        reward = 0.1 * goals_reduced
        if goals_after == 0:
            reward += 1.0  # Goal closed bonus

        return reward

    def summarize(self) -> str:
        """Return a human-readable summary of the proof state."""
        lines = [f"Theorem: {self.theorem_name}"]
        lines.append(f"  Statement: {self.theorem_type}")
        lines.append(f"  Open goals: {self.num_open_goals}")
        lines.append(f"  Tactics applied: {self.num_tactics_applied}")
        lines.append(f"  Depth: {self.depth}")
        lines.append(f"  Complete: {self.is_complete}")
        lines.append("")

        if self.error:
            lines.append(f"  Error: {self.error}")
            lines.append("")

        if self.open_goals:
            lines.append("  Open Goals:")
            for i, goal in enumerate(self.open_goals[:5]):
                lines.append(f"    Goal {i + 1}: {goal.type[:80]}")
                for hyp in goal.hypotheses[:5]:
                    lines.append(f"      {hyp.name}: {hyp.type[:60]}")
                if len(goal.hypotheses) > 5:
                    lines.append(f"      ... and {len(goal.hypotheses) - 5} more hypotheses")

            if len(self.open_goals) > 5:
                lines.append(f"    ... and {len(self.open_goals) - 5} more goals")

        if self.tactic_history:
            lines.append("\n  Recent Tactics:")
            for step in self.tactic_history[-5:]:
                status = "OK" if step.success else f"FAIL: {step.error}"
                lines.append(f"    [{status}] {step.tactic}")

        return "\n".join(lines)
