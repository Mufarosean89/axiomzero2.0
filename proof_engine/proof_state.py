"""Axiom Zero — Proof State

Represents the state of a proof being constructed in Lean 4. The proof state
is the "game board" that the RL agent observes: open goals, local hypotheses,
and the tactic history that led to the current state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class GoalStatus(Enum):
    """Status of an individual goal in the proof tree."""
    OPEN = auto()
    PROVEN = auto()
    FAILED = auto()
    SUSPENDED = auto()

    @property
    def is_closed(self) -> bool:
        return self in (GoalStatus.PROVEN, GoalStatus.FAILED)


@dataclass
class Hypothesis:
    """A hypothesis available in the proof context.

    Attributes:
        name: Binder name (e.g., "h", "x", "H").
        type: The hypothesis type as a Lean string.
        is_parameter: Whether this is a function parameter.
        is_inductive: Whether this is an inductive hypothesis.
    """
    name: str
    type: str
    is_parameter: bool = False
    is_inductive: bool = False

    def to_lean(self) -> str:
        return f"{self.name} : {self.type}"


@dataclass
class Goal:
    """A single proof goal — something that needs to be proven.

    Attributes:
        id: Unique identifier for this goal.
        type: The goal type (what we need to prove).
        hypotheses: Available hypotheses in the local context.
        status: Current status of this goal.
        tactic_applied: The last tactic applied to this goal.
        depth: How deep in the goal tree this is.
        parent_goal_id: ID of the parent goal, if this is a subgoal.
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
        """Render this goal as a Lean 4 ``show`` block."""
        lines = [h.to_lean() for h in self.hypotheses]
        lines.append(f"  ⊢ {self.type}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Goal({self.id[:8]}... ⊢ {self.type[:50]} [{self.status.name}])"


@dataclass
class TacticStep:
    """A single step in the proof, recording what tactic was applied.

    Attributes:
        tactic: The tactic string that was executed.
        before_state: Goal IDs before applying the tactic.
        after_state: New goal IDs after applying the tactic.
        success: Whether the tactic succeeded.
        error: Error message if it failed.
        reward: The reward for this step (for RL).
    """
    tactic: str
    before_state: List[str]
    after_state: List[str]
    success: bool = True
    error: Optional[str] = None
    reward: float = 0.0


@dataclass
class ProofState:
    """The complete proof state at a point in the proof tree.

    This is the "game board" the RL agent observes to decide the next tactic.

    Attributes:
        theorem_name: Name of the theorem being proved.
        theorem_type: The type/statement of the theorem.
        goals: List of open (and possibly suspended) goals.
        closed_goals: Goals that have already been proven.
        tactic_history: Sequence of tactics applied so far.
        depth: Current depth in the proof tree.
        num_tactics_applied: Total tactics applied.
        local_decls: Other local declarations available.
        source_file: Path to the Lean source file.
        is_complete: Whether the proof is finished.
        error: Error message if the proof failed.
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
        return [g for g in self.goals if g.status == GoalStatus.OPEN]

    @property
    def num_open_goals(self) -> int:
        return len(self.open_goals)

    @property
    def is_finished(self) -> bool:
        return self.num_open_goals == 0 and not self.error

    def to_observation(self) -> Dict[str, Any]:
        """Convert the proof state to an observation dict for the RL agent."""
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
                {"tactic": step.tactic, "success": step.success}
                for step in self.tactic_history[-10:]
            ],
        }

    def apply_tactic(self, tactic: str, success: bool, new_goals: List[Goal], error: Optional[str] = None) -> None:
        """Record a tactic application and update the proof state."""
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
            for g in self.goals:
                if g.id not in after_ids:
                    g.status = GoalStatus.PROVEN
                    self.closed_goals.append(g)
            self.goals = new_goals
            self.depth += 1
            if self.num_open_goals == 0:
                self.is_complete = True
        else:
            self.error = error

    def _compute_tactic_reward(self, step: TacticStep) -> float:
        """Compute the reward for a tactic step."""
        if not step.success:
            return -1.0
        goals_before = len(step.before_state)
        goals_after = len(step.after_state)
        goals_reduced = goals_before - goals_after
        reward = 0.1 * goals_reduced
        if goals_after == 0:
            reward += 1.0
        return reward

    def summarize(self) -> str:
        """Return a human-readable summary of the proof state."""
        lines = [
            f"Theorem: {self.theorem_name}",
            f"  Statement: {self.theorem_type}",
            f"  Open goals: {self.num_open_goals}",
            f"  Tactics applied: {self.num_tactics_applied}",
            f"  Depth: {self.depth}",
            f"  Complete: {self.is_complete}",
            "",
        ]
        if self.error:
            lines.append(f"  Error: {self.error}\n")

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
