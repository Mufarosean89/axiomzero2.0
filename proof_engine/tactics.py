"""
Axiom Zero - Tactics Module

Defines the tactic action space for the RL agent and provides a TacticExecutor
that sends tactics to the Lean 4 server and processes results.

The tactic action space starts with ~20 curated tactics covering common proof patterns:
- Introduction: intro, intro h, intros
- Application: apply, exact, refine, eapply
- Rewriting: rewrite, rw, simp
- Case analysis: cases, induction, by_cases
- Automation: omega, decide, ring, linarith
- Structural: have, let, calc, constructor
- Disjunction: left, right
- Quantifier: exists
- Simplification: dsimp, unfold, norm_num
- Other: trivial, rfl, done, assumption
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from .proof_state import ProofState, Goal, Hypothesis, GoalStatus


class TacticCategory(Enum):
    """Category of a tactic for the RL agent's policy network."""
    INTRODUCTION = auto()      # intro, intros
    APPLICATION = auto()       # apply, exact, refine
    REWRITING = auto()         # rewrite, simp
    CASE_ANALYSIS = auto()     # cases, induction
    AUTOMATION = auto()        # omega, decide, ring
    STRUCTURAL = auto()        # have, let, calc
    DISJUNCTION = auto()       # left, right
    EXISTENTIAL = auto()       # exists
    SIMPLIFICATION = auto()    # dsimp, unfold
    CLOSURE = auto()           # trivial, rfl, assumption
    CUSTOM = auto()            # User-defined/learned


@dataclass
class TacticTemplate:
    """
    A tactic template with holes to fill.
    
    Used for tactics that require arguments (e.g., `apply` needs a lemma name).
    The RL agent fills the holes based on the current proof state.
    
    Attributes:
        pattern: The tactic pattern with {hole} placeholders
        category: The tactic category
        description: Human-readable description
        requires_hypothesis: Whether this tactic needs a hypothesis name
        requires_term: Whether this tactic needs a term/expression
        num_holes: Number of holes to fill
    """
    pattern: str
    category: TacticCategory
    description: str = ""
    requires_hypothesis: bool = False
    requires_term: bool = False
    num_holes: int = 0

    def fill(self, *args: str) -> str:
        """Fill holes in the pattern with the given arguments."""
        try:
            return self.pattern.format(*args)
        except (IndexError, KeyError):
            return self.pattern

    def to_lean(self, *args: str) -> str:
        """Convert to a Lean 4 tactic string."""
        return self.fill(*args)


# ─── Core Tactic Templates ─────────────────────────────────────────────────

CORE_TACTICS: Dict[str, TacticTemplate] = {
    # Introduction
    "intro": TacticTemplate(
        pattern="intro {0}",
        category=TacticCategory.INTRODUCTION,
        description="Introduce a hypothesis or variable",
        requires_hypothesis=True,
        num_holes=1,
    ),
    "intros": TacticTemplate(
        pattern="intros",
        category=TacticCategory.INTRODUCTION,
        description="Introduce all hypotheses",
    ),
    "revert": TacticTemplate(
        pattern="revert {0}",
        category=TacticCategory.INTRODUCTION,
        description="Revert a hypothesis to a goal",
        requires_hypothesis=True,
        num_holes=1,
    ),

    # Application
    "apply": TacticTemplate(
        pattern="apply {0}",
        category=TacticCategory.APPLICATION,
        description="Apply a lemma or hypothesis to the goal",
        requires_term=True,
        num_holes=1,
    ),
    "exact": TacticTemplate(
        pattern="exact {0}",
        category=TacticCategory.APPLICATION,
        description="Provide an exact proof term",
        requires_term=True,
        num_holes=1,
    ),
    "refine": TacticTemplate(
        pattern="refine {0}",
        category=TacticCategory.APPLICATION,
        description="Refine the goal with a term containing holes",
        requires_term=True,
        num_holes=1,
    ),
    "eapply": TacticTemplate(
        pattern="eapply {0}",
        category=TacticCategory.APPLICATION,
        description="Apply with existential variables",
        requires_term=True,
        num_holes=1,
    ),

    # Rewriting
    "rewrite": TacticTemplate(
        pattern="rewrite [{0}]",
        category=TacticCategory.REWRITING,
        description="Rewrite using an equality",
        requires_term=True,
        num_holes=1,
    ),
    "rw": TacticTemplate(
        pattern="rw [{0}]",
        category=TacticCategory.REWRITING,
        description="Rewrite shorthand",
        requires_term=True,
        num_holes=1,
    ),
    "simp": TacticTemplate(
        pattern="simp",
        category=TacticCategory.REWRITING,
        description="Simplify using the simplifier",
    ),
    "simp_at": TacticTemplate(
        pattern="simp at {0}",
        category=TacticCategory.REWRITING,
        description="Simplify a hypothesis",
        requires_hypothesis=True,
        num_holes=1,
    ),

    # Case analysis
    "cases": TacticTemplate(
        pattern="cases {0}",
        category=TacticCategory.CASE_ANALYSIS,
        description="Case analysis on a hypothesis",
        requires_hypothesis=True,
        num_holes=1,
    ),
    "induction": TacticTemplate(
        pattern="induction {0}",
        category=TacticCategory.CASE_ANALYSIS,
        description="Induction on a hypothesis",
        requires_hypothesis=True,
        num_holes=1,
    ),
    "by_cases": TacticTemplate(
        pattern="by_cases h : {0}",
        category=TacticCategory.CASE_ANALYSIS,
        description="Case split on a proposition",
        requires_term=True,
        num_holes=1,
    ),

    # Automation
    "omega": TacticTemplate(
        pattern="omega",
        category=TacticCategory.AUTOMATION,
        description="Linear arithmetic solver",
    ),
    "decide": TacticTemplate(
        pattern="decide",
        category=TacticCategory.AUTOMATION,
        description="Decision procedure for decidable propositions",
    ),
    "ring": TacticTemplate(
        pattern="ring",
        category=TacticCategory.AUTOMATION,
        description="Ring algebra solver",
    ),
    "linarith": TacticTemplate(
        pattern="linarith",
        category=TacticCategory.AUTOMATION,
        description="Linear arithmetic solver",
    ),
    "nlinarith": TacticTemplate(
        pattern="nlinarith",
        category=TacticCategory.AUTOMATION,
        description="Non-linear arithmetic solver",
    ),
    "norm_num": TacticTemplate(
        pattern="norm_num",
        category=TacticCategory.AUTOMATION,
        description="Normalize numeric expressions",
    ),
    "positivity": TacticTemplate(
        pattern="positivity",
        category=TacticCategory.AUTOMATION,
        description="Prove positivity of expressions",
    ),

    # Structural
    "have": TacticTemplate(
        pattern="have h{0} : {1}",
        category=TacticCategory.STRUCTURAL,
        description="Introduce a new hypothesis",
        requires_term=True,
        num_holes=2,
    ),
    "let": TacticTemplate(
        pattern="let {0} := {1}",
        category=TacticCategory.STRUCTURAL,
        description="Define a local abbreviation",
        requires_term=True,
        num_holes=2,
    ),
    "calc": TacticTemplate(
        pattern="calc",
        category=TacticCategory.STRUCTURAL,
        description="Start a calculation block",
    ),
    "constructor": TacticTemplate(
        pattern="constructor",
        category=TacticCategory.STRUCTURAL,
        description="Apply the constructor of an inductive type",
    ),

    # Disjunction
    "left": TacticTemplate(
        pattern="left",
        category=TacticCategory.DISJUNCTION,
        description="Pick the left disjunct",
    ),
    "right": TacticTemplate(
        pattern="right",
        category=TacticCategory.DISJUNCTION,
        description="Pick the right disjunct",
    ),

    # Existential
    "exists": TacticTemplate(
        pattern="refine ⟨{0}, ?_⟩",
        category=TacticCategory.EXISTENTIAL,
        description="Provide a witness for an existential",
        requires_term=True,
        num_holes=1,
    ),
    "use": TacticTemplate(
        pattern="use {0}",
        category=TacticCategory.EXISTENTIAL,
        description="Use a witness for an existential",
        requires_term=True,
        num_holes=1,
    ),

    # Simplification
    "dsimp": TacticTemplate(
        pattern="dsimp",
        category=TacticCategory.SIMPLIFICATION,
        description="Definitional simplification",
    ),
    "dsimp_at": TacticTemplate(
        pattern="dsimp at {0}",
        category=TacticCategory.SIMPLIFICATION,
        description="Definitional simplification at hypothesis",
        requires_hypothesis=True,
        num_holes=1,
    ),
    "unfold": TacticTemplate(
        pattern="unfold {0}",
        category=TacticCategory.SIMPLIFICATION,
        description="Unfold a definition",
        requires_term=True,
        num_holes=1,
    ),

    # Closure
    "trivial": TacticTemplate(
        pattern="trivial",
        category=TacticCategory.CLOSURE,
        description="Prove simple true statements",
    ),
    "rfl": TacticTemplate(
        pattern="rfl",
        category=TacticCategory.CLOSURE,
        description="Reflexivity of equality",
    ),
    "rfl'": TacticTemplate(
        pattern="rfl",
        category=TacticCategory.CLOSURE,
        description="Reflexivity shorthand",
    ),
    "assumption": TacticTemplate(
        pattern="assumption",
        category=TacticCategory.CLOSURE,
        description="Find a hypothesis that matches the goal",
    ),
    "exfalso": TacticTemplate(
        pattern="exfalso",
        category=TacticCategory.CLOSURE,
        description="Replace goal with False",
    ),
    "contradiction": TacticTemplate(
        pattern="contradiction",
        category=TacticCategory.CLOSURE,
        description="Derive a contradiction from hypotheses",
    ),
    "done": TacticTemplate(
        pattern="done",
        category=TacticCategory.CLOSURE,
        description="Assert that there are no remaining goals",
    ),
}

# Tactics that take no arguments (can be applied directly)
ZERO_ARG_TACTICS = {
    name for name, t in CORE_TACTICS.items()
    if t.num_holes == 0
}

# Tactics that need a hypothesis name
ONE_HYP_TACTICS = {
    name for name, t in CORE_TACTICS.items()
    if t.requires_hypothesis and t.num_holes == 1
}


# ─── Tactic Executor ──────────────────────────────────────────────────────

@dataclass
class TacticResult:
    """
    Result of applying a tactic.
    
    Attributes:
        success: Whether the tactic was applied successfully
        tactic: The tactic string that was applied
        new_goals: New goals generated (empty if proof is complete)
        error: Error message if the tactic failed
        num_subgoals: Number of new subgoals created
    """
    success: bool
    tactic: str
    new_goals: List[Goal] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def num_subgoals(self) -> int:
        return len(self.new_goals)

    @property
    def is_proof_complete(self) -> bool:
        return self.success and len(self.new_goals) == 0


class TacticExecutor:
    """
    Executes tactics by communicating with the Lean 4 server.
    Translates high-level tactic names to Lean 4 syntax and sends them
    to the LeanEnv for execution.
    
    Usage:
        executor = TacticExecutor(lean_env)
        result = executor.apply("intro x")
        result = executor.apply("simp")
    """

    def __init__(self, lean_env):
        """
        Initialize the tactic executor.
        
        Args:
            lean_env: An initialized LeanEnv instance
        """
        self.lean_env = lean_env
        self.tactic_templates = dict(CORE_TACTICS)

    def apply(self, tactic: str, args: Optional[List[str]] = None) -> TacticResult:
        """
        Apply a tactic and return the result.
        
        Args:
            tactic: Tactic name or full tactic string
            args: Arguments for the tactic (used if tactic is a name)
            
        Returns:
            TacticResult with success status and new goals
        """
        # Resolve tactic to a full Lean 4 command
        tactic_str = self._resolve_tactic(tactic, args)

        if tactic_str is None:
            return TacticResult(
                success=False,
                tactic=tactic,
                error=f"Unknown tactic: {tactic}",
            )

        # Send to Lean server
        try:
            response = self.lean_env.run_tactic(tactic_str)
            return self._parse_response(tactic_str, response)
        except Exception as e:
            return TacticResult(
                success=False,
                tactic=tactic_str,
                error=str(e),
            )

    def apply_to_goal(self, tactic: str, goal: Goal, args: Optional[List[str]] = None) -> TacticResult:
        """
        Apply a tactic focused on a specific goal.
        
        Args:
            tactic: Tactic name or full tactic string
            goal: The specific goal to target
            args: Arguments for the tactic
            
        Returns:
            TacticResult
        """
        # Use focus notation: `· tactic` to apply to the first goal
        focused_tactic = f"· {tactic}"
        return self.apply(focused_tactic, args)

    def get_available_tactics(self, state: ProofState) -> List[Dict[str, Any]]:
        """
        Get the list of applicable tactics for the current proof state.
        Filters out tactics that are unlikely to be useful based on the goal type.
        
        Args:
            state: Current proof state
            
        Returns:
            List of tactic info dicts with name, category, description
        """
        available = []
        current_goal = state.open_goals[0] if state.open_goals else None

        for name, template in self.tactic_templates.items():
            info = {
                "name": name,
                "category": template.category.name,
                "description": template.description,
                "num_holes": template.num_holes,
                "requires_hypothesis": template.requires_hypothesis,
                "requires_term": template.requires_term,
            }

            # Filter tactics based on goal type
            if current_goal:
                goal_type = current_goal.type

                # Only suggest `left`/`right` for disjunction goals
                if name in ("left", "right") and not any(
                    kw in goal_type for kw in ("Or", "Sum", "∨")
                ):
                    continue

                # Only suggest `exists`/`use` for existential goals
                if name in ("exists", "use") and "∃" not in goal_type and "Exists" not in goal_type:
                    continue

                # Only suggest `constructor` for inductive goals
                if name == "constructor" and not any(
                    kw in goal_type for kw in ("And", "Or", "Prod", "Exists", "∧", "∨", "×")
                ):
                    continue

            available.append(info)

        return available

    def _resolve_tactic(self, tactic: str, args: Optional[List[str]] = None) -> Optional[str]:
        """
        Resolve a tactic name or string to a Lean 4 tactic command.
        
        Args:
            tactic: Tactic name or full tactic string
            args: Arguments for the tactic
            
        Returns:
            Lean 4 tactic string, or None if unknown
        """
        # If it's already a full tactic string, use it as-is
        if " " in tactic or tactic in ("simp", "omega", "decide", "ring", "rfl", "trivial", "done"):
            return tactic

        # Look up the template
        template = self.tactic_templates.get(tactic)
        if template is None:
            return None

        # Fill holes with provided args, or use defaults
        if args:
            return template.to_lean(*args)
        elif template.num_holes == 0:
            return template.to_lean()
        else:
            # Use placeholders for unfilled holes
            placeholders = [f"_{i}" for i in range(template.num_holes)]
            return template.to_lean(*placeholders)

    def _parse_response(self, tactic_str: str, response: Dict[str, Any]) -> TacticResult:
        """
        Parse the Lean server response into a TacticResult.
        
        Args:
            tactic_str: The tactic that was applied
            response: Raw response from the Lean server
            
        Returns:
            Structured TacticResult
        """
        success = response.get("success", False)
        error = response.get("error")
        goals_data = response.get("goals", [])

        new_goals = []
        if isinstance(goals_data, list):
            for i, g in enumerate(goals_data):
                if isinstance(g, Goal):
                    new_goals.append(g)
                elif isinstance(g, dict):
                    goal = Goal(
                        id=g.get("id", f"g_{i}"),
                        type=g.get("type", g.get("target", "?")),
                        hypotheses=[
                            Hypothesis(
                                name=h.get("name", "?"),
                                type=h.get("type", "?"),
                            )
                            for h in g.get("hypotheses", [])
                        ],
                        depth=i,
                    )
                    new_goals.append(goal)

        return TacticResult(
            success=success,
            tactic=tactic_str,
            new_goals=new_goals if success else [],
            error=error,
        )

    def register_custom_tactic(self, name: str, template: TacticTemplate):
        """
        Register a custom tactic template for the agent to use.
        Enables the system to learn new tactics over time.
        
        Args:
            name: Name for the tactic
            template: TacticTemplate defining the pattern
        """
        self.tactic_templates[name] = template


# ─── Tactic Prediction Helpers ─────────────────────────────────────────────

def suggest_tactics_for_goal(goal: Goal) -> List[str]:
    """
    Heuristically suggest tactics for a given goal.
    Used to seed the RL agent's search or provide fallback suggestions.
    
    Args:
        goal: The goal to analyze
        
    Returns:
        List of suggested tactic names
    """
    suggestions = ["simp", "rfl", "trivial", "assumption", "omega", "decide"]
    goal_type = goal.type

    # For implications / forall: try intro
    if "→" in goal_type or "∀" in goal_type or "->" in goal_type:
        suggestions.insert(0, "intro")

    # For conjunction: try constructor
    if "∧" in goal_type or "And" in goal_type or "×" in goal_type or "Prod" in goal_type:
        suggestions.insert(0, "constructor")

    # For disjunction: try left/right
    if "∨" in goal_type or "Or" in goal_type or "Sum" in goal_type:
        suggestions = ["left", "right"] + suggestions

    # For equality: try rfl, rewrite
    if "=" in goal_type or "≡" in goal_type:
        suggestions = ["rfl", "simp"] + suggestions
        # Check hypotheses for relevant equalities
        for hyp in goal.hypotheses:
            if "=" in hyp.type or "≡" in hyp.type:
                suggestions.insert(1, f"rw [{hyp.name}]")

    # For existential: try exists
    if "∃" in goal_type or "Exists" in goal_type:
        suggestions.insert(0, "exists")

    # For negation: try exfalso
    if "¬" in goal_type or "Not" in goal_type:
        suggestions.insert(0, "exfalso")

    return suggestions


def compute_tactic_embedding(tactic_name: str) -> List[float]:
    """
    Compute a simple one-hot-like embedding for a tactic.
    Used by the RL agent to represent tactic choices.
    
    Args:
        tactic_name: Name of the tactic
        
    Returns:
        List of floats representing the tactic embedding
    """
    tactic_list = list(CORE_TACTICS.keys())
    embedding = [0.0] * (len(tactic_list) + 1)

    if tactic_name in tactic_list:
        idx = tactic_list.index(tactic_name)
        embedding[idx] = 1.0
    else:
        embedding[-1] = 1.0  # Unknown tactic

    return embedding


def tactic_from_embedding(embedding: List[float]) -> Optional[str]:
    """
    Convert a tactic embedding back to a tactic name.
    
    Args:
        embedding: List of floats representing the tactic
        
    Returns:
        Tactic name, or None if unknown
    """
    tactic_list = list(CORE_TACTICS.keys())
    if len(embedding) != len(tactic_list) + 1:
        return None

    max_idx = max(range(len(embedding)), key=lambda i: embedding[i])
    if max_idx < len(tactic_list):
        return tactic_list[max_idx]

    return None
