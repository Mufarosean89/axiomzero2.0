"""Axiom Zero — Tactics Module

Defines the tactic action space for the RL agent and provides a TacticExecutor
that sends tactics to the Lean 4 server and processes results.

The tactic action space covers ~30 curated tactics across categories:
introduction, application, rewriting, case analysis, automation, structural,
disjunction, existential, simplification, and closure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from .proof_state import ProofState, Goal, Hypothesis, GoalStatus
from .lemma_db import LemmaDatabase, LemmaSuggestion


class TacticCategory(Enum):
    """Category of a tactic for the RL agent's policy network."""
    INTRODUCTION = auto()
    APPLICATION = auto()
    REWRITING = auto()
    CASE_ANALYSIS = auto()
    AUTOMATION = auto()
    STRUCTURAL = auto()
    DISJUNCTION = auto()
    EXISTENTIAL = auto()
    SIMPLIFICATION = auto()
    CLOSURE = auto()
    CUSTOM = auto()


@dataclass
class TacticTemplate:
    """A tactic pattern with holes to fill.

    Attributes:
        pattern: The tactic pattern with ``{0}``, ``{1}`` placeholders.
        category: The tactic category.
        description: Human-readable description.
        requires_hypothesis: Whether this tactic needs a hypothesis name.
        requires_term: Whether this tactic needs a term/expression.
        num_holes: Number of holes to fill.
    """
    pattern: str
    category: TacticCategory
    description: str = ""
    requires_hypothesis: bool = False
    requires_term: bool = False
    num_holes: int = 0

    def fill(self, *args: str) -> str:
        try:
            return self.pattern.format(*args)
        except (IndexError, KeyError):
            return self.pattern

    def to_lean(self, *args: str) -> str:
        return self.fill(*args)


# ─── Core Tactic Templates ─────────────────────────────────────────────────

CORE_TACTICS: Dict[str, TacticTemplate] = {
    # Introduction
    "intro": TacticTemplate(pattern="intro {0}", category=TacticCategory.INTRODUCTION, description="Introduce a hypothesis", requires_hypothesis=True, num_holes=1),
    "intros": TacticTemplate(pattern="intros", category=TacticCategory.INTRODUCTION, description="Introduce all hypotheses"),
    "revert": TacticTemplate(pattern="revert {0}", category=TacticCategory.INTRODUCTION, description="Revert a hypothesis to a goal", requires_hypothesis=True, num_holes=1),

    # Application
    "apply": TacticTemplate(pattern="apply {0}", category=TacticCategory.APPLICATION, description="Apply a lemma or hypothesis to the goal", requires_term=True, num_holes=1),
    "exact": TacticTemplate(pattern="exact {0}", category=TacticCategory.APPLICATION, description="Provide an exact proof term", requires_term=True, num_holes=1),
    "refine": TacticTemplate(pattern="refine {0}", category=TacticCategory.APPLICATION, description="Refine the goal with a term containing holes", requires_term=True, num_holes=1),
    "eapply": TacticTemplate(pattern="eapply {0}", category=TacticCategory.APPLICATION, description="Apply with existential variables", requires_term=True, num_holes=1),

    # Rewriting
    "rewrite": TacticTemplate(pattern="rewrite [{0}]", category=TacticCategory.REWRITING, description="Rewrite using an equality", requires_term=True, num_holes=1),
    "rw": TacticTemplate(pattern="rw [{0}]", category=TacticCategory.REWRITING, description="Rewrite shorthand", requires_term=True, num_holes=1),
    "simp": TacticTemplate(pattern="simp", category=TacticCategory.REWRITING, description="Simplify using the simplifier"),
    "simp_at": TacticTemplate(pattern="simp at {0}", category=TacticCategory.REWRITING, description="Simplify a hypothesis", requires_hypothesis=True, num_holes=1),

    # Case analysis
    "cases": TacticTemplate(pattern="cases {0}", category=TacticCategory.CASE_ANALYSIS, description="Case analysis on a hypothesis", requires_hypothesis=True, num_holes=1),
    "induction": TacticTemplate(pattern="induction {0}", category=TacticCategory.CASE_ANALYSIS, description="Induction on a hypothesis", requires_hypothesis=True, num_holes=1),
    "by_cases": TacticTemplate(pattern="by_cases h : {0}", category=TacticCategory.CASE_ANALYSIS, description="Case split on a proposition", requires_term=True, num_holes=1),

    # Automation
    "omega": TacticTemplate(pattern="omega", category=TacticCategory.AUTOMATION, description="Linear arithmetic solver"),
    "decide": TacticTemplate(pattern="decide", category=TacticCategory.AUTOMATION, description="Decision procedure for decidable propositions"),
    "ring": TacticTemplate(pattern="ring", category=TacticCategory.AUTOMATION, description="Ring algebra solver"),
    "linarith": TacticTemplate(pattern="linarith", category=TacticCategory.AUTOMATION, description="Linear arithmetic solver"),
    "nlinarith": TacticTemplate(pattern="nlinarith", category=TacticCategory.AUTOMATION, description="Non-linear arithmetic solver"),
    "norm_num": TacticTemplate(pattern="norm_num", category=TacticCategory.AUTOMATION, description="Normalize numeric expressions"),
    "positivity": TacticTemplate(pattern="positivity", category=TacticCategory.AUTOMATION, description="Prove positivity of expressions"),

    # Structural
    "have": TacticTemplate(pattern="have h{0} : {1}", category=TacticCategory.STRUCTURAL, description="Introduce a new hypothesis", requires_term=True, num_holes=2),
    "let": TacticTemplate(pattern="let {0} := {1}", category=TacticCategory.STRUCTURAL, description="Define a local abbreviation", requires_term=True, num_holes=2),
    "calc": TacticTemplate(pattern="calc", category=TacticCategory.STRUCTURAL, description="Start a calculation block"),
    "constructor": TacticTemplate(pattern="constructor", category=TacticCategory.STRUCTURAL, description="Apply the constructor of an inductive type"),

    # Disjunction
    "left": TacticTemplate(pattern="left", category=TacticCategory.DISJUNCTION, description="Pick the left disjunct"),
    "right": TacticTemplate(pattern="right", category=TacticCategory.DISJUNCTION, description="Pick the right disjunct"),

    # Existential
    "exists": TacticTemplate(pattern="refine ⟨{0}, ?_⟩", category=TacticCategory.EXISTENTIAL, description="Provide a witness for an existential", requires_term=True, num_holes=1),
    "use": TacticTemplate(pattern="use {0}", category=TacticCategory.EXISTENTIAL, description="Use a witness for an existential", requires_term=True, num_holes=1),

    # Simplification
    "dsimp": TacticTemplate(pattern="dsimp", category=TacticCategory.SIMPLIFICATION, description="Definitional simplification"),
    "dsimp_at": TacticTemplate(pattern="dsimp at {0}", category=TacticCategory.SIMPLIFICATION, description="Definitional simplification at hypothesis", requires_hypothesis=True, num_holes=1),
    "unfold": TacticTemplate(pattern="unfold {0}", category=TacticCategory.SIMPLIFICATION, description="Unfold a definition", requires_term=True, num_holes=1),

    # Closure
    "trivial": TacticTemplate(pattern="trivial", category=TacticCategory.CLOSURE, description="Prove simple true statements"),
    "rfl": TacticTemplate(pattern="rfl", category=TacticCategory.CLOSURE, description="Reflexivity of equality"),
    "assumption": TacticTemplate(pattern="assumption", category=TacticCategory.CLOSURE, description="Find a hypothesis that matches the goal"),
    "exfalso": TacticTemplate(pattern="exfalso", category=TacticCategory.CLOSURE, description="Replace goal with False"),
    "contradiction": TacticTemplate(pattern="contradiction", category=TacticCategory.CLOSURE, description="Derive a contradiction from hypotheses"),
    "done": TacticTemplate(pattern="done", category=TacticCategory.CLOSURE, description="Assert that there are no remaining goals"),
}

ZERO_ARG_TACTICS: set = {name for name, t in CORE_TACTICS.items() if t.num_holes == 0}
ONE_HYP_TACTICS: set = {name for name, t in CORE_TACTICS.items() if t.requires_hypothesis and t.num_holes == 1}


# ─── Tactic Executor ──────────────────────────────────────────────────────

@dataclass
class TacticResult:
    """Result of applying a tactic.

    Attributes:
        success: Whether the tactic was applied successfully.
        tactic: The tactic string that was applied.
        new_goals: New goals generated (empty if proof is complete).
        error: Error message if the tactic failed.
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
    """Executes tactics by communicating with the Lean 4 server.

    Translates high-level tactic names to Lean 4 syntax and sends them to the LeanEnv.
    """

    def __init__(self, lean_env, lemma_db: Optional[LemmaDatabase] = None):
        self.lean_env = lean_env
        self.tactic_templates = dict(CORE_TACTICS)
        self.lemma_db = lemma_db or LemmaDatabase()

    def apply(self, tactic: str, args: Optional[List[str]] = None, goal: Optional[Goal] = None) -> TacticResult:
        """Apply a tactic and return the result.

        Args:
            tactic: Tactic name or full tactic string.
            args: Arguments for the tactic (used if tactic is a name).
            goal: Current goal context (used for lemma database lookups).

        Returns:
            TacticResult with success status and new goals.
        """
        tactic_str = self._resolve_tactic(tactic, args, goal=goal)
        if tactic_str is None:
            return TacticResult(success=False, tactic=tactic, error=f"Unknown tactic: {tactic}")

        try:
            response = self.lean_env.apply_tactic(tactic_str)
            return self._parse_response(tactic_str, response)
        except Exception as e:
            return TacticResult(success=False, tactic=tactic_str, error=str(e))

    def apply_to_goal(self, tactic: str, goal: Goal, args: Optional[List[str]] = None) -> TacticResult:
        """Apply a tactic focused on a specific goal using focus notation (``·``)."""
        return self.apply(f"· {tactic}", args, goal=goal)

    def get_available_tactics(self, state: ProofState) -> List[Dict[str, Any]]:
        """Get the list of applicable tactics for the current proof state.

        Filters out tactics unlikely to be useful based on the goal type.
        """
        available: List[Dict[str, Any]] = []
        current_goal = state.open_goals[0] if state.open_goals else None

        for name, template in self.tactic_templates.items():
            info: Dict[str, Any] = {
                "name": name,
                "category": template.category.name,
                "description": template.description,
                "num_holes": template.num_holes,
                "requires_hypothesis": template.requires_hypothesis,
                "requires_term": template.requires_term,
            }

            if current_goal:
                gt = current_goal.type
                if name in ("left", "right") and not any(kw in gt for kw in ("Or", "Sum", "∨")):
                    continue
                if name in ("exists", "use") and "∃" not in gt and "Exists" not in gt:
                    continue
                if name == "constructor" and not any(kw in gt for kw in ("And", "Or", "Prod", "Exists", "∧", "∨", "×")):
                    continue

                if template.requires_term and template.num_holes == 1:
                    suggestion = self.lemma_db.suggest_for_goal(gt, tactic=name)
                    if suggestion:
                        info["suggested_lemma"] = suggestion.lemma_name
                        info["suggested_tactic_rendered"] = suggestion.rendered
                        info["lemma_score"] = round(suggestion.score, 3)

            available.append(info)

        return available

    def _resolve_tactic(self, tactic: str, args: Optional[List[str]] = None, goal: Optional[Goal] = None) -> Optional[str]:
        """Resolve a tactic name or string to a Lean 4 tactic command."""
        if " " in tactic or tactic in ("simp", "omega", "decide", "ring", "rfl", "trivial", "done"):
            return tactic

        template = self.tactic_templates.get(tactic)
        if template is None:
            return None

        if args:
            return template.to_lean(*args)
        elif template.num_holes == 0:
            return template.to_lean()
        elif goal and template.requires_term and self.lemma_db:
            suggestion = self.lemma_db.suggest_for_goal(goal.type, tactic=tactic)
            if suggestion:
                return suggestion.rendered
            placeholders = [f"_{i}" for i in range(template.num_holes)]
            return template.to_lean(*placeholders)
        else:
            placeholders = [f"_{i}" for i in range(template.num_holes)]
            return template.to_lean(*placeholders)

    def _parse_response(self, tactic_str: str, response: Dict[str, Any]) -> TacticResult:
        """Parse the Lean server response into a TacticResult."""
        success = response.get("success", False)
        error = response.get("error")
        goals_data = response.get("goals", [])
        new_goals: List[Goal] = []

        if isinstance(goals_data, list):
            for i, g in enumerate(goals_data):
                if isinstance(g, Goal):
                    new_goals.append(g)
                elif isinstance(g, dict):
                    new_goals.append(Goal(
                        id=g.get("id", f"g_{i}"),
                        type=g.get("type", g.get("target", "?")),
                        hypotheses=[
                            Hypothesis(name=h.get("name", "?"), type=h.get("type", "?"))
                            for h in g.get("hypotheses", [])
                        ],
                        depth=i,
                    ))

        return TacticResult(
            success=success,
            tactic=tactic_str,
            new_goals=new_goals if success else [],
            error=error,
        )

    def suggest_lemma(self, tactic_name: str, goal: Goal) -> Optional[LemmaSuggestion]:
        """Get a lemma suggestion for a tactic applied to a goal."""
        return self.lemma_db.suggest_for_goal(goal.type, tactic=tactic_name)

    def search_lemmas(self, goal_type: str, top_k: int = 5) -> List[LemmaSuggestion]:
        """Search the lemma database for lemmas relevant to a goal type."""
        return self.lemma_db.search(goal_type, top_k=top_k)

    def register_custom_tactic(self, name: str, template: TacticTemplate) -> None:
        """Register a custom tactic template for the agent to use."""
        self.tactic_templates[name] = template


# ─── Tactic Prediction Helpers ─────────────────────────────────────────────

def suggest_tactics_for_goal(goal: Goal) -> List[str]:
    """Heuristically suggest tactics for a given goal.

    Used to seed the RL agent's search or provide fallback suggestions.
    """
    suggestions: List[str] = ["simp", "rfl", "trivial", "assumption", "omega", "decide"]
    gt = goal.type

    if "→" in gt or "∀" in gt or "->" in gt:
        suggestions.insert(0, "intro")
    if "∧" in gt or "And" in gt or "×" in gt or "Prod" in gt:
        suggestions.insert(0, "constructor")
    if "∨" in gt or "Or" in gt or "Sum" in gt:
        suggestions = ["left", "right"] + suggestions
    if "=" in gt or "≡" in gt:
        suggestions = ["rfl", "simp"] + suggestions
        for hyp in goal.hypotheses:
            if "=" in hyp.type or "≡" in hyp.type:
                suggestions.insert(1, f"rw [{hyp.name}]")
    if "∃" in gt or "Exists" in gt:
        suggestions.insert(0, "exists")
    if "¬" in gt or "Not" in gt:
        suggestions.insert(0, "exfalso")

    return suggestions


def compute_tactic_embedding(tactic_name: str) -> List[float]:
    """Compute a simple one-hot-like embedding for a tactic name."""
    tactic_list = list(CORE_TACTICS.keys())
    embedding = [0.0] * (len(tactic_list) + 1)
    if tactic_name in tactic_list:
        embedding[tactic_list.index(tactic_name)] = 1.0
    else:
        embedding[-1] = 1.0
    return embedding


def tactic_from_embedding(embedding: List[float]) -> Optional[str]:
    """Convert a tactic embedding back to a tactic name."""
    tactic_list = list(CORE_TACTICS.keys())
    if len(embedding) != len(tactic_list) + 1:
        return None
    max_idx = max(range(len(embedding)), key=lambda i: embedding[i])
    return tactic_list[max_idx] if max_idx < len(tactic_list) else None
