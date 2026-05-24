"""
Axiom Zero - Monte Carlo Tree Search (MCTS)
Phase 3: RL Agent

AlphaZero-style MCTS adapted for proof search.

Key differences from game MCTS
-------------------------------
- States are ProofState objects (immutable snapshots after each tactic).
- Actions are tactic strings from CORE_TACTICS (39 actions).
- The environment is *stochastic at the proof level*: the same tactic can
  result in different subgoals depending on Lean's elaboration.  We model
  this as deterministic for now (same tactic → same child state), which
  holds when Lean is deterministic for a given proof state.
- Terminal states: is_complete == True  → reward +1
                   max_depth reached    → reward 0
                   tactic failed        → reward -1 (not expanded further)
- We run MCTS *without* a live Lean server in Phase 3 simulation mode.
  The TacticSimulator (below) approximates tactic outcomes so the MCTS
  loop can be tested end-to-end.  When a real LeanEnv is available,
  swap it in via RealTacticSimulator:

    from proof_engine import LeanEnv, TacticExecutor
    lean = LeanEnv(lean_path="lean", verbose=True)
    lean.start()
    executor = TacticExecutor(lean)
    sim = RealTacticSimulator(lean, executor)

    mcts = MCTS(net, simulator=sim, num_simulations=50)

Algorithm  (AlphaZero UCB)
--------------------------
  U(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

  where:
    Q(s, a) = mean value of positions reached by action a from s
    P(s, a) = prior probability from the policy network
    N(s)    = visit count of parent node s
    N(s, a) = visit count of edge (s, a)
    c_puct  = exploration constant (default 1.5)
"""

from __future__ import annotations

import math
import random
import copy
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from proof_engine import (
    ProofState, Goal, Hypothesis, GoalStatus,
    CORE_TACTICS, suggest_tactics_for_goal,
)
from proof_engine.lean_env import LeanEnv
from proof_engine.tactics import TacticExecutor
from .encoder import encode

# ── Constants ────────────────────────────────────────────────────────────────
C_PUCT = 1.5
MAX_DEPTH = 15          # max tactics before declaring timeout
DIRICHLET_ALPHA = 0.3   # noise added to root priors (exploration)
DIRICHLET_EPSILON = 0.25


# ── Helpers ──────────────────────────────────────────────────────────────────

def _python_predicate_to_lean(predicate: str) -> str:
    """Convert Python-style predicate syntax to Lean 4.

    Replaces ``==`` with ``=``, ``!=`` with ``≠``,
    `` and `` with `` ∧ ``, `` or `` with `` ∨ ``, etc.
    """
    pred = predicate.strip()
    pred = pred.replace("==", " = ")
    pred = pred.replace("!=", " ≠ ")
    pred = pred.replace(" and ", " ∧ ")
    pred = pred.replace(" or ", " ∨ ")
    pred = pred.replace("not ", "¬ ")
    # Collapse multiple spaces left by replacements
    pred = re.sub(r'\s+', ' ', pred)
    return pred


# ── Tactic Simulator (Phase 3 stand-in for a live LeanEnv) ──────────────────

class TacticSimulator:
    """
    Simulates tactic application without a running Lean server.

    Used for end-to-end testing of the MCTS loop.  The simulator uses the
    heuristic `suggest_tactics_for_goal` to decide whether a tactic is
    "likely to succeed" and produces plausible synthetic proof states.

    Replace this with a real TacticExecutor backed by LeanEnv for production.
    """

    def apply(
        self, state: ProofState, tactic: str
    ) -> Tuple[bool, ProofState, float]:
        """
        Apply `tactic` to `state`.

        Returns:
            (success, new_state, reward)
            reward: +1.0 if proof complete, -0.1 per tactic (cost), -1.0 on fail
        """
        if not state.open_goals:
            return False, state, 0.0

        goal = state.open_goals[0]
        suggestions = suggest_tactics_for_goal(goal)
        base_tactic = tactic.split()[0]
        is_suggested = base_tactic in suggestions

        # Deterministic success: tactic succeeds iff it is suggested for the goal
        success = is_suggested

        if not success:
            new_state = copy.deepcopy(state)
            new_state.error = f"Tactic '{tactic}' failed (simulated)"
            return False, new_state, -1.0

        # Build a new synthetic state
        new_state = copy.deepcopy(state)
        new_goals: List[Goal] = []

        # Closing tactics: rfl, trivial, omega, decide, assumption, done
        closing = {"rfl", "trivial", "omega", "decide", "assumption", "done",
                   "ring", "norm_num", "linarith", "contradiction"}

        if base_tactic in closing:
            # Close the first open goal
            closed_goal = copy.deepcopy(goal)
            closed_goal.status = GoalStatus.PROVEN
            new_state.closed_goals.append(closed_goal)
            new_goals = new_state.goals[1:]  # remaining goals
        elif base_tactic in ("cases", "induction", "by_cases"):
            # Split into 2 sub-goals
            for suffix in ("_case1", "_case2"):
                sub = copy.deepcopy(goal)
                sub.id = goal.id + suffix
                sub.depth = goal.depth + 1
                sub.parent_goal_id = goal.id
                new_goals.append(sub)
            new_goals += new_state.goals[1:]
        elif base_tactic in ("intro", "intros"):
            # Move a hypothesis from goal type to context
            sub = copy.deepcopy(goal)
            sub.id = goal.id + "_intro"
            sub.depth = goal.depth + 1
            hyp_name = tactic.split()[1] if len(tactic.split()) > 1 else "h"
            sub.hypotheses.append(Hypothesis(name=hyp_name, type="Prop", is_parameter=True))
            new_goals = [sub] + new_state.goals[1:]
        else:
            # Generic: keep same goal but record progress
            sub = copy.deepcopy(goal)
            sub.id = goal.id + f"_{base_tactic}"
            sub.depth = goal.depth + 1
            new_goals = [sub] + new_state.goals[1:]

        new_state.goals = new_goals
        new_state.num_tactics_applied += 1
        new_state.depth += 1

        if not new_state.goals:
            new_state.is_complete = True
            reward = 1.0
        else:
            reward = -0.05  # small step cost

        return True, new_state, reward


# ── Real Tactic Simulator (production, backed by Lean 4 server) ─────────────

class RealTacticSimulator:
    """
    Production tactic simulator backed by a real Lean 4 server.

    Wraps ``LeanEnv`` + ``TacticExecutor`` to implement the
    ``apply(state, tactic)`` interface that MCTS expects.

    Each call to ``apply()`` creates a **fresh** Lean file with the full
    proof replayed up to that point, so MCTS can branch freely without
    cross-contamination between exploration paths.

    Usage::

        from proof_engine import LeanEnv, TacticExecutor
        from rl_agent import MCTS, RealTacticSimulator

        lean = LeanEnv(lean_path="lean", verbose=True)
        lean.start()
        executor = TacticExecutor(lean)
        sim = RealTacticSimulator(lean, executor)

        mcts = MCTS(net, simulator=sim, num_simulations=50)
        action_probs, value = mcts.search(initial_state)
    """

    def __init__(
        self,
        lean_env: LeanEnv,
        executor: TacticExecutor,
    ) -> None:
        """
        Args:
            lean_env: An initialised ``LeanEnv`` instance (already started).
            executor: A ``TacticExecutor`` instance wrapping ``lean_env``.
        """
        self.lean_env = lean_env
        self.executor = executor

    def apply(
        self, state: ProofState, tactic: str
    ) -> Tuple[bool, ProofState, float]:
        """
        Apply ``tactic`` to ``state`` using the real Lean 4 server.

        Creates a fresh Lean file replaying the full tactic history,
        applies the new tactic, and reports the resulting goals.

        Returns:
            (success, new_state, reward)
        """
        if not state.open_goals:
            return False, state, 0.0

        # Resolve tactic name → full Lean command via the executor
        tactic_str = self.executor._resolve_tactic(
            tactic,
            goal=state.open_goals[0] if state.open_goals else None,
        )
        if tactic_str is None:
            tactic_str = tactic  # fall through to raw string

        # Build Lean source with all previous tactics replayed + a ``sorry``
        source = self._build_lean_source(state)

        # Create a fresh file (unique name to avoid collisions between branches)
        file_id = uuid.uuid4().hex[:12]
        self.lean_env.open_file(source, f"_mcts_{file_id}")

        # Apply the tactic via LeanEnv's apply_tactic (replaces ``sorry``,
        # sends LSP didChange, queries goals via RPC)
        response = self.lean_env.apply_tactic(tactic_str)

        success = response.get("success", False)
        error = response.get("error")
        new_goals_raw = response.get("goals", [])

        new_state = copy.deepcopy(state)

        if success:
            # Convert raw goals to Goal objects if needed
            goal_objects = []
            for g in new_goals_raw:
                if isinstance(g, Goal):
                    goal_objects.append(g)
                else:
                    from proof_engine.proof_state import Goal, Hypothesis, GoalStatus
                    goal_objects.append(Goal(
                        id=g.get("id", f"g_{len(goal_objects)}"),
                        type=g.get("type", g.get("target", "?")),
                        hypotheses=[
                            Hypothesis(
                                name=h.get("name", "?"),
                                type=h.get("type", "?"),
                            )
                            for h in g.get("hyps", g.get("hypotheses", []))
                        ],
                    ))

            # Record the tactic step in history and update open/closed goals
            new_state.apply_tactic(tactic_str, True, goal_objects)

            if not new_state.open_goals:
                new_state.is_complete = True
                reward = 1.0
            else:
                reward = -0.05
        else:
            new_state.error = error or "Unknown Lean server error"
            reward = -1.0

        return success, new_state, reward

    @staticmethod
    def _sanitize_lean_name(name: str) -> str:
        """Sanitize a string to be a valid Lean 4 identifier."""
        if not name:
            return "_axiom_zero_goal"
        # Replace any non-alphanumeric character except underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it doesn't start with a digit
        if sanitized and sanitized[0].isdigit():
            sanitized = "_" + sanitized
        return sanitized or "_axiom_zero_goal"

    def _build_lean_source(self, state: ProofState) -> str:
        """
        Build a Lean 4 theorem string from a ``ProofState``, replaying all
        previously successful tactics and ending with ``sorry`` so that
        ``LeanEnv.apply_tactic()`` can replace it with the next tactic.
        """
        theorem_type = _python_predicate_to_lean(state.theorem_type)
        name = self._sanitize_lean_name(state.theorem_name)

        lines = [
            "import Mathlib",
            "",
            "set_option pp.unicode.fun true",
            "",
            f"theorem {name} : {theorem_type} := by",
        ]

        # Replay all successful tactics from the history
        for step in state.tactic_history:
            if step.success and step.tactic:
                for t_line in step.tactic.split("\n"):
                    if t_line:  # skip empty lines
                        lines.append(f"  {t_line}")

        # Placeholder that ``apply_tactic`` will replace with the new tactic
        lines.append("  sorry")

        return "\n".join(lines)


# ── MCTS Node ────────────────────────────────────────────────────────────────

@dataclass
class MCTSNode:
    """A single node in the MCTS tree."""
    state: ProofState
    prior: float = 0.0          # P(s, a) from parent
    visit_count: int = 0        # N(s, a)
    value_sum: float = 0.0      # sum of backed-up values
    children: Dict[str, "MCTSNode"] = field(default_factory=dict)
    is_terminal: bool = False
    terminal_value: float = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visit_count: int) -> float:
        """AlphaZero UCB formula."""
        u = C_PUCT * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.q_value + u

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0


# ── MCTS Search ──────────────────────────────────────────────────────────────

class MCTS:
    """
    AlphaZero-style MCTS for proof search.

    Args:
        net          : PolicyValueNet — provides (priors, value) for a state
        simulator    : ``TacticSimulator`` (heuristic) or ``RealTacticSimulator``
                       (real Lean 4 server).  Defaults to ``TacticSimulator``.
        num_simulations: number of MCTS rollouts per search call
        c_puct       : exploration constant
    """

    def __init__(
        self,
        net,
        simulator: Union[TacticSimulator, RealTacticSimulator, None] = None,
        num_simulations: int = 50,
        c_puct: float = C_PUCT,
    ) -> None:
        self.net = net
        self.sim = simulator or TacticSimulator()
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self._tactic_list = list(CORE_TACTICS.keys())

    # ── Public API ────────────────────────────────────────────────────────

    def search(self, root_state: ProofState) -> Tuple[List[float], float]:
        """
        Run MCTS from root_state.

        Returns:
            (action_probs, root_value)
            action_probs : visit-count policy (length NUM_ACTIONS), sums to 1
            root_value   : estimated value of root
        """
        root = MCTSNode(state=root_state)
        self._expand(root)
        self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node, path = self._select(root)
            if node.is_terminal:
                value = node.terminal_value
            else:
                if not node.is_expanded:
                    value = self._expand(node)
                else:
                    value = node.q_value  # already expanded, just backup
            self._backup(path, value)

        # Derive action probabilities from visit counts
        total_visits = sum(
            child.visit_count for child in root.children.values()
        ) or 1
        action_probs = [0.0] * len(self._tactic_list)
        for tactic_name, child in root.children.items():
            if tactic_name in self._tactic_list:
                idx = self._tactic_list.index(tactic_name)
                action_probs[idx] = child.visit_count / total_visits

        return action_probs, root.q_value

    def best_action(self, root_state: ProofState) -> str:
        """Return the tactic with the highest visit count after search."""
        action_probs, _ = self.search(root_state)
        best_idx = max(range(len(action_probs)), key=lambda i: action_probs[i])
        return self._tactic_list[best_idx]

    # ── Internal methods ──────────────────────────────────────────────────

    def _expand(self, node: MCTSNode) -> float:
        """
        Expand node using the policy/value network and simulate all tactic outcomes.

        Each child gets its state set via the simulator immediately,
        so _select only needs to traverse without calling the simulator.

        Returns the value estimate for backup.
        """
        if node.state.is_complete:
            node.is_terminal = True
            node.terminal_value = 1.0
            return 1.0

        if node.state.depth >= MAX_DEPTH:
            node.is_terminal = True
            node.terminal_value = 0.0
            return 0.0

        # Get network predictions
        obs = node.state.to_observation()
        state_vec = encode(obs)
        priors, value = self.net.forward(state_vec)

        # Add a child for each action and simulate the tactic outcome immediately
        for i, tactic_name in enumerate(self._tactic_list):
            success, new_state, _ = self.sim.apply(node.state, tactic_name)
            child = MCTSNode(state=new_state, prior=priors[i])
            if not success:
                child.is_terminal = True
                child.terminal_value = -1.0
            elif new_state.is_complete:
                child.is_terminal = True
                child.terminal_value = 1.0
            node.children[tactic_name] = child

        return value

    def _select(self, root: MCTSNode) -> Tuple[MCTSNode, List[Tuple[MCTSNode, str]]]:
        """
        Traverse the tree from root, selecting children by UCB score.

        Simulation is done during _expand, so this method only traverses
        using UCB without calling the simulator.

        Returns:
            (leaf_node, path)
            path: list of (node, action) pairs leading to the leaf
        """
        node = root
        path: List[Tuple[MCTSNode, str]] = []

        while node.is_expanded and not node.is_terminal:
            # Pick the child with the highest UCB score
            best_action = max(
                node.children.keys(),
                key=lambda a: node.children[a].ucb_score(node.visit_count),
            )
            child = node.children[best_action]
            path.append((node, best_action))
            node = child

        return node, path

    def _backup(self, path: List[Tuple[MCTSNode, str]], value: float) -> None:
        """Back-propagate value up the tree."""
        for node, action in reversed(path):
            child = node.children[action]
            child.visit_count += 1
            child.value_sum += value
            node.visit_count += 1
            # Flip sign at each level (AlphaZero convention for 2-player;
            # for proof search we keep same sign — prover wants +1 always)

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """Add Dirichlet noise to root priors for exploration."""
        if not root.children:
            return
        # Simple symmetric Dirichlet sampling via gamma method
        alphas = [DIRICHLET_ALPHA] * len(root.children)
        gammas = [random.gammavariate(a, 1.0) for a in alphas]
        total = sum(gammas) or 1.0
        noise = [g / total for g in gammas]

        for i, child in enumerate(root.children.values()):
            child.prior = (
                (1 - DIRICHLET_EPSILON) * child.prior
                + DIRICHLET_EPSILON * noise[i]
            )
