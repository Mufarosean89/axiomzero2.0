"""Dataset loaders for cold-start pre-training of the RL agent.

Provides parsers for public Lean proof datasets (LeanDojo, miniF2F, ProofNet,
Lean Workbook) and a built-in seed data generator using the Phase 4 benchmark
suite to bootstrap the replay buffer.
"""

from __future__ import annotations

import json
import os
import copy
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from proof_engine import ProofState, suggest_tactics_for_goal

from .encoder import encode
from .self_play import TrainingExample
from .mcts import TacticSimulator

# Lazy-loaded tactic list (same pattern as encoder._get_tactic_list)
_TACTIC_LIST: Optional[List[str]] = None


def _get_tactic_list() -> List[str]:
    global _TACTIC_LIST
    if _TACTIC_LIST is None:
        from proof_engine import CORE_TACTICS
        _TACTIC_LIST = list(CORE_TACTICS.keys())
    return _TACTIC_LIST


# ── Dataset format identifiers ───────────────────────────────────────────────

class DatasetFormat(Enum):
    """Supported external dataset formats."""
    LEAN_DOJO = auto()
    MINIF2F = auto()
    PROOF_NET = auto()
    LEAN_WORKBOOK = auto()
    BUILTIN = auto()


# ── Intermediate representation (dataset-agnostic) ───────────────────────────

@dataclass
class RawProofStep:
    """
    A single step from a human-written proof, in a dataset-agnostic format.

    This is the intermediate representation between the dataset parsers and
    the ``convert_to_training_examples()`` converter.

    Attributes:
        state_observation : The proof state observation dict (same schema as
                           ``ProofState.to_observation()``).
        tactic           : The tactic the human applied at this step.
        outcome          : Outcome of the full proof (+1 proved, -1 failed, 0 unknown).
        theorem_name     : Name of the theorem being proved.
    """
    state_observation: Dict[str, Any]
    tactic: str
    outcome: float = 0.0
    theorem_name: str = ""


@dataclass
class RawProofTrace:
    """A complete proof trace (multiple steps) from a human-written proof."""
    theorem_name: str = ""
    steps: List[RawProofStep] = field(default_factory=list)
    outcome: float = 0.0
    source: str = "unknown"


# ── Streaming JSON array reader ────────────────────────────────────────────

def _iter_json_array_objects(filepath: str, max_objects: Optional[int] = None):
    """Yield JSON objects from a large array file using a streaming reader.

    Reads in 64 KB chunks using ``json.JSONDecoder.raw_decode`` to avoid
    ``MemoryError`` on files >50 MB (e.g. the Lean Workbook).
    """
    decoder = json.JSONDecoder()
    with open(filepath, "rb") as f:
        buffer = ""
        # Find the opening bracket '['
        while True:
            chunk = f.read(65536).decode("utf-8", errors="replace")
            if not chunk:
                break
            buffer += chunk
            idx = buffer.find("[")
            if idx >= 0:
                buffer = buffer[idx + 1:]
                break

        count = 0
        while max_objects is None or count < max_objects:
            buffer = buffer.lstrip()
            if not buffer:
                chunk = f.read(65536).decode("utf-8", errors="replace")
                if not chunk:
                    break
                buffer += chunk
                continue

            if buffer.startswith("]"):
                break

            try:
                obj, pos = decoder.raw_decode(buffer)
                yield obj
                count += 1
                buffer = buffer[pos:]
                buffer = buffer.lstrip(", \t\n\r")
            except json.JSONDecodeError:
                chunk = f.read(65536).decode("utf-8", errors="replace")
                if not chunk:
                    break
                buffer += chunk


# ═════════════════════════════════════════════════════════════════════════════
# Dataset Parsers
# ═════════════════════════════════════════════════════════════════════════════

def _make_observation(
    theorem: str,
    goal_type: str,
    hypotheses: Optional[List[Dict[str, str]]] = None,
    step_idx: int = 0,
    prior_tactics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a standard observation dict shared across all dataset parsers."""
    prior = prior_tactics or []
    return {
        "theorem": theorem,
        "num_open_goals": 1,
        "num_total_goals": 1,
        "num_tactics_applied": step_idx,
        "depth": step_idx,
        "is_complete": False,
        "goals": [{
            "id": f"g_{step_idx}",
            "type": goal_type,
            "num_hypotheses": len(hypotheses or []),
            "hypotheses": hypotheses or [],
        }],
        "tactic_history": [
            {"tactic": t, "success": True}
            for t in prior
        ],
    }


def parse_leandojo_trace(path: str) -> List[RawProofTrace]:
    """Parse a LeanDojo ``traj.json`` trace file into proof traces."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"LeanDojo trace not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Normalise to a list of traces (single trace or list of traces)
    traces_data = raw if isinstance(raw, list) else [raw]

    result: List[RawProofTrace] = []
    for entry in traces_data:
        theorem = entry.get("theorem", entry.get("full_name", "unknown"))
        traj = entry.get("traj", entry.get("trace", []))
        result_str = entry.get("result", "")

        outcome = 1.0 if result_str in ("proved", "proven", "success") else 0.0

        steps: List[RawProofStep] = []
        for step_data in traj:
            state_obs = _normalize_leandojo_state(step_data.get("state", {}))
            tactic = step_data.get("tactic", "")
            if tactic:
                steps.append(RawProofStep(
                    state_observation=state_obs,
                    tactic=tactic,
                    outcome=outcome,
                    theorem_name=theorem,
                ))

        result.append(RawProofTrace(
            theorem_name=theorem,
            steps=steps,
            outcome=outcome,
            source="leandojo",
        ))

    return result


def _normalize_leandojo_state(ld_state: Dict[str, Any]) -> Dict[str, Any]:
    """Convert LeanDojo state to an observation dict compatible with encode()."""
    goal_str = ld_state.get("goal", ld_state.get("target", ""))
    hypotheses_raw = ld_state.get("hypotheses", ld_state.get("hyps", []))

    hypotheses: List[Dict[str, str]] = []
    for hyp in hypotheses_raw:
        if isinstance(hyp, dict):
            hypotheses.append({
                "name": hyp.get("name", "h"),
                "type": hyp.get("type", hyp.get("goal", "")),
            })
        elif isinstance(hyp, (list, tuple)) and len(hyp) >= 2:
            hypotheses.append({"name": str(hyp[0]), "type": str(hyp[1])})

    return _make_observation(
        theorem=ld_state.get("theorem", ""),
        goal_type=goal_str,
        hypotheses=hypotheses,
    )


def parse_minif2f(path: str) -> List[RawProofTrace]:
    """Parse a miniF2F JSON file into proof traces."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"miniF2F file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data if isinstance(data, list) else data.get("data", [])

    result: List[RawProofTrace] = []
    for entry in entries:
        theorem = entry.get("id", entry.get("name", "unknown"))
        formal_stmt = entry.get("formal_statement", "")
        formal_proof = entry.get("formal_proof", "")

        # Extract the goal type from the formal statement
        goal_str = _extract_goal_from_lean_statement(formal_stmt)

        steps: List[RawProofStep] = []
        if formal_proof:
            tactics = _split_lean_proof(formal_proof)
            for i, tactic in enumerate(tactics):
                steps.append(RawProofStep(
                    state_observation=_make_observation(
                        theorem=theorem,
                        goal_type=goal_str,
                        step_idx=i,
                        prior_tactics=tactics[:i],
                    ),
                    tactic=tactic,
                    outcome=1.0,
                    theorem_name=theorem,
                ))

        result.append(RawProofTrace(
            theorem_name=theorem,
            steps=steps,
            outcome=1.0 if formal_proof else 0.0,
            source="minif2f",
        ))

    return result


def _extract_theorem_name(formal_statement: str) -> str:
    """Extract the theorem name from a Lean formal statement.

    Handles formats like::

        "theorem lean_workbook_18 ... : ..."        -> "lean_workbook_18"
        "theorem add_comm (a b : ℕ) : ..."           -> "add_comm"
        "example : ..."                               -> "example"

    Args:
        formal_statement: A Lean theorem statement (e.g. from a JSON entry).

    Returns:
        The theorem name string, or ``"unknown"`` if it cannot be extracted.
    """
    stmt = formal_statement.strip()
    for prefix in ("theorem", "lemma", "def", "example"):
        if stmt.startswith(prefix):
            rest = stmt[len(prefix):].lstrip()
            # Name is the first identifier before ':', '(', or whitespace
            name = []
            for ch in rest:
                if ch in (" ", ":", "("):
                    break
                name.append(ch)
            return "".join(name) if name else prefix
    return "unknown"


def _extract_goal_from_lean_statement(statement: str) -> str:
    """Extract the goal type from a Lean theorem statement."""
    # Remove leading "theorem name ... :"
    goal = statement.strip()
    for prefix in ("theorem", "lemma", "def", "example"):
        if goal.startswith(prefix):
            # Find the ':' separator after the binder/name list
            idx = goal.find(":")
            if idx >= 0:
                goal = goal[idx + 1:].strip()
            break
    # Strip everything after ``:=`` (handles ``:= by sorry``, ``:=``, etc.)
    if ":=" in goal:
        goal = goal[:goal.index(":=")].strip()
    return goal.strip()


def _split_lean_proof(proof_block: str) -> List[str]:
    """Split a Lean proof block into individual tactic lines."""
    tactics: List[str] = []
    for line in proof_block.replace(";", "\n").split("\n"):
        line = line.strip()
        if line and not line.startswith("--") and not line.startswith("/-"):
            # Remove leading bullets
            cleaned = line.lstrip("·*+-").strip()
            if cleaned:
                tactics.append(cleaned)
    return tactics


def parse_proofnet(path: str) -> List[RawProofTrace]:
    """Parse a ProofNet JSONL file into proof traces."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ProofNet file not found: {path}")

    result: List[RawProofTrace] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            header = entry.get("header", {})
            theorem = header.get("problem_id", entry.get("id", "unknown"))
            formal_stmt = entry.get("formal_statement", "")
            formal_proof = entry.get("formal_proof", "")

            goal_str = _extract_goal_from_lean_statement(formal_stmt)
            tactics = _split_lean_proof(formal_proof)
            steps: List[RawProofStep] = []
            for i, tactic in enumerate(tactics):
                steps.append(RawProofStep(
                    state_observation=_make_observation(
                        theorem=theorem,
                        goal_type=goal_str,
                        step_idx=i,
                        prior_tactics=tactics[:i],
                    ),
                    tactic=tactic,
                    outcome=1.0,
                    theorem_name=theorem,
                ))

            result.append(RawProofTrace(
                theorem_name=theorem,
                steps=steps,
                outcome=1.0 if formal_proof else 0.0,
                source="proofnet",
            ))

        return result


def parse_lean_workbook(
    path: str,
    max_entries: Optional[int] = 500,
) -> List[RawProofTrace]:
    """Parse a Lean Workbook JSON file into proof traces.

    Skips entries with empty proofs. Uses streaming JSON reader to handle
    files >90 MB without ``MemoryError``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Lean Workbook file not found: {path}")

    result: List[RawProofTrace] = []

    for entry in _iter_json_array_objects(path, max_objects=max_entries):
        formal_statement = entry.get("formal_statement", "")
        proof_tactics = entry.get("proof", [])

        # Skip entries without any proof steps
        if not proof_tactics:
            continue

        theorem = _extract_theorem_name(formal_statement)
        goal_str = _extract_goal_from_lean_statement(formal_statement)

        steps: List[RawProofStep] = []
        for i, tactic in enumerate(proof_tactics):
            tactic = tactic.strip()
            if not tactic:
                continue
            # Skip standalone comment lines
            if tactic.startswith(("--", "/-")) and len(tactic) < 20:
                continue
            steps.append(RawProofStep(
                state_observation=_make_observation(
                    theorem=theorem,
                    goal_type=goal_str,
                    step_idx=i,
                    prior_tactics=proof_tactics[:i],
                ),
                tactic=tactic,
                outcome=1.0,
                theorem_name=theorem,
            ))

        if steps:
            result.append(RawProofTrace(
                theorem_name=theorem,
                steps=steps,
                outcome=1.0,
                source="lean_workbook",
            ))

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Built-in Seed Data Generator
# ═════════════════════════════════════════════════════════════════════════════

def generate_builtin_seed_data(
    proof_states: Optional[List["ProofState"]] = None,
    max_traces: int = 30,
    max_steps_per_trace: int = 10,
    temperature: float = 0.1,
    verbose: bool = False,
) -> List[RawProofTrace]:
    """Generate seed data from built-in proof states using heuristic tactic suggestions.

    Uses ``suggest_tactics_for_goal()`` to produce plausible proof traces
    without a trained network or MCTS.
    """
    from proof_engine import ProofState, suggest_tactics_for_goal
    from proof_engine.proof_state import GoalStatus

    if proof_states is None:
        proof_states = _load_benchmark_proof_states()

    traces: List[RawProofTrace] = []
    random.shuffle(proof_states)

    for ps in proof_states[:max_traces]:
        if verbose:
            print(f"  Generating seed trace for: {ps.theorem_name}")

        trace = _generate_single_trace(
            ps,
            max_steps=max_steps_per_trace,
            temperature=temperature,
        )
        traces.append(trace)

        if verbose:
            status = "[OK]" if trace.outcome > 0 else "[FAIL]"
            print(f"    {status} ({len(trace.steps)} steps, outcome={trace.outcome})")

    return traces


def _load_benchmark_proof_states() -> List["ProofState"]:
    """Load proof states from the Phase 4 benchmark suite."""
    from lean_compiler.benchmark_suite import BenchmarkSuite
    from ast_extractor import parse_source, normalize
    from abstract_interpreter import analyze
    from spec_ingestion import extract_specs
    from proof_engine import build_proof_state_collection

    suite = BenchmarkSuite()
    all_states: List["ProofState"] = []

    for problem in suite.problems:
        try:
            ir = parse_source(problem.source, f"bench_{problem.name}")
            ir = normalize(ir)
            abstract = analyze(ir)
            specs = extract_specs(ir, abstract)
            states = build_proof_state_collection(specs)
            all_states.extend(states)
        except Exception:
            # Skip problems that fail to parse
            continue

    return all_states


def _generate_single_trace(
    initial_state: "ProofState",
    max_steps: int = 10,
    temperature: float = 0.1,
) -> RawProofTrace:
    """Generate a single proof trace using heuristic tactic suggestions."""
    sim = TacticSimulator()
    steps: List[RawProofStep] = []
    current = copy.deepcopy(initial_state)

    for step_idx in range(max_steps):
        if current.is_complete or not current.open_goals:
            break

        goal = current.open_goals[0]
        suggestions = suggest_tactics_for_goal(goal)

        if not suggestions:
            break

        # Select the expert tactic from suggestions
        if temperature == 0:
            expert_tactic = suggestions[0]
        else:
            # Weighted: earlier suggestions are more likely
            weights = [1.0 / (i + 1) for i in range(len(suggestions))]
            total = sum(weights)
            weights = [w / total for w in weights]
            r = random.random()
            cumulative = 0.0
            expert_tactic = suggestions[0]
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    expert_tactic = suggestions[i]
                    break

        # Record the step (state BEFORE tactic)
        obs = current.to_observation()
        steps.append(RawProofStep(
            state_observation=obs,
            tactic=expert_tactic,
            theorem_name=initial_state.theorem_name,
        ))

        # Apply the tactic via TacticSimulator (reuses all simulation logic)
        success, new_state, _ = sim.apply(current, expert_tactic)
        if not success:
            break
        current = new_state

        if current.is_complete:
            break

    outcome = 1.0 if current.is_complete else 0.0

    # Back-fill outcome
    for step in steps:
        step.outcome = outcome

    return RawProofTrace(
        theorem_name=initial_state.theorem_name,
        steps=steps,
        outcome=outcome,
        source="builtin",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Conversion: RawProofTrace → TrainingExample
# ═════════════════════════════════════════════════════════════════════════════

def convert_to_training_examples(
    traces: List[RawProofTrace],
) -> List[TrainingExample]:
    """Convert raw proof traces into TrainingExample objects for seeding the replay buffer.

    Each step becomes a single example with a one-hot policy for the expert tactic.
    """
    tactic_list = _get_tactic_list()
    num_actions = len(tactic_list)

    examples: List[TrainingExample] = []

    for trace in traces:
        for step in trace.steps:
            state_vec = encode(step.state_observation)

            # Create one-hot policy for the expert's tactic
            expert_tactic = step.tactic.split()[0] if step.tactic else ""
            policy = [0.0] * num_actions
            if expert_tactic in tactic_list:
                idx = tactic_list.index(expert_tactic)
                policy[idx] = 1.0
            elif expert_tactic:
                # Unknown tactic: distribute uniformly
                policy = [1.0 / num_actions] * num_actions
            else:
                # Empty tactic: uniform
                policy = [1.0 / num_actions] * num_actions

            examples.append(TrainingExample(
                state_vec=state_vec,
                mcts_policy=policy,
                outcome=trace.outcome,
            ))

    return examples


def convert_with_mcts_policy(
    traces: List[RawProofTrace],
    num_actions: int,
    smooth_eps: float = 0.1,
) -> List[TrainingExample]:
    """Convert traces to training examples using a label-smoothed policy.

    Assigns ``(1 - eps)`` probability to the expert action, reducing overfitting
    on noisy expert demonstrations.
    """
    tactic_list = _get_tactic_list()

    examples: List[TrainingExample] = []
    uniform = smooth_eps / max(num_actions - 1, 1)

    for trace in traces:
        for step in trace.steps:
            state_vec = encode(step.state_observation)

            expert_tactic = step.tactic.split()[0] if step.tactic else ""
            policy = [uniform] * num_actions
            if expert_tactic in tactic_list:
                idx = tactic_list.index(expert_tactic)
                policy[idx] = 1.0 - smooth_eps
            else:
                # Uniform if tactic is unknown
                policy = [1.0 / num_actions] * num_actions

            examples.append(TrainingExample(
                state_vec=state_vec,
                mcts_policy=policy,
                outcome=trace.outcome,
            ))

    return examples


# ═════════════════════════════════════════════════════════════════════════════
# Persistence
# ═════════════════════════════════════════════════════════════════════════════

def save_seed_data(examples: List[TrainingExample], path: str) -> None:
    """Save seed training examples to a JSON file."""
    data = {
        "format": "axiom_zero_seed_data_v1",
        "num_examples": len(examples),
        "examples": [
            {
                "state_vec": ex.state_vec,
                "mcts_policy": ex.mcts_policy,
                "outcome": ex.outcome,
            }
            for ex in examples
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(examples)} seed examples to {path}")


def load_seed_data(path: str) -> List[TrainingExample]:
    """Load seed training examples from a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Seed data not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[TrainingExample] = []
    for ex_data in data.get("examples", []):
        examples.append(TrainingExample(
            state_vec=ex_data["state_vec"],
            mcts_policy=ex_data["mcts_policy"],
            outcome=ex_data.get("outcome", 0.0),
        ))

    print(f"Loaded {len(examples)} seed examples from {path}")
    return examples


# ═════════════════════════════════════════════════════════════════════════════
# Convenience: auto-detect and load any supported dataset
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset(
    path: str,
    format: Optional[DatasetFormat] = None,
    max_traces: Optional[int] = None,
) -> List[RawProofTrace]:
    """Auto-detect and load a dataset by path, inferring format from the filename."""
    if format is None:
        format = _infer_format(path)

    parsers = {
        DatasetFormat.LEAN_DOJO: parse_leandojo_trace,
        DatasetFormat.MINIF2F: parse_minif2f,
        DatasetFormat.PROOF_NET: parse_proofnet,
        DatasetFormat.LEAN_WORKBOOK: parse_lean_workbook,
    }

    parser = parsers.get(format)
    if parser is None:
        raise ValueError(f"No parser available for format: {format}")

    traces = parser(path)

    if max_traces is not None:
        traces = traces[:max_traces]

    return traces


def _infer_format(path: str) -> DatasetFormat:
    """Infer the dataset format from the file path."""
    name = os.path.basename(path).lower()

    if "traj" in name and (name.endswith(".json") or name.endswith(".jsonl")):
        return DatasetFormat.LEAN_DOJO
    elif "minif2f" in name or "mini_f2f" in name or "f2f" in name:
        return DatasetFormat.MINIF2F
    elif "proofnet" in name or "proof_net" in name:
        return DatasetFormat.PROOF_NET
    elif "workbook" in name or "lean_workbook" in name:
        return DatasetFormat.LEAN_WORKBOOK
    elif name.endswith(".jsonl"):
        return DatasetFormat.PROOF_NET
    else:
        # Default to miniF2F (JSON with id/informal/formal fields)
        return DatasetFormat.MINIF2F
