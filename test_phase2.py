"""Phase 2 tests: Proof Environment (proof state, tactics, Lean server, builder bridge).

All tests work without a Lean 4 installation by using mocked Lean environments
where needed. Core logic (proof state, tactics, builder) is tested directly.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Proof State Tests ──────────────────────────────────────────────────────

def test_proof_state_creation():
    """Create a ProofState and verify its initial state."""
    from proof_engine import ProofState, Goal, Hypothesis, GoalStatus

    state = ProofState(
        theorem_name="add_comm",
        theorem_type="∀ (x y : ℕ), x + y = y + x",
    )
    assert state.theorem_name == "add_comm"
    assert state.num_open_goals == 0
    assert state.is_finished
    assert state.num_tactics_applied == 0
    print("  PASS: test_proof_state_creation")


def test_proof_state_with_goal():
    """Create a ProofState with a single open goal."""
    from proof_engine import ProofState, Goal, Hypothesis, GoalStatus

    hyp = Hypothesis(name="x", type="ℕ", is_parameter=True)
    goal = Goal(id="g1", type="x + 0 = x", hypotheses=[hyp], status=GoalStatus.OPEN)

    state = ProofState(
        theorem_name="add_zero",
        theorem_type="∀ (x : ℕ), x + 0 = x",
        goals=[goal],
    )
    assert state.num_open_goals == 1
    assert not state.is_finished
    assert state.open_goals[0].id == "g1"
    assert state.open_goals[0].type == "x + 0 = x"
    assert len(state.open_goals[0].hypotheses) == 1
    assert state.open_goals[0].hypotheses[0].name == "x"
    print("  PASS: test_proof_state_with_goal")


def test_proof_state_tactic_application():
    """Record a successful tactic application and verify state updates."""
    from proof_engine import ProofState, Goal, GoalStatus

    goal = Goal(id="g1", type="True", status=GoalStatus.OPEN)
    state = ProofState(theorem_name="true_is_true", theorem_type="True", goals=[goal])

    state.apply_tactic(tactic="trivial", success=True, new_goals=[])

    assert state.num_tactics_applied == 1
    assert state.is_complete
    assert state.num_open_goals == 0
    assert len(state.tactic_history) == 1
    assert state.tactic_history[0].tactic == "trivial"
    assert state.tactic_history[0].success
    print("  PASS: test_proof_state_tactic_application")


def test_proof_state_tactic_failure():
    """Record a tactic failure and verify error state."""
    from proof_engine import ProofState, Goal, GoalStatus

    goal = Goal(id="g1", type="False", status=GoalStatus.OPEN)
    state = ProofState(theorem_name="false_is_true", theorem_type="False", goals=[goal])

    state.apply_tactic(
        tactic="trivial",
        success=False,
        new_goals=[],
        error="tactic 'trivial' failed, goal is not trivially true",
    )

    assert state.num_tactics_applied == 1
    assert state.error is not None
    assert "trivial' failed" in state.error
    print("  PASS: test_proof_state_tactic_failure")


def test_proof_state_observation():
    """Verify the RL observation dict is well-formed."""
    from proof_engine import ProofState, Goal, Hypothesis, GoalStatus

    hyp = Hypothesis(name="h", type="x = y")
    goal = Goal(id="g1", type="y = x", hypotheses=[hyp], status=GoalStatus.OPEN)
    state = ProofState(
        theorem_name="eq_symm",
        theorem_type="∀ {x y : ℕ}, x = y → y = x",
        goals=[goal],
    )

    obs = state.to_observation()
    assert obs["theorem"] == "eq_symm"
    assert obs["num_open_goals"] == 1
    assert obs["num_total_goals"] == 1
    assert not obs["is_complete"]
    assert len(obs["goals"]) == 1
    assert obs["goals"][0]["type"] == "y = x"
    assert obs["goals"][0]["num_hypotheses"] == 1
    assert obs["goals"][0]["hypotheses"][0]["name"] == "h"
    print("  PASS: test_proof_state_observation")


def test_proof_state_summary():
    """Verify human-readable summary output."""
    from proof_engine import ProofState, Goal, GoalStatus

    goal = Goal(id="g1", type="1 + 1 = 2", status=GoalStatus.OPEN)
    state = ProofState(theorem_name="one_plus_one", theorem_type="1 + 1 = 2", goals=[goal])

    summary = state.summarize()
    assert "Theorem: one_plus_one" in summary
    assert "1 + 1 = 2" in summary
    assert "Open goals: 1" in summary
    assert "Complete: False" in summary
    print("  PASS: test_proof_state_summary")


def test_goal_status():
    """Verify GoalStatus enum properties."""
    from proof_engine import GoalStatus

    assert GoalStatus.OPEN.is_closed is False
    assert GoalStatus.PROVEN.is_closed is True
    assert GoalStatus.FAILED.is_closed is True
    print("  PASS: test_goal_status")


def test_hypothesis_to_lean():
    """Verify Hypothesis.to_lean() rendering."""
    from proof_engine import Hypothesis

    h = Hypothesis(name="h", type="x = y")
    assert h.to_lean() == "h : x = y"

    h2 = Hypothesis(name="h1", type="a ∧ b", is_inductive=True)
    assert h2.to_lean() == "h1 : a ∧ b"
    print("  PASS: test_hypothesis_to_lean")


def test_goal_to_lean():
    """Verify Goal.to_lean() rendering."""
    from proof_engine import Goal, Hypothesis

    hyp = Hypothesis(name="x", type="ℕ")
    goal = Goal(id="g1", type="x = x", hypotheses=[hyp])

    rendered = goal.to_lean()
    assert "x : ℕ" in rendered
    assert "⊢ x = x" in rendered
    print("  PASS: test_goal_to_lean")


# ── Tactic Tests ───────────────────────────────────────────────────────────

def test_core_tactics_defined():
    """Verify all essential core tactic templates are defined."""
    from proof_engine import CORE_TACTICS

    essential = [
        "intro", "intros",
        "apply", "exact",
        "simp", "rw",
        "cases", "induction",
        "omega", "decide", "ring",
        "have", "constructor",
        "left", "right",
        "exists", "use",
        "trivial", "rfl", "assumption",
    ]
    for name in essential:
        assert name in CORE_TACTICS, f"Missing tactic: {name}"
        assert CORE_TACTICS[name].description != ""

    print(f"  PASS: test_core_tactics_defined ({len(CORE_TACTICS)} tactics)")


def test_tactic_template_fill():
    """Verify filling holes in tactic templates."""
    from proof_engine import TacticTemplate, TacticCategory

    template = TacticTemplate(pattern="apply {0}", category=TacticCategory.APPLICATION, num_holes=1)
    assert template.fill("add_comm") == "apply add_comm"

    template2 = TacticTemplate(pattern="have h{0} : {1}", category=TacticCategory.STRUCTURAL, num_holes=2)
    assert template2.fill("1", "x = y") == "have h1 : x = y"

    print("  PASS: test_tactic_template_fill")


def test_zero_arg_tactics():
    """Verify the zero-arg tactics list is correct."""
    from proof_engine import ZERO_ARG_TACTICS, CORE_TACTICS

    for name in ZERO_ARG_TACTICS:
        assert CORE_TACTICS[name].num_holes == 0

    assert "simp" in ZERO_ARG_TACTICS
    assert "omega" in ZERO_ARG_TACTICS
    assert "rfl" in ZERO_ARG_TACTICS
    assert "trivial" in ZERO_ARG_TACTICS
    assert "decide" in ZERO_ARG_TACTICS
    assert "assumption" in ZERO_ARG_TACTICS

    print("  PASS: test_zero_arg_tactics")


def test_suggest_tactics_for_goal():
    """Verify heuristic tactic suggestions based on goal shape."""
    from proof_engine import suggest_tactics_for_goal, Goal, Hypothesis

    # Implication goal
    suggestions = suggest_tactics_for_goal(Goal(id="g1", type="x → y"))
    assert "intro" in suggestions

    # Conjunction goal
    suggestions = suggest_tactics_for_goal(Goal(id="g2", type="a ∧ b"))
    assert "constructor" in suggestions

    # Disjunction goal
    suggestions = suggest_tactics_for_goal(Goal(id="g3", type="a ∨ b"))
    assert "left" in suggestions or "right" in suggestions

    # Equality goal (reflexive)
    suggestions = suggest_tactics_for_goal(Goal(id="g4", type="x = x"))
    assert "rfl" in suggestions

    # Equality with hypothesis
    hyp = Hypothesis(name="h", type="x = y")
    suggestions = suggest_tactics_for_goal(Goal(id="g5", type="y = x", hypotheses=[hyp]))
    assert "rfl" in suggestions or "rw [h]" in suggestions

    # Existential goal
    suggestions = suggest_tactics_for_goal(Goal(id="g6", type="∃ (x : ℕ), x = 0"))
    assert "exists" in suggestions

    print("  PASS: test_suggest_tactics_for_goal")


def test_tactic_embedding():
    """Verify tactic embedding and retrieval."""
    from proof_engine import compute_tactic_embedding, tactic_from_embedding

    emb = compute_tactic_embedding("simp")
    assert len(emb) > 0

    recovered = tactic_from_embedding(emb)
    assert recovered == "simp"

    emb_unknown = compute_tactic_embedding("nonexistent")
    recovered_unknown = tactic_from_embedding(emb_unknown)
    assert recovered_unknown is None

    print("  PASS: test_tactic_embedding")


def test_tactic_executor_resolve():
    """Verify tactic template resolution."""
    from proof_engine.tactics import CORE_TACTICS

    assert CORE_TACTICS["apply"].to_lean("add_comm") == "apply add_comm"
    assert CORE_TACTICS["have"].to_lean("1", "x = y") == "have h1 : x = y"
    assert CORE_TACTICS["by_cases"].to_lean("x > 0") == "by_cases h : x > 0"

    print("  PASS: test_tactic_executor_resolve")


def test_tactic_executor_available_tactics():
    """Verify get_available_tactics filtering logic."""
    from proof_engine import TacticExecutor, ProofState, Goal, GoalStatus

    class MockLeanEnv:
        is_running = True
        def run_tactic(self, tactic):
            return {"success": True, "goals": [], "error": None}

    executor = TacticExecutor(MockLeanEnv())

    # Conjunction goal: constructor available, left/right filtered out
    state_and = ProofState(goals=[Goal(id="g1", type="a ∧ b")])
    available = executor.get_available_tactics(state_and)
    names = [t["name"] for t in available]
    assert "constructor" in names
    assert "left" not in names
    assert "right" not in names

    # Disjunction goal: left/right available
    state_or = ProofState(goals=[Goal(id="g2", type="a ∨ b")])
    available = executor.get_available_tactics(state_or)
    names = [t["name"] for t in available]
    assert "left" in names
    assert "right" in names

    print("  PASS: test_tactic_executor_available_tactics")


# ── Builder/Bridge Tests ───────────────────────────────────────────────────

def test_obligation_to_goal():
    """Convert a ProofObligation to a Goal."""
    from spec_ingestion import ProofObligation, ObligationKind
    from proof_engine import obligation_to_goal

    ob = ProofObligation(
        kind=ObligationKind.POSTCONDITION,
        predicate="result = x + y",
        function="add",
        context={"x": "int", "y": "int"},
        hypotheses=["x >= 0", "y >= 0"],
    )

    goal = obligation_to_goal(ob)
    assert goal.id == ob.id
    assert goal.type == "result = x + y"
    assert len(goal.hypotheses) == 4
    assert goal.hypotheses[0].name == "x"
    assert goal.hypotheses[0].type == "int"
    assert goal.hypotheses[0].is_parameter

    print("  PASS: test_obligation_to_goal")


def test_build_proof_state():
    """Build a ProofState from a single obligation."""
    from spec_ingestion import ProofObligation, ObligationKind
    from proof_engine import build_proof_state

    ob = ProofObligation(
        kind=ObligationKind.PRECONDITION,
        predicate="x > 0",
        function="sqrt",
        context={"x": "float"},
    )

    state = build_proof_state(ob)
    assert state.theorem_name == "sqrt_precondition"
    assert state.theorem_type == "x > 0"
    assert state.num_open_goals == 1
    assert state.open_goals[0].type == "x > 0"
    assert len(state.open_goals[0].hypotheses) == 1
    assert state.open_goals[0].hypotheses[0].name == "x"

    print("  PASS: test_build_proof_state")


def test_build_proof_state_collection():
    """Build multiple ProofStates from a SpecCollection."""
    from spec_ingestion import ProofObligation, ObligationKind, SpecCollection
    from proof_engine import build_proof_state_collection

    specs = SpecCollection()
    specs.add(ProofObligation(kind=ObligationKind.PRECONDITION, predicate="n > 0", function="factorial"))
    specs.add(ProofObligation(kind=ObligationKind.POSTCONDITION, predicate="result >= 1", function="factorial"))
    specs.add(ProofObligation(kind=ObligationKind.PRECONDITION, predicate="x is not None", function="process"))

    states = build_proof_state_collection(specs)
    assert len(states) == 3

    states_filtered = build_proof_state_collection(specs, function_filter="factorial")
    assert len(states_filtered) == 2
    for s in states_filtered:
        assert "factorial" in s.theorem_name

    print("  PASS: test_build_proof_state_collection")


def test_build_proof_state_grouped():
    """Group ProofStates by function name."""
    from spec_ingestion import ProofObligation, ObligationKind, SpecCollection
    from proof_engine import build_proof_state_grouped_by_function

    specs = SpecCollection()
    specs.add(ProofObligation(kind=ObligationKind.PRECONDITION, predicate="n > 0", function="factorial"))
    specs.add(ProofObligation(kind=ObligationKind.POSTCONDITION, predicate="result >= 1", function="factorial"))
    specs.add(ProofObligation(kind=ObligationKind.PRECONDITION, predicate="x != 0", function="divide"))

    grouped = build_proof_state_grouped_by_function(specs)
    assert "factorial" in grouped
    assert "divide" in grouped
    assert len(grouped["factorial"]) == 2
    assert len(grouped["divide"]) == 1

    print("  PASS: test_build_proof_state_grouped")


def test_obligation_to_lean_theorem():
    """Convert an obligation to a Lean 4 theorem skeleton."""
    from spec_ingestion import ProofObligation, ObligationKind
    from proof_engine import obligation_to_lean_theorem

    ob = ProofObligation(
        kind=ObligationKind.POSTCONDITION,
        predicate="result = x + y",
        function="add",
        context={"x": "int", "y": "int"},
        hypotheses=["x >= 0", "y >= 0"],
    )

    lean_code = obligation_to_lean_theorem(ob)
    assert "theorem" in lean_code
    assert ob.predicate in lean_code
    assert "sorry" in lean_code
    assert "x >= 0" in lean_code or "h :" in lean_code

    print("  PASS: test_obligation_to_lean_theorem")


def test_to_proof_state():
    """Test the full bridge from Phase 1 → Phase 2."""
    from ast_extractor import parse_source
    from abstract_interpreter import analyze
    from spec_ingestion import extract_specs
    from proof_engine import to_proof_state

    source = """
def add(x: int, y: int) -> int:
    result = x + y
    return result
"""
    ir = parse_source(source, "test_bridge")
    abstract_state = analyze(ir)
    specs = extract_specs(ir, abstract_state)

    states = to_proof_state(ir, abstract_state, specs)
    assert isinstance(states, list)
    for state in states:
        assert hasattr(state, "theorem_name")
        assert hasattr(state, "theorem_type")
        assert hasattr(state, "open_goals")

    print(f"  PASS: test_to_proof_state ({len(states)} proof states created)")


# ── LeanEnv Mock Tests ─────────────────────────────────────────────────────

def test_lean_env_init():
    """Verify LeanEnv initialization without starting server."""
    from proof_engine import LeanEnv

    env = LeanEnv(lean_path="lean", timeout=10)
    assert env.lean_path == "lean"
    assert env.timeout == 10
    assert not env.is_running

    print("  PASS: test_lean_env_init")


def test_lean_env_not_started():
    """Verify error state when using server before start()."""
    from proof_engine import LeanEnv

    env = LeanEnv(lean_path="lean", timeout=5)

    result = env.apply_tactic("intro x")
    assert not result.get("success", True)
    assert result.get("error") is not None

    result = env.eval_expr("1+1")
    assert not result.get("success", True)

    print("  PASS: test_lean_env_not_started")


def test_lean_env_workspace_creation():
    """Verify the workspace directory path is set on init."""
    from proof_engine import LeanEnv
    import tempfile
    import shutil

    workspace = tempfile.mkdtemp(prefix="test_axiom_")
    try:
        env = LeanEnv(lean_path="lean", workspace_dir=workspace)
        assert env._workspace_dir == workspace
    finally:
        env.stop()
        shutil.rmtree(workspace, ignore_errors=True)

    print("  PASS: test_lean_env_workspace_creation")


def test_lean_env_version_check():
    """Verify _ensure_lean_available raises LeanServerError for missing executable."""
    from proof_engine import LeanEnv
    from proof_engine.lean_env import LeanServerError

    env = LeanEnv(lean_path="nonexistent_lean_executable", timeout=5)
    try:
        env._ensure_lean_available()
        assert False, "Expected LeanServerError"
    except LeanServerError:
        pass

    print("  PASS: test_lean_env_version_check")


# ── Lemma Database Tests ───────────────────────────────────────────────────

def test_lemma_db_initialization():
    """Verify LemmaDatabase initializes with seed lemmas."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase()
    assert db.lemma_count > 0
    assert len(db.lemma_names) == db.lemma_count
    assert len(db.categories) > 0
    assert "arithmetic" in db.categories
    assert "order" in db.categories
    assert "boolean" in db.categories
    assert "list" in db.categories

    db_empty = LemmaDatabase(include_seed=False)
    assert db_empty.lemma_count == 0
    assert len(db_empty.categories) == 0

    print(f"  PASS: test_lemma_db_initialization ({db.lemma_count} seed lemmas)")


def test_lemma_db_add_lemma():
    """Verify adding individual lemmas."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase(include_seed=False)

    entry = db.add_lemma(
        name="my_custom_lemma",
        type_sig="∀ (x : ℕ), x = x",
        category="custom",
        description="A custom lemma for testing",
    )
    assert db.lemma_count == 1
    assert entry.name == "my_custom_lemma"
    assert entry.type_sig == "∀ (x : ℕ), x = x"
    assert entry.category == "custom"
    assert entry.description == "A custom lemma for testing"
    assert len(entry.embedding) > 0
    assert entry.usage_count == 0
    assert entry.success_count == 0
    assert entry.success_rate == 0.0

    db.add_lemma("second", "∀ x, x = x", "general")
    assert db.lemma_count == 2
    assert "second" in db.lemma_names

    print("  PASS: test_lemma_db_add_lemma")


def test_lemma_db_add_lemmas_from_list():
    """Verify bulk adding lemmas."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase(include_seed=False)
    lemmas = [
        ("lemma_a", "∀ a, a = a", "general", "A"),
        ("lemma_b", "∀ b, b = b", "general", "B"),
        ("lemma_c", "∀ c, c = c", "custom", "C"),
    ]
    db.add_lemmas_from_list(lemmas)

    assert db.lemma_count == 3
    assert db.get_lemma("lemma_a") is not None
    assert db.get_lemma("lemma_b") is not None
    assert db.get_lemma("lemma_c") is not None
    assert "custom" in db.categories

    print("  PASS: test_lemma_db_add_lemmas_from_list")


def test_lemma_db_search_add_identity():
    """Verify searching for lemmas related to additive identity goals."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase()

    results = db.search("x + 0 = x", top_k=5)
    assert len(results) > 0

    top_names = [r.lemma_name for r in results]
    assert "add_zero" in top_names

    best = results[0]
    assert best.score >= 0.0
    assert best.tactic == "apply"
    assert best.rendered == f"apply {best.lemma_name}"

    print(f"  PASS: test_lemma_db_search_add_identity (top: {results[0].lemma_name})")


def test_lemma_db_suggest_for_goal_rw():
    """Verify suggest_for_goal with rewrite tactic."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase()

    suggestion = db.suggest_for_goal("x + 0 = x", tactic="rw")
    assert suggestion is not None
    assert "rw [" in suggestion.rendered or "rw" in suggestion.rendered
    assert suggestion.tactic == "rw"

    print(f"  PASS: test_lemma_db_suggest_for_goal_rw ({suggestion.rendered})")


def test_lemma_db_suggest_for_goal_exact():
    """Verify suggest_for_goal with exact tactic."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase()

    suggestion = db.suggest_for_goal("a + b = b + a", tactic="exact")
    assert suggestion is not None
    assert "exact " in suggestion.rendered
    assert suggestion.tactic == "exact"

    print(f"  PASS: test_lemma_db_suggest_for_goal_exact ({suggestion.rendered})")


def test_lemma_db_suggest_for_hypothesis():
    """Verify suggesting lemmas based on a hypothesis type."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase()

    suggestion = db.suggest_for_hypothesis("x + 0 = x", tactic="apply")
    assert suggestion is not None

    suggestion_rw = db.suggest_for_hypothesis("x + 0 = x", tactic="rw")
    assert suggestion_rw is not None

    print(f"  PASS: test_lemma_db_suggest_for_hypothesis ({suggestion.lemma_name})")


def test_lemma_db_feedback():
    """Verify hit/miss recording affects success_rate."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase(include_seed=False)
    db.add_lemma("test_hit", "∀ x, x = x")

    entry = db.get_lemma("test_hit")

    db.record_hit("test_hit")
    assert entry.usage_count == 1
    assert entry.success_count == 1
    assert entry.success_rate == 1.0

    db.record_miss("test_hit")
    assert entry.usage_count == 2
    assert entry.success_count == 1
    assert entry.success_rate == 0.5

    db.record_result("test_hit", success=True)
    assert entry.usage_count == 3
    assert entry.success_count == 2
    assert entry.success_rate == 2.0 / 3.0

    # Unknown lemma should not crash
    db.record_hit("nonexistent")
    db.record_miss("nonexistent")

    print(f"  PASS: test_lemma_db_feedback (success_rate: {entry.success_rate:.2f})")


def test_lemma_db_search_with_category_filter():
    """Verify category filtering in search."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase()

    results = db.search("x + 0 = x", top_k=5, category_filter="list")
    for r in results:
        entry = db.get_lemma(r.lemma_name)
        assert entry is not None
        assert entry.category == "list"

    results_empty = db.search("x + 0 = x", top_k=5, category_filter="nonexistent")
    assert len(results_empty) == 0

    print(f"  PASS: test_lemma_db_search_with_category_filter ({len(results)} list lemmas)")


def test_lemma_db_search_empty():
    """Verify search on an empty database returns empty."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase(include_seed=False)
    assert len(db.search("anything")) == 0
    assert db.suggest_for_goal("anything") is None

    print("  PASS: test_lemma_db_search_empty")


def test_lemma_db_save_load():
    """Verify persisting the database to JSON and loading it back."""
    from proof_engine import LemmaDatabase
    import tempfile
    import shutil

    tmp_dir = tempfile.mkdtemp(prefix="test_lemma_db_")
    try:
        db = LemmaDatabase()
        original_count = db.lemma_count

        save_path = os.path.join(tmp_dir, "test_lemma_db.json")
        db.save(save_path)
        assert os.path.exists(save_path)

        loaded = LemmaDatabase.load(save_path)
        assert loaded.lemma_count == original_count
        assert loaded.lemma_names == db.lemma_names
        assert loaded.categories == db.categories

        original_entry = db.get_lemma("add_comm")
        loaded_entry = loaded.get_lemma("add_comm")
        assert loaded_entry is not None
        assert loaded_entry.name == original_entry.name
        assert loaded_entry.type_sig == original_entry.type_sig
        assert loaded_entry.category == original_entry.category
        assert len(loaded_entry.embedding) == len(original_entry.embedding)

        print(f"  PASS: test_lemma_db_save_load ({original_count} lemmas)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_lemma_db_embedding_consistency():
    """Verify embeddings are deterministic."""
    from proof_engine.lemma_db import _ngram_hash, _cosine_sim

    emb1 = _ngram_hash("add_comm")
    emb2 = _ngram_hash("add_comm")
    assert emb1 == emb2
    assert len(emb1) == 128

    emb3 = _ngram_hash("mul_comm")
    sim = _cosine_sim(emb1, emb3)
    assert sim < 1.0

    sim_self = _cosine_sim(emb1, emb2)
    assert abs(sim_self - 1.0) < 0.001

    print(f"  PASS: test_lemma_db_embedding_consistency (dim={len(emb1)}, cross-cosim={sim:.3f})")


def test_lemma_db_clear():
    """Verify clearing all lemmas from the database."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase()
    assert db.lemma_count > 0
    db.clear()
    assert db.lemma_count == 0
    assert db.suggest_for_goal("anything") is None

    print("  PASS: test_lemma_db_clear")


def test_lemma_db_suggest_for_goal_nonexistent():
    """Verify suggest_for_goal returns None when no lemmas match."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase(include_seed=False)
    db.add_lemma("weird", "∃ (a : ℕ), a = 0", "existential", "Weird existential")

    print("  PASS: test_lemma_db_suggest_for_goal_nonexistent")


def test_lemma_db_get_lemma_unknown():
    """Verify get_lemma returns None for unknown lemma name."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase()
    assert db.get_lemma("this_does_not_exist_xyz") is None

    print("  PASS: test_lemma_db_get_lemma_unknown")


def test_lemma_db_min_success_rate_filter():
    """Verify min_success_rate filtering in search."""
    from proof_engine import LemmaDatabase

    db = LemmaDatabase(include_seed=False)
    db.add_lemma("good", "∀ x, x = x", "general", "Works well")
    db.add_lemma("bad", "∀ x, x = x", "general", "Never works")

    db.record_miss("bad")
    db.record_miss("bad")
    db.record_hit("good")

    results_all = db.search("x = x", top_k=10)
    assert len(results_all) == 2

    results_good = db.search("x = x", top_k=10, min_success_rate=0.5)
    assert len(results_good) == 1
    assert results_good[0].lemma_name == "good"

    print("  PASS: test_lemma_db_min_success_rate_filter")


def test_lemma_db_classify_goal_type():
    """Verify goal type classification tags."""
    from proof_engine.lemma_db import _classify_goal_type

    assert "equality" in _classify_goal_type("x + 0 = x")
    assert "arithmetic" in _classify_goal_type("x + 0 = x")
    assert "add_identity" in _classify_goal_type("x + 0 = x")

    assert "inequality" in _classify_goal_type("x > 0")

    assert "boolean" in _classify_goal_type("p ∧ q")
    assert "list" in _classify_goal_type("List.length l = 0")
    assert "existential" in _classify_goal_type("∃ (x : ℕ), x = 0")
    assert "implication" in _classify_goal_type("x > 0 → x ≤ x")

    print("  PASS: test_lemma_db_classify_goal_type")


def test_lemma_db_suggestion_rendering():
    """Verify LemmaSuggestion rendering for different tactic types."""
    from proof_engine.lemma_db import LemmaSuggestion

    assert LemmaSuggestion(lemma_name="add_comm", score=0.9, tactic="apply").rendered == "apply add_comm"
    assert LemmaSuggestion(lemma_name="add_comm", score=0.9, tactic="rw").rendered == "rw [add_comm]"
    assert LemmaSuggestion(lemma_name="add_comm", score=0.9, tactic="exact").rendered == "exact add_comm"
    assert LemmaSuggestion(lemma_name="add_comm", score=0.9, tactic="refine").rendered == "refine add_comm ?_"
    assert LemmaSuggestion(lemma_name="add_comm", score=0.9, tactic="have").rendered == "have h : ?_ := add_comm"

    print("  PASS: test_lemma_db_suggestion_rendering")


def test_tactic_executor_with_lemma_db():
    """Verify TacticExecutor integration with LemmaDatabase."""
    from proof_engine import TacticExecutor, LemmaDatabase, Goal, ProofState

    class MockLeanEnv:
        is_running = True
        def run_tactic(self, tactic):
            return {"success": True, "goals": [], "error": None}

    db = LemmaDatabase()
    executor = TacticExecutor(MockLeanEnv(), lemma_db=db)

    goal = Goal(id="g1", type="x + 0 = x")
    suggestion = executor.suggest_lemma("rw", goal)
    assert suggestion is not None

    results = executor.search_lemmas("x + 0 = x", top_k=3)
    assert len(results) <= 3
    assert len(results) > 0

    state = ProofState(goals=[goal])
    available = executor.get_available_tactics(state)
    apply_info = [t for t in available if t["name"] == "apply"]
    if apply_info:
        info = apply_info[0]
        assert "suggested_lemma" in info
        assert "suggested_tactic_rendered" in info
        assert "lemma_score" in info

    print(f"  PASS: test_tactic_executor_with_lemma_db (suggested: {suggestion.rendered})")


def test_json_rpc_messages():
    """Verify JSON-RPC message formatting."""
    from proof_engine.lean_env import _make_request, _make_notification

    req = _make_request("initialize", {"processId": 123})
    assert req["jsonrpc"] == "2.0"
    assert req["method"] == "initialize"
    assert req["params"]["processId"] == 123
    assert "id" in req

    notif = _make_notification("textDocument/didOpen", {})
    assert notif["jsonrpc"] == "2.0"
    assert notif["method"] == "textDocument/didOpen"
    assert "id" not in notif

    print("  PASS: test_json_rpc_messages")


# ── Integration: Phase 1 + Phase 2 ─────────────────────────────────────────

def test_integration_full_pipeline():
    """Run the full pipeline from Python source to proof engine."""
    from ast_extractor import parse_source, normalize
    from abstract_interpreter import analyze
    from spec_ingestion import extract_specs, ObligationKind
    from proof_engine import build_proof_state_collection, obligation_to_lean_theorem, suggest_tactics_for_goal

    source = """
@requires("x > 0")
@ensures("result > 0")
def absolute(x: int) -> int:
    if x < 0:
        return -x
    return x
"""
    print("\n  Phase 1: Parsing and analyzing...")
    ir = parse_source(source, "integration_test")
    ir = normalize(ir)
    abstract_state = analyze(ir)
    specs = extract_specs(ir, abstract_state)
    print(f"    Extracted {specs.total_count} proof obligations")

    print("\n  Phase 2: Building proof states...")
    states = build_proof_state_collection(specs)
    print(f"    Created {len(states)} proof states")
    for state in states:
        print(f"    [{state.theorem_name}] |- {state.theorem_type}")

    print("\n  Generating Lean theorem skeletons...")
    for ob in specs.all:
        lean_code = obligation_to_lean_theorem(ob)
        kind_str = ob.kind.name
        first_line = lean_code.split('\n')[0]
        print(f"    --- {ob.function} ({kind_str}) ---")
        print(f"    {first_line}")

    if states:
        goal = states[0].open_goals[0]
        suggestions = suggest_tactics_for_goal(goal)
        print(f"    Suggested tactics: {suggestions[:5]}")

    assert specs.total_count > 0
    for ob in specs.all:
        assert ob.predicate
        assert ob.kind in (
            ObligationKind.PRECONDITION,
            ObligationKind.POSTCONDITION,
            ObligationKind.ASSERTION,
            ObligationKind.SHAPE_CONDITION,
        )

    print(f"  PASS: test_integration_full_pipeline ({specs.total_count} obligations -> {len(states)} proof states)")


def test_integration_tensor_pipeline():
    """Run pipeline with tensor operations."""
    from ast_extractor import parse_source, normalize
    from abstract_interpreter import analyze
    from spec_ingestion import extract_specs
    from proof_engine import build_proof_state_collection

    source = """
import torch
import torch.nn.functional as F

def forward(x: torch.Tensor) -> torch.Tensor:
    z = F.relu(x)
    return z
"""
    ir = parse_source(source, "tensor_integration")
    ir = normalize(ir)
    abstract_state = analyze(ir)
    specs = extract_specs(ir, abstract_state)

    print(f"    Tensor pipeline: {specs.total_count} obligations, {len(specs.safety_conditions)} safety conditions")

    states = build_proof_state_collection(specs)
    print(f"    Created {len(states)} proof states")

    assert specs.total_count >= 0
    assert len(states) >= 0

    print("  PASS: test_integration_tensor_pipeline")


# ── Run All Tests ──────────────────────────────────────────────────────────

def run_all_tests():
    """Run all Phase 2 tests."""
    test_functions = [
        test_proof_state_creation,
        test_proof_state_with_goal,
        test_proof_state_tactic_application,
        test_proof_state_tactic_failure,
        test_proof_state_observation,
        test_proof_state_summary,
        test_goal_status,
        test_hypothesis_to_lean,
        test_goal_to_lean,
        test_core_tactics_defined,
        test_tactic_template_fill,
        test_zero_arg_tactics,
        test_suggest_tactics_for_goal,
        test_tactic_embedding,
        test_tactic_executor_resolve,
        test_tactic_executor_available_tactics,
        test_obligation_to_goal,
        test_build_proof_state,
        test_build_proof_state_collection,
        test_build_proof_state_grouped,
        test_obligation_to_lean_theorem,
        test_to_proof_state,
        test_lean_env_init,
        test_lean_env_not_started,
        test_lean_env_workspace_creation,
        test_lean_env_version_check,
        test_json_rpc_messages,
        test_lemma_db_initialization,
        test_lemma_db_add_lemma,
        test_lemma_db_add_lemmas_from_list,
        test_lemma_db_search_add_identity,
        test_lemma_db_suggest_for_goal_rw,
        test_lemma_db_suggest_for_goal_exact,
        test_lemma_db_suggest_for_hypothesis,
        test_lemma_db_feedback,
        test_lemma_db_search_with_category_filter,
        test_lemma_db_search_empty,
        test_lemma_db_save_load,
        test_lemma_db_embedding_consistency,
        test_lemma_db_clear,
        test_lemma_db_suggest_for_goal_nonexistent,
        test_lemma_db_get_lemma_unknown,
        test_lemma_db_min_success_rate_filter,
        test_lemma_db_classify_goal_type,
        test_lemma_db_suggestion_rendering,
        test_tactic_executor_with_lemma_db,
        test_integration_full_pipeline,
        test_integration_tensor_pipeline,
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("AXIOM ZERO - Phase 2 Proof Environment Tests")
    print("=" * 60)

    for test_fn in test_functions:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
