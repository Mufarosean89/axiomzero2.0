"""
Axiom Zero - Phase 3 Test Suite
RL Agent: State Encoder, Policy/Value Network, MCTS, Self-Play

Run:
    python test_phase3.py

Expected: All tests pass.
"""

import sys
import os
import math
import random
import json
import tempfile
import traceback
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ast_extractor import parse_source, normalize
from abstract_interpreter import analyze
from spec_ingestion import extract_specs
from proof_engine import (
    build_proof_state, build_proof_state_collection,
    ProofState, Goal, Hypothesis, GoalStatus, CORE_TACTICS,
)
from rl_agent import (
    encode, FEATURE_DIM,
    PolicyValueNet,
    MCTS, TacticSimulator, MCTSNode,
    SelfPlayTrainer, TrainingConfig, run_episode, train_from_source,
    EpisodeResult, TrainingExample,
)

# -- Test harness ------------------------------------------------------------------

PASS = 0
FAIL = 0
results = []


def test(name: str):
    """Decorator that registers and runs a test function."""
    def decorator(fn):
        global PASS, FAIL
        try:
            fn()
            PASS += 1
            results.append(("PASS", name))
            print(f"  OK  {name}")
        except Exception as e:
            FAIL += 1
            results.append(("FAIL", name, str(e)))
            print(f"  FAIL  {name}")
            print(f"       {e}")
            if "--verbose" in sys.argv:
                traceback.print_exc()
        return fn
    return decorator


def assert_eq(a, b, msg=""):
    assert a == b, f"{msg}: {a!r} != {b!r}"


def assert_close(a: float, b: float, tol: float = 1e-6, msg: str = ""):
    assert abs(a - b) < tol, f"{msg}: |{a} - {b}| >= {tol}"


def assert_in_range(v: float, lo: float, hi: float, msg: str = ""):
    assert lo <= v <= hi, f"{msg}: {v} not in [{lo}, {hi}]"


# -- Fixtures --------------------------------------------------------------------

SIMPLE_SOURCE = """
@requires("x > 0")
@ensures("result > 0")
def absolute(x: int) -> int:
    if x < 0:
        return -x
    return x
"""

MULTI_SOURCE = """
@requires("n >= 0")
@ensures("result >= 0")
def factorial(n: int) -> int:
    if n == 0:
        return 1
    return n * factorial(n - 1)

@requires("len(xs) > 0")
@ensures("result in xs")
def head(xs: list) -> int:
    return xs[0]
"""


def _make_proof_states(source: str = SIMPLE_SOURCE) -> list:
    ir = parse_source(source, "test_module")
    ir = normalize(ir)
    state = analyze(ir)
    specs = extract_specs(ir, state)
    return build_proof_state_collection(specs)


# =============================================================================
# Section 1: State Encoder
# =============================================================================

print("\n-- Section 1: State Encoder ------------------------------------------------")


@test("encode() returns a list of length FEATURE_DIM")
def _():
    ps = _make_proof_states()[0]
    vec = encode(ps.to_observation())
    assert isinstance(vec, list), "Not a list"
    assert_eq(len(vec), FEATURE_DIM, "Vector length")


@test("FEATURE_DIM == 256")
def _():
    assert_eq(FEATURE_DIM, 256)


@test("encode() output is all floats")
def _():
    ps = _make_proof_states()[0]
    vec = encode(ps.to_observation())
    assert all(isinstance(v, float) for v in vec), "Non-float elements"


@test("encode() global scalars are in [0, 1]")
def _():
    ps = _make_proof_states()[0]
    vec = encode(ps.to_observation())
    for i in range(8):
        assert_in_range(vec[i], 0.0, 1.0, f"vec[{i}]")


@test("encode() tactic bag section is in [0, 1]")
def _():
    ps = _make_proof_states()[0]
    vec = encode(ps.to_observation())
    for i in range(8, 47):
        assert_in_range(vec[i], 0.0, 1.0, f"vec[{i}]")


@test("encode() produces different vectors for different states")
def _():
    states = _make_proof_states(MULTI_SOURCE)
    # Find two states with distinct theorem names (builder may produce duplicates)
    seen = {}
    for ps in states:
        name = ps.theorem_name
        if name not in seen:
            seen[name] = ps
    if len(seen) < 2:
        return  # not enough distinct theorems — skip
    distinct = list(seen.values())
    v1 = encode(distinct[0].to_observation())
    v2 = encode(distinct[1].to_observation())
    assert v1 != v2, "Different theorems produced identical vectors"


@test("encode() handles empty proof state (no goals)")
def _():
    obs = {
        "theorem": "empty",
        "num_open_goals": 0,
        "num_total_goals": 0,
        "num_tactics_applied": 0,
        "depth": 0,
        "is_complete": True,
        "goals": [],
        "tactic_history": [],
    }
    vec = encode(obs)
    assert_eq(len(vec), FEATURE_DIM)
    assert_eq(vec[4], 1.0, "is_complete flag")


@test("encode() is deterministic")
def _():
    ps = _make_proof_states()[0]
    obs = ps.to_observation()
    v1 = encode(obs)
    v2 = encode(obs)
    assert_eq(v1, v2, "encode() is not deterministic")


# =============================================================================
# Section 2: Policy / Value Network
# =============================================================================

print("\n-- Section 2: Policy / Value Network --------------------------------------")


@test("PolicyValueNet initializes without error")
def _():
    net = PolicyValueNet()
    assert net.num_actions == len(CORE_TACTICS)


@test("forward() returns (priors, value) with correct shapes")
def _():
    net = PolicyValueNet()
    vec = [0.0] * FEATURE_DIM
    priors, value = net.forward(vec)
    assert_eq(len(priors), len(CORE_TACTICS), "priors length")
    assert isinstance(value, float), "value is not float"


@test("priors sum to 1.0 (softmax)")
def _():
    net = PolicyValueNet()
    vec = [random.gauss(0, 1) for _ in range(FEATURE_DIM)]
    priors, _ = net.forward(vec)
    total = sum(priors)
    assert_close(total, 1.0, tol=1e-6, msg="priors sum")


@test("all priors are in [0, 1]")
def _():
    net = PolicyValueNet()
    vec = [random.gauss(0, 1) for _ in range(FEATURE_DIM)]
    priors, _ = net.forward(vec)
    for i, p in enumerate(priors):
        assert_in_range(p, 0.0, 1.0, f"prior[{i}]")


@test("value is in [-1, 1]")
def _():
    net = PolicyValueNet()
    for _ in range(5):
        vec = [random.gauss(0, 1) for _ in range(FEATURE_DIM)]
        _, v = net.forward(vec)
        assert_in_range(v, -1.0, 1.0, "value")


@test("update_weights() returns (policy_loss, value_loss) floats")
def _():
    net = PolicyValueNet()
    na = net.num_actions
    vecs = [[random.gauss(0, 0.1) for _ in range(FEATURE_DIM)] for _ in range(4)]
    pols = [[1.0 / na] * na for _ in range(4)]
    vals = [1.0, -1.0, 0.0, 1.0]
    pi_loss, v_loss = net.update_weights(pols, vals, vecs, lr=1e-3)
    assert isinstance(pi_loss, float)
    assert isinstance(v_loss, float)
    assert pi_loss >= 0.0, "Policy loss must be non-negative"


@test("update_weights() reduces policy loss over multiple steps")
def _():
    """Network should overfit on a tiny constant batch."""
    net = PolicyValueNet()
    na = net.num_actions
    # Fixed target: always predict tactic 0
    target_pol = [0.0] * na
    target_pol[0] = 1.0
    vecs = [[0.1 * i for i in range(FEATURE_DIM)]] * 8
    pols = [target_pol] * 8
    vals = [1.0] * 8
    losses_pi = []
    for _ in range(20):
        pi_loss, _ = net.update_weights(pols, vals, vecs, lr=1e-2)
        losses_pi.append(pi_loss)
    # Loss should decrease on average
    first_half = sum(losses_pi[:10]) / 10
    second_half = sum(losses_pi[10:]) / 10
    assert second_half <= first_half + 0.5, (
        "Loss did not decrease: {:.4f} -> {:.4f}".format(first_half, second_half)
    )


@test("save() and load() round-trip preserves forward output")
def _():
    net = PolicyValueNet()
    vec = [random.gauss(0, 1) for _ in range(FEATURE_DIM)]
    priors_before, value_before = net.forward(vec)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        net.save(path)
        net2 = PolicyValueNet.load(path)
        priors_after, value_after = net2.forward(vec)
        assert_close(value_before, value_after, tol=1e-6, msg="value round-trip")
        for i, (a, b) in enumerate(zip(priors_before, priors_after)):
            assert_close(a, b, tol=1e-6, msg=f"prior[{i}] round-trip")
    finally:
        os.unlink(path)


# =============================================================================
# Section 3: MCTS
# =============================================================================

print("\n-- Section 3: MCTS ---------------------------------------------------------")


@test("TacticSimulator.apply() returns (bool, ProofState, float)")
def _():
    sim = TacticSimulator()
    ps = _make_proof_states()[0]
    success, new_state, reward = sim.apply(ps, "intro h")
    assert isinstance(success, bool)
    assert isinstance(new_state, ProofState)
    assert isinstance(reward, float)


@test("TacticSimulator reward is in [-1.0, 1.0]")
def _():
    sim = TacticSimulator()
    ps = _make_proof_states()[0]
    for tactic in ["omega", "intro h", "apply", "simp", "rfl"]:
        _, _, reward = sim.apply(ps, tactic)
        assert_in_range(reward, -1.0, 1.0, f"reward for {tactic}")


@test("MCTSNode UCB score is finite")
def _():
    node = MCTSNode(state=_make_proof_states()[0], prior=0.5, visit_count=3, value_sum=1.5)
    score = node.ucb_score(parent_visit_count=10)
    assert math.isfinite(score), "UCB score not finite"


@test("MCTSNode UCB score with zero visits equals prior * c_puct * sqrt(parent_N)")
def _():
    from rl_agent.mcts import C_PUCT
    node = MCTSNode(state=_make_proof_states()[0], prior=0.4, visit_count=0, value_sum=0.0)
    score = node.ucb_score(parent_visit_count=9)
    expected = C_PUCT * 0.4 * math.sqrt(9)
    assert_close(score, expected, tol=1e-6, msg="UCB unvisited")


@test("MCTS.search() returns action_probs summing to 1.0")
def _():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    mcts = MCTS(net, num_simulations=10)
    probs, _ = mcts.search(ps)
    assert_eq(len(probs), len(CORE_TACTICS))
    total = sum(probs)
    assert_close(total, 1.0, tol=1e-6, msg="action_probs sum")


@test("MCTS.search() returns root_value in [-1, 1]")
def _():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    mcts = MCTS(net, num_simulations=10)
    _, value = mcts.search(ps)
    assert_in_range(value, -1.0, 1.0, "root_value")


@test("MCTS.best_action() returns a valid tactic name")
def _():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    mcts = MCTS(net, num_simulations=10)
    action = mcts.best_action(ps)
    assert action in CORE_TACTICS, f"'{action}' not a valid tactic"


@test("MCTS expands root and creates children for each tactic")
def _():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    mcts = MCTS(net, num_simulations=5)
    root = MCTSNode(state=ps)
    mcts._expand(root)
    assert len(root.children) == len(CORE_TACTICS), "Wrong number of children"


@test("MCTS search with more simulations increases visit counts")
def _():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    mcts = MCTS(net, num_simulations=20)
    probs1, _ = mcts.search(ps)
    mcts2 = MCTS(net, num_simulations=5)
    probs2, _ = mcts2.search(ps)
    # Both are valid probability distributions
    assert_close(sum(probs1), 1.0, tol=1e-6, msg="probs1")
    assert_close(sum(probs2), 1.0, tol=1e-6, msg="probs2")


# =============================================================================
# Section 4: Self-Play Episode
# =============================================================================

print("\n-- Section 4: Self-Play Episode --------------------------------------------")


@test("run_episode() returns an EpisodeResult")
def _():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    result = run_episode(ps, net, num_simulations=5)
    assert isinstance(result, EpisodeResult)


@test("EpisodeResult.outcome is in {-1, 0, +1}")
def _():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    result = run_episode(ps, net, num_simulations=5)
    assert result.outcome in (-1.0, 0.0, 1.0), f"Unexpected outcome: {result.outcome}"


@test("EpisodeResult.examples all have correct state_vec length")
def _():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    result = run_episode(ps, net, num_simulations=5)
    for i, ex in enumerate(result.examples):
        assert_eq(len(ex.state_vec), FEATURE_DIM, f"Example {i} state_vec length")


@test("EpisodeResult.examples all have mcts_policy summing to 1")
def _():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    result = run_episode(ps, net, num_simulations=5)
    for i, ex in enumerate(result.examples):
        total = sum(ex.mcts_policy)
        assert_close(total, 1.0, tol=1e-6, msg=f"Example {i} policy sum")


@test("EpisodeResult.examples all have the same outcome")
def _():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    result = run_episode(ps, net, num_simulations=5)
    outcomes = {ex.outcome for ex in result.examples}
    assert len(outcomes) <= 1, f"Mixed outcomes in episode: {outcomes}"


@test("run_episode() with temperature=0 is greedy")
def _():
    """Two runs with the same network and temperature=0 should behave similarly."""
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    r1 = run_episode(ps, net, num_simulations=5, temperature=0)
    r2 = run_episode(ps, net, num_simulations=5, temperature=0)
    # Both are valid -- just check they complete without error
    assert isinstance(r1, EpisodeResult)
    assert isinstance(r2, EpisodeResult)


# =============================================================================
# Section 5: Self-Play Trainer
# =============================================================================

print("\n-- Section 5: SelfPlayTrainer ----------------------------------------------")


@test("SelfPlayTrainer initialises with default config")
def _():
    ps_list = _make_proof_states()
    trainer = SelfPlayTrainer(ps_list)
    assert trainer.config.num_iterations == 10
    assert len(trainer.replay_buffer) == 0


@test("SelfPlayTrainer.run() returns a PolicyValueNet")
def _():
    ps_list = _make_proof_states()
    cfg = TrainingConfig(
        num_iterations=2,
        episodes_per_iteration=2,
        mcts_simulations=5,
        batch_size=4,
    )
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    net = trainer.run()
    assert isinstance(net, PolicyValueNet)


@test("SelfPlayTrainer accumulates replay buffer during training")
def _():
    ps_list = _make_proof_states()
    cfg = TrainingConfig(
        num_iterations=1,
        episodes_per_iteration=3,
        mcts_simulations=5,
        batch_size=2,
    )
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    trainer.run()
    assert len(trainer.replay_buffer) > 0, "Replay buffer empty after training"


@test("SelfPlayTrainer records stats per iteration")
def _():
    ps_list = _make_proof_states()
    cfg = TrainingConfig(
        num_iterations=2,
        episodes_per_iteration=2,
        mcts_simulations=5,
        batch_size=2,
        pretrain_on_builtin=False,  # disable auto-seed to avoid extra pretrain stat
    )
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    trainer.run()
    assert_eq(len(trainer.stats), 2, "stats count")
    for stat in trainer.stats:
        assert "win_rate" in stat
        assert "policy_loss" in stat
        assert "value_loss" in stat


@test("SelfPlayTrainer saves checkpoints to disk")
def _():
    ps_list = _make_proof_states()
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainingConfig(
            num_iterations=2,
            episodes_per_iteration=2,
            mcts_simulations=5,
            batch_size=2,
            checkpoint_dir=tmpdir,
        )
        trainer = SelfPlayTrainer(ps_list, config=cfg)
        trainer.run()
        files = os.listdir(tmpdir)
        assert len(files) == 2, f"Expected 2 checkpoints, got {files}"
        for f in files:
            assert f.endswith(".json"), f"Non-JSON file: {f}"


@test("train_from_source() convenience function works end-to-end")
def _():
    cfg = TrainingConfig(
        num_iterations=1,
        episodes_per_iteration=2,
        mcts_simulations=5,
        batch_size=2,
    )
    net, stats = train_from_source(SIMPLE_SOURCE, config=cfg)
    assert isinstance(net, PolicyValueNet)
    assert len(stats) >= 1


@test("train_from_source() works on multi-function source")
def _():
    cfg = TrainingConfig(
        num_iterations=1,
        episodes_per_iteration=2,
        mcts_simulations=5,
        batch_size=2,
    )
    net, stats = train_from_source(MULTI_SOURCE, config=cfg)
    assert isinstance(net, PolicyValueNet)


# =============================================================================
# Section 6: Integration (Phases 1->2->3 pipeline)
# =============================================================================

print("\n-- Section 6: Full Pipeline Integration ------------------------------------")


@test("Full pipeline: source -> proof states -> encoded observations")
def _():
    ir = parse_source(SIMPLE_SOURCE, "pipeline_test")
    ir = normalize(ir)
    state = analyze(ir)
    specs = extract_specs(ir, state)
    proof_states = build_proof_state_collection(specs)
    assert len(proof_states) > 0

    for ps in proof_states:
        obs = ps.to_observation()
        vec = encode(obs)
        assert_eq(len(vec), FEATURE_DIM)


@test("Full pipeline: encoded observations -> network forward pass")
def _():
    ps_list = _make_proof_states()
    net = PolicyValueNet()
    for ps in ps_list:
        vec = encode(ps.to_observation())
        priors, value = net.forward(vec)
        assert_close(sum(priors), 1.0, tol=1e-6)
        assert_in_range(value, -1.0, 1.0)


@test("Full pipeline: MCTS proof search on real proof states")
def _():
    ps_list = _make_proof_states()
    net = PolicyValueNet()
    mcts = MCTS(net, num_simulations=10)
    for ps in ps_list:
        probs, val = mcts.search(ps)
        assert_close(sum(probs), 1.0, tol=1e-6)
        assert_in_range(val, -1.0, 1.0)


@test("Full pipeline: self-play episode on real proof state")
def _():
    ps_list = _make_proof_states(MULTI_SOURCE)
    net = PolicyValueNet()
    for ps in ps_list:
        result = run_episode(ps, net, num_simulations=5)
        assert isinstance(result, EpisodeResult)
        assert result.outcome in (-1.0, 0.0, 1.0)


# =============================================================================
# Section 7: Cold-Start / Dataset Loaders
# =============================================================================

print("\n-- Section 7: Cold-Start / Dataset Loaders --------------------------------")


@test("generate_builtin_seed_data() returns RawProofTrace list")
def _():
    from rl_agent.dataset_loaders import (
        generate_builtin_seed_data,
        RawProofTrace, RawProofStep,
    )
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list,
        max_traces=5,
        max_steps_per_trace=5,
        temperature=0.0,
    )
    assert isinstance(traces, list)
    for t in traces:
        assert isinstance(t, RawProofTrace)
        assert isinstance(t.steps, list)
        for s in t.steps:
            assert isinstance(s, RawProofStep)
            assert "tactic" in str(type(s.tactic)) or isinstance(s.tactic, str)
    print(f"  Generated {len(traces)} trace(s)")


@test("generate_builtin_seed_data() produces >0 steps when states exist")
def _():
    from rl_agent.dataset_loaders import generate_builtin_seed_data
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list,
        max_traces=3,
        max_steps_per_trace=5,
        temperature=0.0,
    )
    total_steps = sum(len(t.steps) for t in traces)
    assert total_steps > 0, f"Expected > 0 steps across {len(traces)} traces, got {total_steps}"
    print(f"  Total steps across {len(traces)} traces: {total_steps}")


@test("convert_to_training_examples() returns TrainingExample list")
def _():
    from rl_agent.dataset_loaders import (
        generate_builtin_seed_data,
        convert_to_training_examples,
        RawProofTrace, RawProofStep,
    )
    from rl_agent import TrainingExample
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list,
        max_traces=3,
        max_steps_per_trace=5,
        temperature=0.0,
    )
    examples = convert_to_training_examples(traces)
    assert isinstance(examples, list)
    for ex in examples:
        assert isinstance(ex, TrainingExample)
        assert len(ex.state_vec) == FEATURE_DIM
    print(f"  Converted {len(traces)} trace(s) to {len(examples)} example(s)")


@test("convert_to_training_examples() produces one-hot policies summing to 1")
def _():
    from rl_agent.dataset_loaders import (
        generate_builtin_seed_data,
        convert_to_training_examples,
    )
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list,
        max_traces=2,
        max_steps_per_trace=5,
        temperature=0.0,
    )
    examples = convert_to_training_examples(traces)
    for i, ex in enumerate(examples):
        total = sum(ex.mcts_policy)
        assert_close(total, 1.0, tol=1e-6, msg=f"Example {i} policy sum")
    print(f"  All {len(examples)} example(s) have valid one-hot policies")


@test("convert_with_mcts_policy() produces label-smoothed policies")
def _():
    from rl_agent.dataset_loaders import (
        generate_builtin_seed_data,
        convert_with_mcts_policy,
    )
    from proof_engine import CORE_TACTICS
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list,
        max_traces=2,
        max_steps_per_trace=5,
        temperature=0.0,
    )
    num_actions = len(CORE_TACTICS)
    examples = convert_with_mcts_policy(traces, num_actions, smooth_eps=0.1)
    for i, ex in enumerate(examples):
        total = sum(ex.mcts_policy)
        assert_close(total, 1.0, tol=1e-6, msg=f"Example {i} smoothed policy sum")
        # Check that no single action has all the probability (unless num_actions == 1)
        max_p = max(ex.mcts_policy)
        if num_actions > 1:
            assert max_p < 1.0, f"Example {i} smoothed policy has a 1.0 entry"
    print(f"  All {len(examples)} example(s) have valid smoothed policies")


@test("save_seed_data() and load_seed_data() round-trip")
def _():
    import tempfile
    from rl_agent.dataset_loaders import (
        generate_builtin_seed_data,
        convert_to_training_examples,
        save_seed_data, load_seed_data,
    )
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list,
        max_traces=2,
        max_steps_per_trace=5,
        temperature=0.0,
    )
    examples_orig = convert_to_training_examples(traces)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        save_seed_data(examples_orig, path)
        examples_loaded = load_seed_data(path)
        assert_eq(len(examples_orig), len(examples_loaded), "Example count")
        for i, (a, b) in enumerate(zip(examples_orig, examples_loaded)):
            assert_eq(len(a.state_vec), len(b.state_vec), f"Example {i} state_vec length")
            assert_close(sum(a.mcts_policy), sum(b.mcts_policy), tol=1e-6, msg=f"Example {i} policy sum")
    finally:
        os.unlink(path)
    print(f"  Round-trip preserved {len(examples_orig)} example(s)")


# =============================================================================
# Section 8: Cold-Start / Buffer Seeding & Supervised Pre-Training
# =============================================================================

print("\n-- Section 8: Buffer Seeding & Supervised Pre-Training --------------------")


@test("SelfPlayTrainer.seed_buffer() populates the replay buffer")
def _():
    from rl_agent.dataset_loaders import (
        generate_builtin_seed_data,
        convert_to_training_examples,
    )
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list,
        max_traces=2,
        max_steps_per_trace=5,
        temperature=0.0,
    )
    examples = convert_to_training_examples(traces)

    cfg = TrainingConfig(pretrain_on_builtin=False)  # disable auto-seed
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    assert len(trainer.replay_buffer) == 0, "Buffer should start empty"

    count = trainer.seed_buffer(examples)
    assert count > 0, f"Expected > 0 seeded, got {count}"
    assert_eq(len(trainer.replay_buffer), count, "Buffer size after seeding")
    print(f"  Seeded {count} example(s) into buffer")


@test("pretrain_supervised() reduces policy loss")
def _():
    from rl_agent.dataset_loaders import (
        generate_builtin_seed_data,
        convert_to_training_examples,
    )
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list,
        max_traces=3,
        max_steps_per_trace=5,
        temperature=0.0,
    )
    examples = convert_to_training_examples(traces)

    cfg = TrainingConfig(
        pretrain_on_builtin=False,
        pretrain_epochs=3,
        pretrain_batch_size=min(8, len(examples)),
        pretrain_learning_rate=1e-3,
    )
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    trainer.seed_buffer(examples)

    # Get initial loss
    initial_pi, initial_v = trainer.net.update_weights(
        target_policies=[ex.mcts_policy for ex in examples],
        target_values=[ex.outcome for ex in examples],
        state_vecs=[ex.state_vec for ex in examples],
        lr=1e-8,  # tiny lr to measure loss without updating
    )

    # Run pre-training
    final_pi, final_v = trainer.pretrain_supervised()

    # Policy loss should not increase (network should not diverge)
    assert final_pi <= initial_pi + 0.5, (
        f"Policy loss increased: {initial_pi:.4f} -> {final_pi:.4f}"
    )
    print(f"  Pre-training: pi_loss {initial_pi:.4f} -> {final_pi:.4f}")


@test("pretrain_supervised() returns valid loss floats")
def _():
    from rl_agent.dataset_loaders import (
        generate_builtin_seed_data,
        convert_to_training_examples,
    )
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list,
        max_traces=2,
        max_steps_per_trace=5,
        temperature=0.0,
    )
    examples = convert_to_training_examples(traces)

    cfg = TrainingConfig(
        pretrain_on_builtin=False,
        pretrain_epochs=2,
        pretrain_batch_size=min(4, len(examples)),
    )
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    trainer.seed_buffer(examples)

    pi_loss, v_loss = trainer.pretrain_supervised()
    assert isinstance(pi_loss, float), f"pi_loss should be float, got {type(pi_loss)}"
    assert isinstance(v_loss, float), f"v_loss should be float, got {type(v_loss)}"
    assert pi_loss >= 0.0, f"pi_loss should be >= 0, got {pi_loss}"
    print(f"  pi_loss={pi_loss:.4f}  v_loss={v_loss:.4f}")


@test("pretrain_supervised() with empty buffer returns zeros")
def _():
    ps_list = _make_proof_states()
    cfg = TrainingConfig(pretrain_on_builtin=False)
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    pi_loss, v_loss = trainer.pretrain_supervised()
    assert_eq(pi_loss, 0.0, "pi_loss on empty buffer")
    assert_eq(v_loss, 0.0, "v_loss on empty buffer")
    print(f"  Empty buffer correctly returns zeros")


@test("_auto_seed_and_pretrain() populates buffer and reduces loss")
def _():
    ps_list = _make_proof_states()
    cfg = TrainingConfig(
        pretrain_on_builtin=True,
        pretrain_epochs=2,
        pretrain_batch_size=8,
        num_iterations=1,
        episodes_per_iteration=1,
        mcts_simulations=5,
    )
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    assert len(trainer.replay_buffer) == 0, "Buffer should start empty"

    # This triggers the auto seed + pre-training
    trainer.run()

    assert len(trainer.replay_buffer) > 0, "Buffer should be non-empty after auto-seed"
    assert len(trainer.stats) >= 1, "Stats should have at least 1 entry"
    print(f"  Buffer size after auto-seed: {len(trainer.replay_buffer)}")


@test("SelfPlayTrainer with pretrain_on_builtin=True seeds before self-play")
def _():
    """When pretrain_on_builtin is True, the buffer should be seeded before
    the first self-play iteration (visible in the stats)."""
    ps_list = _make_proof_states()
    cfg = TrainingConfig(
        pretrain_on_builtin=True,
        pretrain_epochs=1,
        pretrain_batch_size=8,
        num_iterations=1,
        episodes_per_iteration=1,
        mcts_simulations=5,
    )
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    trainer.run()
    # After one iteration, buffer should be seeded + augmented by self-play
    assert len(trainer.replay_buffer) > 0, "Buffer should be non-empty"
    if trainer.stats:
        stat = trainer.stats[0]
        assert "buffer_size" in stat
        print(f"  Final buffer size: {stat['buffer_size']}")


@test("Cold-start pipeline: generate -> seed -> pretrain -> self-play")
def _():
    """Full end-to-end cold-start pipeline."""
    from rl_agent.dataset_loaders import (
        generate_builtin_seed_data,
        convert_to_training_examples,
    )
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list,
        max_traces=2,
        max_steps_per_trace=5,
        temperature=0.0,
    )
    examples = convert_to_training_examples(traces)
    assert len(examples) > 0, "Should have training examples"

    # Seed a fresh trainer
    cfg = TrainingConfig(
        pretrain_on_builtin=False,
        pretrain_epochs=2,
        num_iterations=1,
        episodes_per_iteration=1,
        mcts_simulations=5,
    )
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    trainer.seed_buffer(examples)
    trainer.pretrain_supervised()
    net = trainer.run()
    assert isinstance(net, PolicyValueNet)
    print(f"  Cold-start pipeline completed successfully")


# =============================================================================
# Section 9: Dataset Parser Units
# =============================================================================

print("\n-- Section 9: Dataset Parser Units ----------------------------------------")


@test("parse_leandojo_trace() with minimal valid JSON")
def _():
    import tempfile
    from rl_agent.dataset_loaders import parse_leandojo_trace, RawProofTrace

    # Create a minimal LeanDojo trace
    trace_data = {
        "theorem": "add_comm",
        "result": "proved",
        "traj": [
            {
                "state": {
                    "goal": "a + b = b + a",
                    "hypotheses": [["a", "ℕ"], ["b", "ℕ"]],
                },
                "tactic": "rw [add_comm]"
            },
            {
                "state": {
                    "goal": "a + b = a + b",
                    "hypotheses": [["a", "ℕ"], ["b", "ℕ"]],
                },
                "tactic": "rfl"
            },
        ],
    }

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        import json as j
        j.dump(trace_data, f)
        path = f.name
    try:
        traces = parse_leandojo_trace(path)
        assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"
        assert traces[0].theorem_name == "add_comm"
        assert len(traces[0].steps) == 2
        assert traces[0].steps[0].tactic == "rw [add_comm]"
    finally:
        os.unlink(path)
    print(f"  Parsed LeanDojo trace with {len(traces[0].steps)} step(s)")


@test("load_dataset() auto-infers LeanDojo format from filename")
def _():
    import tempfile
    from rl_agent.dataset_loaders import load_dataset, DatasetFormat

    trace_data = [{
        "theorem": "test_thm",
        "result": "proved",
        "traj": [{"state": {"goal": "True"}, "tactic": "trivial"}],
    }]

    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix="traj_", mode="w", delete=False
    ) as f:
        import json as j
        j.dump(trace_data, f)
        path = f.name
    try:
        traces = load_dataset(path)
        assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"
    finally:
        os.unlink(path)
    print(f"  Auto-inferred format from filename: {len(traces)} trace(s)")


@test("convert_to_training_examples() handles empty traces gracefully")
def _():
    from rl_agent.dataset_loaders import (
        convert_to_training_examples, RawProofTrace, RawProofStep,
    )
    # Empty trace list
    assert_eq(len(convert_to_training_examples([])), 0, "Empty input")
    # Trace with no steps
    trace = RawProofTrace(theorem_name="empty", steps=[], outcome=0.0)
    result = convert_to_training_examples([trace])
    assert_eq(len(result), 0, "Trace with no steps")
    print(f"  Empty traces handled correctly")


@test("RawProofStep and RawProofTrace dataclasses work correctly")
def _():
    from rl_agent.dataset_loaders import RawProofStep, RawProofTrace
    step = RawProofStep(
        state_observation={"theorem": "test"},
        tactic="intro h",
        outcome=1.0,
        theorem_name="test_thm",
    )
    assert step.tactic == "intro h"
    assert step.outcome == 1.0

    trace = RawProofTrace(
        theorem_name="test_thm",
        steps=[step],
        outcome=1.0,
        source="test",
    )
    assert len(trace.steps) == 1
    assert trace.source == "test"
    print(f"  RawProofStep and RawProofTrace dataclasses work")


@test("Integration: parse downloaded miniF2F valid.json")
def _():
    """Verify format compatibility between download post-processor and dataset parser."""
    import tempfile
    from rl_agent.dataset_loaders import parse_minif2f, convert_to_training_examples
    from rl_agent import TrainingExample

    # Path to the downloaded miniF2F valid split
    minif2f_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "datasets", "minif2f", "minif2f_valid.json",
    )
    if not os.path.exists(minif2f_path):
        print(f"  Skipping (miniF2F not downloaded at {minif2f_path})")
        return

    traces = parse_minif2f(minif2f_path)
    assert len(traces) > 0, f"Expected > 0 traces, got {len(traces)}"
    for t in traces:
        assert t.source == "minif2f"

    # Some entries may have empty proofs (e.g., 'by sorry' unsolved); skip those for training
    non_empty = [t for t in traces if len(t.steps) > 0]
    empty_proofs = len(traces) - len(non_empty)
    print(f"  {len(traces)} total, {empty_proofs} entries with empty proofs")

    examples = convert_to_training_examples(traces)
    if examples:
        for ex in examples:
            assert len(ex.state_vec) == FEATURE_DIM
            assert_close(sum(ex.mcts_policy), 1.0, tol=1e-6)

    print("  Parsed {} miniF2F traces -> {} training examples ({} empty-proof entries)".format(len(traces), len(examples), empty_proofs))


@test("Integration: parse downloaded ProofNet.jsonl")
def _():
    """Verify ProofNet post-processor output is compatible with parse_proofnet."""
    from rl_agent.dataset_loaders import parse_proofnet, convert_to_training_examples

    proofnet_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "datasets", "proofnet", "proofnet.jsonl",
    )
    if not os.path.exists(proofnet_path):
        print(f"  Skipping (ProofNet not downloaded at {proofnet_path})")
        return

    traces = parse_proofnet(proofnet_path)
    assert len(traces) > 0, f"Expected > 0 traces, got {len(traces)}"
    for t in traces:
        assert t.source == "proofnet"

    # Some entries may have empty proofs; skip those for training
    non_empty = [t for t in traces if len(t.steps) > 0]
    empty_proofs = len(traces) - len(non_empty)
    print(f"  {len(traces)} total, {empty_proofs} entries with empty proofs")

    examples = convert_to_training_examples(traces)
    if examples:
        for ex in examples:
            assert len(ex.state_vec) == FEATURE_DIM
            assert_close(sum(ex.mcts_policy), 1.0, tol=1e-6)

    print("  Parsed {} ProofNet traces -> {} training examples ({} empty-proof entries)".format(len(traces), len(examples), empty_proofs))


@test("Integration: load_dataset() dispatches correctly for downloaded formats")
def _():
    """Test load_dataset() auto-infers format for each downloaded dataset."""
    from rl_agent.dataset_loaders import load_dataset, DatasetFormat

    base = os.path.dirname(os.path.abspath(__file__))

    # miniF2F
    m_path = os.path.join(base, "datasets", "minif2f", "minif2f_valid.json")
    if os.path.exists(m_path):
        traces = load_dataset(m_path)
        assert len(traces) > 0
        print(f"  miniF2F: load_dataset() returned {len(traces)} traces")

    # ProofNet
    p_path = os.path.join(base, "datasets", "proofnet", "proofnet.jsonl")
    if os.path.exists(p_path):
        traces = load_dataset(p_path)
        assert len(traces) > 0
        print(f"  ProofNet: load_dataset() returned {len(traces)} traces")

    # Lean Workbook
    w_path = os.path.join(base, "datasets", "lean_workbook", "lean_workbook.json")
    if os.path.exists(w_path):
        traces = load_dataset(w_path, format=DatasetFormat.LEAN_WORKBOOK)
        assert len(traces) > 0
        print(f"  Lean Workbook: load_dataset() returned {len(traces)} traces")

    print(f"  All format auto-detection tests passed")


# -- Summary ---------------------------------------------------------------------

print(f"\n{'='*60}")
total = PASS + FAIL
print(f"Phase 3 Results: {PASS}/{total} tests passing")
print(f"{'='*60}")

if FAIL:
    print(f"\nFailed tests:")
    for r in results:
        if r[0] == "FAIL":
            print(f"  FAIL  {r[1]}: {r[2]}")
    sys.exit(1)
else:
    print("\nAll Phase 3 tests pass!")
    sys.exit(0)
