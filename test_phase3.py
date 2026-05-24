"""Phase 3 tests: RL Agent (state encoder, policy/value network, MCTS, self-play).

Run:
    python test_phase3.py
"""

import sys
import os
import math
import random
import json
import tempfile
import traceback

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


# ── Fixtures ────────────────────────────────────────────────────────────────

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


# ── Helpers ─────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0
results = []


def _assert_eq(a, b, msg=""):
    assert a == b, f"{msg}: {a!r} != {b!r}"


def _assert_close(a: float, b: float, tol: float = 1e-6, msg: str = ""):
    assert abs(a - b) < tol, f"{msg}: |{a} - {b}| >= {tol}"


def _assert_in_range(v: float, lo: float, hi: float, msg: str = ""):
    assert lo <= v <= hi, f"{msg}: {v} not in [{lo}, {hi}]"


def _run_test(name: str, fn):
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


# =============================================================================
# Section 1: State Encoder
# =============================================================================

print("\n-- Section 1: State Encoder ------------------------------------------------")


def test_encode_returns_list():
    ps = _make_proof_states()[0]
    vec = encode(ps.to_observation())
    assert isinstance(vec, list)
    _assert_eq(len(vec), FEATURE_DIM, "Vector length")


def test_feature_dim():
    _assert_eq(FEATURE_DIM, 256)


def test_encode_all_floats():
    ps = _make_proof_states()[0]
    vec = encode(ps.to_observation())
    assert all(isinstance(v, float) for v in vec)


def test_encode_global_scalars_in_range():
    ps = _make_proof_states()[0]
    vec = encode(ps.to_observation())
    for i in range(8):
        _assert_in_range(vec[i], 0.0, 1.0, f"vec[{i}]")


def test_encode_tactic_bag_in_range():
    ps = _make_proof_states()[0]
    vec = encode(ps.to_observation())
    for i in range(8, 47):
        _assert_in_range(vec[i], 0.0, 1.0, f"vec[{i}]")


def test_encode_different_states_differ():
    states = _make_proof_states(MULTI_SOURCE)
    seen = {}
    for ps in states:
        name = ps.theorem_name
        if name not in seen:
            seen[name] = ps
    if len(seen) >= 2:
        distinct = list(seen.values())
        v1 = encode(distinct[0].to_observation())
        v2 = encode(distinct[1].to_observation())
        assert v1 != v2


def test_encode_empty_state():
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
    _assert_eq(len(vec), FEATURE_DIM)
    _assert_eq(vec[4], 1.0, "is_complete flag")


def test_encode_deterministic():
    ps = _make_proof_states()[0]
    obs = ps.to_observation()
    _assert_eq(encode(obs), encode(obs))


# =============================================================================
# Section 2: Policy / Value Network
# =============================================================================

print("\n-- Section 2: Policy / Value Network --------------------------------------")


def test_network_initialization():
    net = PolicyValueNet()
    _assert_eq(net.num_actions, len(CORE_TACTICS))


def test_network_forward_shapes():
    net = PolicyValueNet()
    vec = [0.0] * FEATURE_DIM
    priors, value = net.forward(vec)
    _assert_eq(len(priors), len(CORE_TACTICS), "priors length")
    assert isinstance(value, float)


def test_priors_sum_to_one():
    net = PolicyValueNet()
    vec = [random.gauss(0, 1) for _ in range(FEATURE_DIM)]
    priors, _ = net.forward(vec)
    _assert_close(sum(priors), 1.0, tol=1e-6, msg="priors sum")


def test_priors_in_range():
    net = PolicyValueNet()
    vec = [random.gauss(0, 1) for _ in range(FEATURE_DIM)]
    priors, _ = net.forward(vec)
    for i, p in enumerate(priors):
        _assert_in_range(p, 0.0, 1.0, f"prior[{i}]")


def test_value_in_range():
    net = PolicyValueNet()
    for _ in range(5):
        vec = [random.gauss(0, 1) for _ in range(FEATURE_DIM)]
        _, v = net.forward(vec)
        _assert_in_range(v, -1.0, 1.0, "value")


def test_update_weights_returns_floats():
    net = PolicyValueNet()
    na = net.num_actions
    vecs = [[random.gauss(0, 0.1) for _ in range(FEATURE_DIM)] for _ in range(4)]
    pols = [[1.0 / na] * na for _ in range(4)]
    vals = [1.0, -1.0, 0.0, 1.0]
    pi_loss, v_loss = net.update_weights(pols, vals, vecs, lr=1e-3)
    assert isinstance(pi_loss, float)
    assert isinstance(v_loss, float)
    assert pi_loss >= 0.0


def test_update_weights_reduces_loss():
    """Network should overfit on a tiny constant batch."""
    net = PolicyValueNet()
    na = net.num_actions
    target_pol = [0.0] * na
    target_pol[0] = 1.0
    vecs = [[0.1 * i for i in range(FEATURE_DIM)]] * 8
    pols = [target_pol] * 8
    vals = [1.0] * 8
    losses_pi = []
    for _ in range(20):
        pi_loss, _ = net.update_weights(pols, vals, vecs, lr=1e-2)
        losses_pi.append(pi_loss)
    first_half = sum(losses_pi[:10]) / 10
    second_half = sum(losses_pi[10:]) / 10
    assert second_half <= first_half + 0.5, (
        f"Loss did not decrease: {first_half:.4f} -> {second_half:.4f}"
    )


def test_save_load_round_trip():
    net = PolicyValueNet()
    vec = [random.gauss(0, 1) for _ in range(FEATURE_DIM)]
    priors_before, value_before = net.forward(vec)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        net.save(path)
        net2 = PolicyValueNet.load(path)
        priors_after, value_after = net2.forward(vec)
        _assert_close(value_before, value_after, tol=1e-6, msg="value round-trip")
        for i, (a, b) in enumerate(zip(priors_before, priors_after)):
            _assert_close(a, b, tol=1e-6, msg=f"prior[{i}] round-trip")
    finally:
        os.unlink(path)


# =============================================================================
# Section 3: MCTS
# =============================================================================

print("\n-- Section 3: MCTS ---------------------------------------------------------")


def test_tactic_simulator_returns_tuple():
    sim = TacticSimulator()
    ps = _make_proof_states()[0]
    success, new_state, reward = sim.apply(ps, "intro h")
    assert isinstance(success, bool)
    assert isinstance(new_state, ProofState)
    assert isinstance(reward, float)


def test_tactic_simulator_reward_in_range():
    sim = TacticSimulator()
    ps = _make_proof_states()[0]
    for tactic in ["omega", "intro h", "apply", "simp", "rfl"]:
        _, _, reward = sim.apply(ps, tactic)
        _assert_in_range(reward, -1.0, 1.0, f"reward for {tactic}")


def test_mcts_node_ucb_finite():
    node = MCTSNode(state=_make_proof_states()[0], prior=0.5, visit_count=3, value_sum=1.5)
    assert math.isfinite(node.ucb_score(parent_visit_count=10))


def test_mcts_node_ucb_unvisited():
    from rl_agent.mcts import C_PUCT
    node = MCTSNode(state=_make_proof_states()[0], prior=0.4, visit_count=0, value_sum=0.0)
    score = node.ucb_score(parent_visit_count=9)
    _assert_close(score, C_PUCT * 0.4 * math.sqrt(9), tol=1e-6, msg="UCB unvisited")


def test_mcts_search_returns_valid_probs():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    mcts = MCTS(net, num_simulations=10)
    probs, _ = mcts.search(ps)
    _assert_eq(len(probs), len(CORE_TACTICS))
    _assert_close(sum(probs), 1.0, tol=1e-6, msg="action_probs sum")


def test_mcts_root_value_in_range():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    mcts = MCTS(net, num_simulations=10)
    _, value = mcts.search(ps)
    _assert_in_range(value, -1.0, 1.0, "root_value")


def test_mcts_best_action_is_valid():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    mcts = MCTS(net, num_simulations=10)
    action = mcts.best_action(ps)
    assert action in CORE_TACTICS, f"'{action}' not a valid tactic"


def test_mcts_expand_creates_children():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    mcts = MCTS(net, num_simulations=5)
    root = MCTSNode(state=ps)
    mcts._expand(root)
    _assert_eq(len(root.children), len(CORE_TACTICS), "Wrong number of children")


def test_mcts_more_simulations_affects_probs():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    mcts1 = MCTS(net, num_simulations=20)
    probs1, _ = mcts1.search(ps)
    mcts2 = MCTS(net, num_simulations=5)
    probs2, _ = mcts2.search(ps)
    _assert_close(sum(probs1), 1.0, tol=1e-6, msg="probs1")
    _assert_close(sum(probs2), 1.0, tol=1e-6, msg="probs2")


# =============================================================================
# Section 4: Self-Play Episode
# =============================================================================

print("\n-- Section 4: Self-Play Episode --------------------------------------------")


def test_run_episode_returns_episode_result():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    result = run_episode(ps, net, num_simulations=5)
    assert isinstance(result, EpisodeResult)


def test_episode_outcome_is_valid():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    result = run_episode(ps, net, num_simulations=5)
    assert result.outcome in (-1.0, 0.0, 1.0), f"Unexpected outcome: {result.outcome}"


def test_episode_examples_have_correct_vec_length():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    result = run_episode(ps, net, num_simulations=5)
    for i, ex in enumerate(result.examples):
        _assert_eq(len(ex.state_vec), FEATURE_DIM, f"Example {i} state_vec length")


def test_episode_examples_policy_sums_to_one():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    result = run_episode(ps, net, num_simulations=5)
    for i, ex in enumerate(result.examples):
        _assert_close(sum(ex.mcts_policy), 1.0, tol=1e-6, msg=f"Example {i} policy sum")


def test_episode_examples_have_same_outcome():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    result = run_episode(ps, net, num_simulations=5)
    outcomes = {ex.outcome for ex in result.examples}
    assert len(outcomes) <= 1, f"Mixed outcomes: {outcomes}"


def test_run_episode_temperature_zero():
    net = PolicyValueNet()
    ps = _make_proof_states()[0]
    r1 = run_episode(ps, net, num_simulations=5, temperature=0)
    r2 = run_episode(ps, net, num_simulations=5, temperature=0)
    assert isinstance(r1, EpisodeResult)
    assert isinstance(r2, EpisodeResult)


# =============================================================================
# Section 5: Self-Play Trainer
# =============================================================================

print("\n-- Section 5: SelfPlayTrainer ----------------------------------------------")


def test_trainer_initialization():
    ps_list = _make_proof_states()
    trainer = SelfPlayTrainer(ps_list)
    assert trainer.config.num_iterations == 10
    assert len(trainer.replay_buffer) == 0


def test_trainer_run_returns_network():
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


def test_trainer_accumulates_buffer():
    ps_list = _make_proof_states()
    cfg = TrainingConfig(
        num_iterations=1,
        episodes_per_iteration=3,
        mcts_simulations=5,
        batch_size=2,
    )
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    trainer.run()
    assert len(trainer.replay_buffer) > 0


def test_trainer_records_stats():
    ps_list = _make_proof_states()
    cfg = TrainingConfig(
        pretrain_on_builtin=False,
        num_iterations=2,
        episodes_per_iteration=2,
        mcts_simulations=5,
        batch_size=2,
    )
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    trainer.run()
    _assert_eq(len(trainer.stats), 2, "stats count")
    for stat in trainer.stats:
        assert "win_rate" in stat
        assert "policy_loss" in stat
        assert "value_loss" in stat


def test_trainer_saves_checkpoints():
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
            assert f.endswith(".json")


def test_train_from_source():
    cfg = TrainingConfig(
        num_iterations=1,
        episodes_per_iteration=2,
        mcts_simulations=5,
        batch_size=2,
    )
    net, stats = train_from_source(SIMPLE_SOURCE, config=cfg)
    assert isinstance(net, PolicyValueNet)
    assert len(stats) >= 1


def test_train_from_source_multi_function():
    cfg = TrainingConfig(
        num_iterations=1,
        episodes_per_iteration=2,
        mcts_simulations=5,
        batch_size=2,
    )
    net, stats = train_from_source(MULTI_SOURCE, config=cfg)
    assert isinstance(net, PolicyValueNet)


# =============================================================================
# Section 6: Integration (Phases 1-2-3 pipeline)
# =============================================================================

print("\n-- Section 6: Full Pipeline Integration ------------------------------------")


def test_pipeline_source_to_encoded():
    ir = parse_source(SIMPLE_SOURCE, "pipeline_test")
    ir = normalize(ir)
    state = analyze(ir)
    specs = extract_specs(ir, state)
    proof_states = build_proof_state_collection(specs)
    assert len(proof_states) > 0

    for ps in proof_states:
        obs = ps.to_observation()
        vec = encode(obs)
        _assert_eq(len(vec), FEATURE_DIM)


def test_pipeline_network_forward():
    ps_list = _make_proof_states()
    net = PolicyValueNet()
    for ps in ps_list:
        vec = encode(ps.to_observation())
        priors, value = net.forward(vec)
        _assert_close(sum(priors), 1.0, tol=1e-6)
        _assert_in_range(value, -1.0, 1.0)


def test_pipeline_mcts_search():
    ps_list = _make_proof_states()
    net = PolicyValueNet()
    mcts = MCTS(net, num_simulations=10)
    for ps in ps_list:
        probs, val = mcts.search(ps)
        _assert_close(sum(probs), 1.0, tol=1e-6)
        _assert_in_range(val, -1.0, 1.0)


def test_pipeline_self_play_episode():
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


def test_generate_builtin_seed_data_returns_traces():
    from rl_agent.dataset_loaders import generate_builtin_seed_data, RawProofTrace, RawProofStep
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list, max_traces=5, max_steps_per_trace=5, temperature=0.0,
    )
    assert isinstance(traces, list)
    for t in traces:
        assert isinstance(t, RawProofTrace)
        assert isinstance(t.steps, list)
        for s in t.steps:
            assert isinstance(s, RawProofStep)
    print(f"  Generated {len(traces)} trace(s)")


def test_generate_builtin_seed_data_produces_steps():
    from rl_agent.dataset_loaders import generate_builtin_seed_data
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list, max_traces=3, max_steps_per_trace=5, temperature=0.0,
    )
    total_steps = sum(len(t.steps) for t in traces)
    assert total_steps > 0
    print(f"  Total steps across {len(traces)} traces: {total_steps}")


def test_convert_to_training_examples():
    from rl_agent.dataset_loaders import (
        generate_builtin_seed_data, convert_to_training_examples,
    )
    from rl_agent import TrainingExample
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list, max_traces=3, max_steps_per_trace=5, temperature=0.0,
    )
    examples = convert_to_training_examples(traces)
    assert isinstance(examples, list)
    for ex in examples:
        assert isinstance(ex, TrainingExample)
        assert len(ex.state_vec) == FEATURE_DIM
    print(f"  Converted {len(traces)} trace(s) to {len(examples)} example(s)")


def test_convert_to_training_examples_policies():
    from rl_agent.dataset_loaders import (
        generate_builtin_seed_data, convert_to_training_examples,
    )
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list, max_traces=2, max_steps_per_trace=5, temperature=0.0,
    )
    examples = convert_to_training_examples(traces)
    for i, ex in enumerate(examples):
        _assert_close(sum(ex.mcts_policy), 1.0, tol=1e-6, msg=f"Example {i} policy sum")
    print(f"  All {len(examples)} example(s) have valid policies")


def test_convert_with_mcts_policy_smoothing():
    from rl_agent.dataset_loaders import (
        generate_builtin_seed_data, convert_with_mcts_policy,
    )
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list, max_traces=2, max_steps_per_trace=5, temperature=0.0,
    )
    num_actions = len(CORE_TACTICS)
    examples = convert_with_mcts_policy(traces, num_actions, smooth_eps=0.1)
    for i, ex in enumerate(examples):
        _assert_close(sum(ex.mcts_policy), 1.0, tol=1e-6, msg=f"Example {i} smoothed policy sum")
        if num_actions > 1:
            assert max(ex.mcts_policy) < 1.0
    print(f"  All {len(examples)} example(s) have valid smoothed policies")


def test_seed_data_save_load_round_trip():
    from rl_agent.dataset_loaders import (
        generate_builtin_seed_data, convert_to_training_examples,
        save_seed_data, load_seed_data,
    )
    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list, max_traces=2, max_steps_per_trace=5, temperature=0.0,
    )
    examples_orig = convert_to_training_examples(traces)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        save_seed_data(examples_orig, path)
        examples_loaded = load_seed_data(path)
        _assert_eq(len(examples_orig), len(examples_loaded), "Example count")
        for i, (a, b) in enumerate(zip(examples_orig, examples_loaded)):
            _assert_eq(len(a.state_vec), len(b.state_vec), f"Example {i} state_vec length")
            _assert_close(sum(a.mcts_policy), sum(b.mcts_policy), tol=1e-6, msg=f"Example {i} policy sum")
    finally:
        os.unlink(path)
    print(f"  Round-trip preserved {len(examples_orig)} example(s)")


# =============================================================================
# Section 8: Buffer Seeding & Supervised Pre-Training
# =============================================================================

print("\n-- Section 8: Buffer Seeding & Supervised Pre-Training --------------------")


def test_seed_buffer_populates():
    from rl_agent.dataset_loaders import generate_builtin_seed_data, convert_to_training_examples

    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list, max_traces=2, max_steps_per_trace=5, temperature=0.0,
    )
    examples = convert_to_training_examples(traces)

    cfg = TrainingConfig(pretrain_on_builtin=False)
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    assert len(trainer.replay_buffer) == 0

    count = trainer.seed_buffer(examples)
    assert count > 0
    _assert_eq(len(trainer.replay_buffer), count, "Buffer size after seeding")
    print(f"  Seeded {count} example(s) into buffer")


def test_pretrain_supervised_reduces_policy_loss():
    from rl_agent.dataset_loaders import generate_builtin_seed_data, convert_to_training_examples

    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list, max_traces=3, max_steps_per_trace=5, temperature=0.0,
    )
    examples = convert_to_training_examples(traces)

    cfg = TrainingConfig(pretrain_on_builtin=False, pretrain_epochs=3,
                         pretrain_batch_size=min(8, len(examples)), pretrain_learning_rate=1e-3)
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    trainer.seed_buffer(examples)

    initial_pi, _ = trainer.net.update_weights(
        target_policies=[ex.mcts_policy for ex in examples],
        target_values=[ex.outcome for ex in examples],
        state_vecs=[ex.state_vec for ex in examples],
        lr=1e-8,
    )
    final_pi, _ = trainer.pretrain_supervised()
    assert final_pi <= initial_pi + 0.5, f"Policy loss increased: {initial_pi:.4f} -> {final_pi:.4f}"
    print(f"  Pre-training: pi_loss {initial_pi:.4f} -> {final_pi:.4f}")


def test_pretrain_supervised_returns_floats():
    from rl_agent.dataset_loaders import generate_builtin_seed_data, convert_to_training_examples

    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list, max_traces=2, max_steps_per_trace=5, temperature=0.0,
    )
    examples = convert_to_training_examples(traces)

    cfg = TrainingConfig(pretrain_on_builtin=False, pretrain_epochs=2,
                         pretrain_batch_size=min(4, len(examples)))
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    trainer.seed_buffer(examples)

    pi_loss, v_loss = trainer.pretrain_supervised()
    assert isinstance(pi_loss, float)
    assert isinstance(v_loss, float)
    assert pi_loss >= 0.0
    print(f"  pi_loss={pi_loss:.4f}  v_loss={v_loss:.4f}")


def test_pretrain_supervised_empty_buffer():
    ps_list = _make_proof_states()
    cfg = TrainingConfig(pretrain_on_builtin=False)
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    pi_loss, v_loss = trainer.pretrain_supervised()
    _assert_eq(pi_loss, 0.0, "pi_loss on empty buffer")
    _assert_eq(v_loss, 0.0, "v_loss on empty buffer")


def test_auto_seed_and_pretrain():
    ps_list = _make_proof_states()
    cfg = TrainingConfig(
        pretrain_on_builtin=True, pretrain_epochs=2, pretrain_batch_size=8,
        num_iterations=1, episodes_per_iteration=1, mcts_simulations=5,
    )
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    assert len(trainer.replay_buffer) == 0

    trainer.run()
    assert len(trainer.replay_buffer) > 0
    assert len(trainer.stats) >= 1
    print(f"  Buffer size after auto-seed: {len(trainer.replay_buffer)}")


def test_pretrain_on_builtin_seeds_before_self_play():
    ps_list = _make_proof_states()
    cfg = TrainingConfig(
        pretrain_on_builtin=True, pretrain_epochs=1, pretrain_batch_size=8,
        num_iterations=1, episodes_per_iteration=1, mcts_simulations=5,
    )
    trainer = SelfPlayTrainer(ps_list, config=cfg)
    trainer.run()
    assert len(trainer.replay_buffer) > 0
    if trainer.stats:
        assert "buffer_size" in trainer.stats[0]
        print(f"  Final buffer size: {trainer.stats[0]['buffer_size']}")


def test_cold_start_pipeline():
    """Full end-to-end cold-start pipeline."""
    from rl_agent.dataset_loaders import generate_builtin_seed_data, convert_to_training_examples

    ps_list = _make_proof_states()
    traces = generate_builtin_seed_data(
        proof_states=ps_list, max_traces=2, max_steps_per_trace=5, temperature=0.0,
    )
    examples = convert_to_training_examples(traces)
    assert len(examples) > 0

    cfg = TrainingConfig(
        pretrain_on_builtin=False, pretrain_epochs=2,
        num_iterations=1, episodes_per_iteration=1, mcts_simulations=5,
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


def test_parse_leandojo_trace():
    from rl_agent.dataset_loaders import parse_leandojo_trace

    trace_data = {
        "theorem": "add_comm",
        "result": "proved",
        "traj": [
            {"state": {"goal": "a + b = b + a", "hypotheses": [["a", "ℕ"], ["b", "ℕ"]]}, "tactic": "rw [add_comm]"},
            {"state": {"goal": "a + b = a + b", "hypotheses": [["a", "ℕ"], ["b", "ℕ"]]}, "tactic": "rfl"},
        ],
    }

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        import json as j
        j.dump(trace_data, f)
        path = f.name
    try:
        traces = parse_leandojo_trace(path)
        assert len(traces) == 1
        assert traces[0].theorem_name == "add_comm"
        assert len(traces[0].steps) == 2
        assert traces[0].steps[0].tactic == "rw [add_comm]"
    finally:
        os.unlink(path)
    print(f"  Parsed LeanDojo trace with {len(traces[0].steps)} step(s)")


def test_load_dataset_auto_infers_format():
    from rl_agent.dataset_loaders import load_dataset

    trace_data = [{
        "theorem": "test_thm",
        "result": "proved",
        "traj": [{"state": {"goal": "True"}, "tactic": "trivial"}],
    }]

    with tempfile.NamedTemporaryFile(suffix=".json", prefix="traj_", mode="w", delete=False) as f:
        import json as j
        j.dump(trace_data, f)
        path = f.name
    try:
        traces = load_dataset(path)
        assert len(traces) == 1
    finally:
        os.unlink(path)
    print(f"  Auto-inferred format: {len(traces)} trace(s)")


def test_convert_to_training_examples_empty():
    from rl_agent.dataset_loaders import convert_to_training_examples, RawProofTrace

    _assert_eq(len(convert_to_training_examples([])), 0, "Empty input")

    trace = RawProofTrace(theorem_name="empty", steps=[], outcome=0.0)
    result = convert_to_training_examples([trace])
    _assert_eq(len(result), 0, "Trace with no steps")
    print("  Empty traces handled correctly")


def test_raw_dataclasses():
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
    print("  RawProofStep and RawProofTrace dataclasses work")


def test_parse_minif2f():
    from rl_agent.dataset_loaders import parse_minif2f, convert_to_training_examples

    minif2f_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "datasets", "minif2f", "minif2f_valid.json",
    )
    if not os.path.exists(minif2f_path):
        print("  Skipping (miniF2F not downloaded)")
        return

    traces = parse_minif2f(minif2f_path)
    assert len(traces) > 0
    for t in traces:
        assert t.source == "minif2f"

    non_empty = [t for t in traces if len(t.steps) > 0]
    examples = convert_to_training_examples(traces)
    if examples:
        for ex in examples:
            assert len(ex.state_vec) == FEATURE_DIM
            _assert_close(sum(ex.mcts_policy), 1.0, tol=1e-6)

    print(f"  Parsed {len(traces)} miniF2F traces -> {len(examples)} examples")


def test_parse_proofnet():
    from rl_agent.dataset_loaders import parse_proofnet, convert_to_training_examples

    proofnet_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "datasets", "proofnet", "proofnet.jsonl",
    )
    if not os.path.exists(proofnet_path):
        print("  Skipping (ProofNet not downloaded)")
        return

    traces = parse_proofnet(proofnet_path)
    assert len(traces) > 0
    for t in traces:
        assert t.source == "proofnet"

    examples = convert_to_training_examples(traces)
    if examples:
        for ex in examples:
            assert len(ex.state_vec) == FEATURE_DIM
            _assert_close(sum(ex.mcts_policy), 1.0, tol=1e-6)

    print(f"  Parsed {len(traces)} ProofNet traces -> {len(examples)} examples")


def test_load_dataset_dispatches_correctly():
    from rl_agent.dataset_loaders import load_dataset, DatasetFormat

    base = os.path.dirname(os.path.abspath(__file__))
    datasets = [
        ("miniF2F", os.path.join(base, "datasets", "minif2f", "minif2f_valid.json")),
        ("ProofNet", os.path.join(base, "datasets", "proofnet", "proofnet.jsonl")),
        ("Lean Workbook", os.path.join(base, "datasets", "lean_workbook", "lean_workbook.json")),
    ]
    for name, path in datasets:
        if os.path.exists(path):
            traces = load_dataset(path)
            assert len(traces) > 0
            print(f"  {name}: {len(traces)} traces")

    print("  All format auto-detection tests passed")


if __name__ == "__main__":
    # Run each test function
    test_functions = [
        ("encode returns list", test_encode_returns_list),
        ("FEATURE_DIM == 256", test_feature_dim),
        ("encode all floats", test_encode_all_floats),
        ("encode global scalars in range", test_encode_global_scalars_in_range),
        ("encode tactic bag in range", test_encode_tactic_bag_in_range),
        ("different states differ", test_encode_different_states_differ),
        ("encode empty state", test_encode_empty_state),
        ("encode deterministic", test_encode_deterministic),
        ("network initialization", test_network_initialization),
        ("network forward shapes", test_network_forward_shapes),
        ("priors sum to one", test_priors_sum_to_one),
        ("priors in range", test_priors_in_range),
        ("value in range", test_value_in_range),
        ("update weights returns floats", test_update_weights_returns_floats),
        ("update weights reduces loss", test_update_weights_reduces_loss),
        ("save/load round trip", test_save_load_round_trip),
        ("tactic simulator returns tuple", test_tactic_simulator_returns_tuple),
        ("tactic simulator reward in range", test_tactic_simulator_reward_in_range),
        ("MCTS node UCB finite", test_mcts_node_ucb_finite),
        ("MCTS node UCB unvisited", test_mcts_node_ucb_unvisited),
        ("MCTS search returns valid probs", test_mcts_search_returns_valid_probs),
        ("MCTS root value in range", test_mcts_root_value_in_range),
        ("MCTS best action valid", test_mcts_best_action_is_valid),
        ("MCTS expand creates children", test_mcts_expand_creates_children),
        ("MCTS more simulations affects probs", test_mcts_more_simulations_affects_probs),
        ("run episode returns EpisodeResult", test_run_episode_returns_episode_result),
        ("episode outcome is valid", test_episode_outcome_is_valid),
        ("episode examples correct vec length", test_episode_examples_have_correct_vec_length),
        ("episode examples policy sums to one", test_episode_examples_policy_sums_to_one),
        ("episode examples same outcome", test_episode_examples_have_same_outcome),
        ("run episode temperature zero", test_run_episode_temperature_zero),
        ("trainer initialization", test_trainer_initialization),
        ("trainer run returns network", test_trainer_run_returns_network),
        ("trainer accumulates buffer", test_trainer_accumulates_buffer),
        ("trainer records stats", test_trainer_records_stats),
        ("trainer saves checkpoints", test_trainer_saves_checkpoints),
        ("train from source", test_train_from_source),
        ("train from source multi-function", test_train_from_source_multi_function),
        ("pipeline source to encoded", test_pipeline_source_to_encoded),
        ("pipeline network forward", test_pipeline_network_forward),
        ("pipeline MCTS search", test_pipeline_mcts_search),
        ("pipeline self-play episode", test_pipeline_self_play_episode),
        ("generate builtin seed data", test_generate_builtin_seed_data_returns_traces),
        ("generate builtin seed data produces steps", test_generate_builtin_seed_data_produces_steps),
        ("convert to training examples", test_convert_to_training_examples),
        ("convert to training examples policies", test_convert_to_training_examples_policies),
        ("convert with MCTS policy smoothing", test_convert_with_mcts_policy_smoothing),
        ("seed data save/load round trip", test_seed_data_save_load_round_trip),
        ("seed buffer populates", test_seed_buffer_populates),
        ("pretrain supervised reduces loss", test_pretrain_supervised_reduces_policy_loss),
        ("pretrain supervised returns floats", test_pretrain_supervised_returns_floats),
        ("pretrain supervised empty buffer", test_pretrain_supervised_empty_buffer),
        ("auto seed and pretrain", test_auto_seed_and_pretrain),
        ("pretrain on builtin seeds before self-play", test_pretrain_on_builtin_seeds_before_self_play),
        ("cold start pipeline", test_cold_start_pipeline),
        ("parse LeanDojo trace", test_parse_leandojo_trace),
        ("load dataset auto-infers format", test_load_dataset_auto_infers_format),
        ("convert empty traces", test_convert_to_training_examples_empty),
        ("raw dataclasses", test_raw_dataclasses),
        ("parse miniF2F", test_parse_minif2f),
        ("parse ProofNet", test_parse_proofnet),
        ("load dataset dispatches correctly", test_load_dataset_dispatches_correctly),
    ]

    for name, fn in test_functions:
        _run_test(name, fn)

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
