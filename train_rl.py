#!/usr/bin/env python3
"""RL Agent Training Script — loads datasets, seeds replay buffer, runs AlphaZero self-play."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from proof_engine import ProofState, build_proof_state_collection
from proof_engine.proof_state import Goal, GoalStatus, Hypothesis
from ast_extractor import parse_source, normalize
from abstract_interpreter import analyze
from spec_ingestion import extract_specs
from rl_agent import (
    PolicyValueNet, SelfPlayTrainer, TrainingConfig,
    TrainingExample, RealTacticSimulator, run_episode, encode, FEATURE_DIM,
)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Loading
# ═══════════════════════════════════════════════════════════════════════════


def _load_dataset(dataset_dir: Path, filename: str, parser_fn_name: str, verbose: bool) -> List[TrainingExample]:
    from rl_agent.dataset_loaders import convert_to_training_examples
    parser_fn = __import__("rl_agent.dataset_loaders", fromlist=[parser_fn_name]).__dict__[parser_fn_name]

    path = dataset_dir / filename
    if not path.exists():
        if verbose:
            print(f"  {parser_fn_name}: file not found at {path}")
        return []

    traces = parser_fn(str(path))
    traces = [t for t in traces if len(t.steps) > 0]
    if verbose:
        print(f"  {parser_fn_name}: {len(traces)} non-empty traces")

    examples = convert_to_training_examples(traces)
    if verbose:
        print(f"  -> {len(examples)} training examples")
    return examples


def load_minif2f(dataset_dir: Path, verbose: bool = True) -> List[TrainingExample]:
    examples = _load_dataset(dataset_dir / "minif2f", "minif2f_valid.json", "parse_minif2f", verbose)
    examples += _load_dataset(dataset_dir / "minif2f", "minif2f_test.json", "parse_minif2f", verbose)
    return examples


def load_proofnet(dataset_dir: Path, verbose: bool = True) -> List[TrainingExample]:
    return _load_dataset(dataset_dir / "proofnet", "proofnet.jsonl", "parse_proofnet", verbose)


def load_lean_workbook(dataset_dir: Path, verbose: bool = True) -> List[TrainingExample]:
    return _load_dataset(dataset_dir / "lean_workbook", "lean_workbook.json", "parse_lean_workbook", verbose)


def load_builtin_seed(
    proof_states: List[ProofState],
    max_traces: int = 30,
    max_steps: int = 10,
    verbose: bool = True,
) -> List[TrainingExample]:
    from rl_agent.dataset_loaders import generate_builtin_seed_data, convert_to_training_examples

    raw_traces = generate_builtin_seed_data(
        proof_states=proof_states, max_traces=max_traces,
        max_steps_per_trace=max_steps, temperature=0.0, verbose=verbose,
    )
    raw_traces = [t for t in raw_traces if len(t.steps) > 0]
    if verbose:
        print(f"  Built-in: {len(raw_traces)} non-empty proof traces")

    examples = convert_to_training_examples(raw_traces)
    if verbose:
        print(f"  -> {len(examples)} training examples")
    return examples


# ═══════════════════════════════════════════════════════════════════════════
# Proof State Loading
# ═══════════════════════════════════════════════════════════════════════════


def load_proof_states_from_benchmarks(verbose: bool = True) -> List[ProofState]:
    from lean_compiler.benchmark_suite import BenchmarkSuite

    suite = BenchmarkSuite()
    all_states: List[ProofState] = []
    failures = 0

    for problem in suite.problems:
        try:
            ir = parse_source(problem.source, f"bench_{problem.name}")
            ir = normalize(ir)
            abstract = analyze(ir)
            specs = extract_specs(ir, abstract)
            states = build_proof_state_collection(specs)
            all_states.extend(states)
            if verbose:
                for s in states:
                    print(f"  [OK] {s.theorem_name}: {s.theorem_type[:60]}...")
        except Exception as e:
            failures += 1
            if verbose:
                print(f"  [FAIL] {problem.name}: {e}")

    if verbose:
        print(f"\n  Loaded {len(all_states)} proof states across {len(suite.problems)} benchmarks")
        if failures:
            print(f"  ({failures} benchmarks failed to parse)")
    return all_states


def load_proof_states_from_source(sources: Dict[str, str], verbose: bool = True) -> List[ProofState]:
    all_states: List[ProofState] = []
    for name, source in sources.items():
        try:
            ir = parse_source(source, name)
            ir = normalize(ir)
            abstract = analyze(ir)
            specs = extract_specs(ir, abstract)
            states = build_proof_state_collection(specs)
            all_states.extend(states)
            if verbose:
                for s in states:
                    print(f"  [OK] {s.theorem_name}")
        except Exception as e:
            if verbose:
                print(f"  [FAIL] {name}: {e}")
    return all_states


# ═══════════════════════════════════════════════════════════════════════════
# Training Orchestration
# ═══════════════════════════════════════════════════════════════════════════


def run_training(
    proof_states: List[ProofState],
    seed_examples: List[TrainingExample],
    config: TrainingConfig,
    output_dir: Path,
    verbose: bool = True,
    simulator: Optional[RealTacticSimulator] = None,
) -> Tuple[PolicyValueNet, SelfPlayTrainer]:
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Starting RL Agent Training")
        print(f"  Proof states:       {len(proof_states)}")
        print(f"  Seed examples:      {len(seed_examples)}")
        print(f"  Iterations:         {config.num_iterations}")
        print(f"  Episodes/iteration: {config.episodes_per_iteration}")
        print(f"  MCTS simulations:   {config.mcts_simulations}")
        print(f"  Output:             {output_dir}")
        print(f"{'=' * 70}")

    trainer = SelfPlayTrainer(proof_states=proof_states, config=config, simulator=simulator)

    if seed_examples:
        seeded = trainer.seed_buffer(seed_examples)
        if verbose:
            print(f"  Buffer seeded with {seeded} examples\n")

    if config.pretrain_epochs > 0 and len(trainer.replay_buffer) > 0:
        trainer.pretrain_supervised(
            num_epochs=config.pretrain_epochs,
            batch_size=config.pretrain_batch_size,
            lr=config.pretrain_learning_rate,
        )
    elif verbose:
        print("  Skipping supervised pre-training (no seed data or pretrain_epochs=0)")

    net = trainer.run()

    output_dir.mkdir(parents=True, exist_ok=True)
    net.save(str(output_dir / "net_final.json"))
    trainer.save_stats(str(output_dir / "training_stats.json"))

    summary = {
        "proof_states_count": len(proof_states),
        "seed_examples_count": len(seed_examples),
        "iterations": config.num_iterations,
        "episodes_per_iteration": config.episodes_per_iteration,
        "mcts_simulations": config.mcts_simulations,
        "pretrain_epochs": config.pretrain_epochs,
        "final_buffer_size": len(trainer.replay_buffer),
        "stats": trainer.stats,
    }
    with open(str(output_dir / "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return net, trainer


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="RL Agent Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--datasets", type=str, default="all",
        help="Comma-separated datasets (all, builtin, minif2f, proofnet, workbook)")
    parser.add_argument("--datasets-dir", type=str, default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--mcts-simulations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--quick", action="store_true",
        help="Quick smoke test: 2 iter, 2 eps, 10 MCTS, 1 pretrain")
    parser.add_argument("--max-traces", type=int, default=500)
    parser.add_argument("--real-lean", action="store_true",
        help="Use real Lean 4 server instead of heuristic simulator")
    parser.add_argument("--lean-path", type=str, default="lean")
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    output_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  RL Agent Training")
    print(f"  Datasets: {datasets_dir}")
    print(f"  Output:   {output_dir}")
    print(f"{'=' * 70}")

    if args.quick:
        args.iterations = 2
        args.episodes = 2
        args.mcts_simulations = 10
        args.pretrain_epochs = 1
        args.batch_size = 8
        print("  [QUICK] Smoke test mode enabled")

    # Step 1: Load proof states
    print(f"\n-- Step 1: Loading Proof States -------------------------------------")
    proof_states = load_proof_states_from_benchmarks(verbose=args.verbose)

    if len(proof_states) == 0:
        print("  No proof states loaded. Creating synthetic fallback...")
        g = Goal(id="g0", type="x > 0 -> x + 1 > 0", hypotheses=[Hypothesis(name="x", type="ℤ")])
        proof_states = [ProofState(theorem_name="minimal_example", theorem_type="x > 0 -> x + 1 > 0", goals=[g])]

    # Step 2: Load datasets
    print(f"\n-- Step 2: Loading Datasets ----------------------------------------")
    dataset_keys = ["builtin", "minif2f", "proofnet", "workbook"] if args.datasets == "all" else [d.strip().lower() for d in args.datasets.split(",")]

    loaders = {
        "builtin": lambda: load_builtin_seed(proof_states, max_traces=min(args.max_traces, 50), max_steps=10, verbose=args.verbose),
        "minif2f": lambda: load_minif2f(datasets_dir / "minif2f", verbose=args.verbose),
        "proofnet": lambda: load_proofnet(datasets_dir / "proofnet", verbose=args.verbose),
        "workbook": lambda: load_lean_workbook(datasets_dir / "lean_workbook", verbose=args.verbose),
    }

    all_examples: List[TrainingExample] = []
    for key in dataset_keys:
        if key in loaders:
            print(f"  Loading {key}...")
            all_examples.extend(loaders[key]())

    print(f"\n  Total training examples: {len(all_examples)}")
    if len(all_examples) == 0:
        print("  No examples loaded — self-play will generate its own data.")

    # Step 3: Configure training
    print(f"\n-- Step 3: Configuring Training -------------------------------------")
    config = TrainingConfig(
        num_iterations=args.iterations, episodes_per_iteration=args.episodes,
        mcts_simulations=args.mcts_simulations, batch_size=args.batch_size,
        checkpoint_dir=str(output_dir / "checkpoints"),
        pretrain_on_builtin=False, pretrain_epochs=args.pretrain_epochs,
        pretrain_batch_size=min(args.batch_size, 32), pretrain_learning_rate=1e-3,
        temperature_threshold=args.episodes,
    )

    print(f"  Iterations: {config.num_iterations}, Episodes: {config.episodes_per_iteration}")
    print(f"  MCTS: {config.mcts_simulations}, Batch: {config.batch_size}")
    print(f"  Pretrain: {config.pretrain_epochs} epochs")

    # Step 4: Run training
    print(f"\n-- Step 4: Running Training ----------------------------------------")
    lean_env = None
    start_time = time.time()

    try:
        sim: Optional[RealTacticSimulator] = None
        if args.real_lean:
            print("  Starting Lean 4 server...")
            from proof_engine.lean_env import LeanEnv
            from proof_engine.tactics import TacticExecutor
            lean_env = LeanEnv(lean_path=args.lean_path)
            lean_env.start()
            sim = RealTacticSimulator(lean_env, TacticExecutor(lean_env))
            print(f"  Lean 4 server started ({lean_env.lean_version})\n")

        net, trainer = run_training(
            proof_states=proof_states, seed_examples=all_examples,
            config=config, output_dir=output_dir,
            verbose=args.verbose, simulator=sim,
        )
    finally:
        if lean_env is not None:
            lean_env.stop()

    elapsed = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"  Training Complete!  {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Network: {output_dir / 'net_final.json'}")
    print(f"  Stats:   {output_dir / 'training_stats.json'}")
    print(f"{'=' * 70}")

    if trainer.stats:
        print(f"\n  Progress:")
        for stat in trainer.stats:
            phase = stat.get("phase", f"iter_{stat.get('iteration', '?')}")
            wr = stat.get("win_rate")
            pl = stat.get("policy_loss")
            vl = stat.get("value_loss")
            if wr is not None:
                print(f"    {phase:>20s}: win={wr:.1%}, pi_loss={pl:.4f}, v_loss={vl:.4f}, buf={stat.get('buffer_size')}")
            elif pl is not None:
                print(f"    {phase:>20s}: pi_loss={pl:.4f}, v_loss={vl:.4f}")


if __name__ == "__main__":
    main()
