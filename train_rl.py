#!/usr/bin/env python3
"""
Axiom Zero - RL Agent Training Script

Loads external Lean proof datasets (miniF2F, ProofNet, Lean Workbook),
prepares the Axiom Zero proof states from the benchmark suite, seeds the
replay buffer with dataset-derived training examples, runs supervised
pre-training, and then executes the AlphaZero self-play training loop.

Usage:
    python train_rl.py                                    # Use all datasets
    python train_rl.py --datasets minif2f,proofnet        # Subset
    python train_rl.py --iterations 20 --episodes 5       # Custom params
    python train_rl.py --datasets builtin                 # Built-in seed only
    python train_rl.py --checkpoint-dir ./my_checkpoints  # Custom output dir
    python train_rl.py --quick                            # Quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in the Python path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from proof_engine import ProofState, build_proof_state_collection
from proof_engine.proof_state import Goal, GoalStatus, Hypothesis
from ast_extractor import parse_source, normalize
from abstract_interpreter import analyze
from spec_ingestion import extract_specs
from rl_agent import (
    PolicyValueNet,
    SelfPlayTrainer,
    TrainingConfig,
    TrainingExample,
    RealTacticSimulator,
    run_episode,
    encode,
    FEATURE_DIM,
)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Loading
# ═══════════════════════════════════════════════════════════════════════════


def load_minif2f(dataset_dir: Path, verbose: bool = True) -> List[TrainingExample]:
    """Load miniF2F dataset and convert to training examples."""
    from rl_agent.dataset_loaders import parse_minif2f, convert_to_training_examples

    valid_path = dataset_dir / "minif2f_valid.json"
    test_path = dataset_dir / "minif2f_test.json"
    all_traces = []

    for path in [valid_path, test_path]:
        if path.exists():
            traces = parse_minif2f(str(path))
            # Filter to entries with actual proofs
            traces = [t for t in traces if len(t.steps) > 0]
            if verbose:
                print(f"  miniF2F ({path.name}): {len(traces)} non-empty proof traces")
            all_traces.extend(traces)

    examples = convert_to_training_examples(all_traces)
    if verbose:
        print(f"  -> {len(examples)} training examples")
    return examples


def load_proofnet(dataset_dir: Path, verbose: bool = True) -> List[TrainingExample]:
    """Load ProofNet dataset and convert to training examples."""
    from rl_agent.dataset_loaders import parse_proofnet, convert_to_training_examples

    path = dataset_dir / "proofnet.jsonl"
    if not path.exists():
        if verbose:
            print(f"  ProofNet: file not found at {path}")
        return []

    traces = parse_proofnet(str(path))
    # Filter to entries with actual proofs
    traces = [t for t in traces if len(t.steps) > 0]
    if verbose:
        print(f"  ProofNet: {len(traces)} non-empty proof traces")

    examples = convert_to_training_examples(traces)
    if verbose:
        print(f"  -> {len(examples)} training examples")
    return examples


def load_lean_workbook(dataset_dir: Path, verbose: bool = True) -> List[TrainingExample]:
    """Load Lean Workbook dataset and convert to training examples."""
    from rl_agent.dataset_loaders import parse_lean_workbook, convert_to_training_examples

    path = dataset_dir / "lean_workbook.json"
    if not path.exists():
        if verbose:
            print(f"  Lean Workbook: file not found at {path}")
        return []

    # Parse using streaming iterator (handles 94 MB files without MemoryError)
    # Process all entries since the user wants the full dataset
    if verbose:
        print(f"  Lean Workbook: reading all entries (streaming, may take a moment)...")
    traces = parse_lean_workbook(str(path), max_entries=None)

    # Filter to entries with actual steps (already filtered by parse, but double-check)
    traces = [t for t in traces if len(t.steps) > 0]
    if verbose:
        print(f"  Lean Workbook: {len(traces)} non-empty proof traces")

    examples = convert_to_training_examples(traces)
    if verbose:
        print(f"  -> {len(examples)} training examples")
    return examples


def load_builtin_seed(
    proof_states: List[ProofState],
    max_traces: int = 30,
    max_steps: int = 10,
    verbose: bool = True,
) -> List[TrainingExample]:
    """Generate seed data from built-in benchmark using heuristic tactic suggestions."""
    from rl_agent.dataset_loaders import generate_builtin_seed_data, convert_to_training_examples

    raw_traces = generate_builtin_seed_data(
        proof_states=proof_states,
        max_traces=max_traces,
        max_steps_per_trace=max_steps,
        temperature=0.0,  # greedy / deterministic
        verbose=verbose,
    )

    # Filter to traces with actual steps
    raw_traces = [t for t in raw_traces if len(t.steps) > 0]
    if verbose:
        print(f"  Built-in: {len(raw_traces)} non-empty proof traces")

    examples = convert_to_training_examples(raw_traces)
    if verbose:
        print(f"  -> {len(examples)} training examples")
    return examples


# ═══════════════════════════════════════════════════════════════════════════
# Proof State Loading (from benchmark suite)
# ═══════════════════════════════════════════════════════════════════════════


def load_proof_states_from_benchmarks(verbose: bool = True) -> List[ProofState]:
    """Generate ProofState objects from the built-in benchmark suite."""
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
                    if verbose:
                        print(f"  [OK] {s.theorem_name}: {s.theorem_type[:60]}...")
        except Exception as e:
            failures += 1
            if verbose:
                print(f"  [FAIL] {problem.name}: {e}")
            continue

    if verbose:
        print(f"\n  Loaded {len(all_states)} proof states across {len(suite.problems)} benchmarks")
        if failures:
            print(f"  ({failures} benchmarks failed to parse)")

    return all_states


def load_proof_states_from_source(sources: Dict[str, str], verbose: bool = True) -> List[ProofState]:
    """Generate ProofState objects from custom Python source strings."""
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
    """
    Run the full training pipeline:
      1. Seed the replay buffer with dataset/builtin examples.
      2. Run supervised behavioural cloning pre-training.
      3. Run AlphaZero self-play training.
      4. Save checkpoints and stats.

    Returns:
        (trained_network, trainer_with_stats)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Starting RL Agent Training")
        print(f"  Proof states:       {len(proof_states)}")
        print(f"  Seed examples:      {len(seed_examples)}")
        print(f"  Training iterations: {config.num_iterations}")
        print(f"  Episodes/iteration: {config.episodes_per_iteration}")
        print(f"  MCTS simulations:   {config.mcts_simulations}")
        print(f"  Output directory:   {output_dir}")
        print(f"{'='*70}")

    # Create the trainer
    trainer = SelfPlayTrainer(
        proof_states=proof_states,
        config=config,
        simulator=simulator,
    )

    # ── Step 1: Seed the replay buffer ───────────────────────────────────
    if seed_examples:
        seeded = trainer.seed_buffer(seed_examples)
        if verbose:
            print(f"  Buffer seeded with {seeded} examples\n")

    # ── Step 2: Supervised pre-training ──────────────────────────────────
    if config.pretrain_epochs > 0 and len(trainer.replay_buffer) > 0:
        trainer.pretrain_supervised(
            num_epochs=config.pretrain_epochs,
            batch_size=config.pretrain_batch_size,
            lr=config.pretrain_learning_rate,
        )
    else:
        if verbose:
            print("  Skipping supervised pre-training (no seed data or pretrain_epochs=0)")

    # ── Step 3: Self-play training ───────────────────────────────────────
    net = trainer.run()

    # ── Step 4: Save results ─────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the final network
    net_path = output_dir / "net_final.json"
    net.save(str(net_path))
    if verbose:
        print(f"  Final network saved to {net_path}")

    # Save training stats
    stats_path = output_dir / "training_stats.json"
    trainer.save_stats(str(stats_path))

    # Save a summary file
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
    summary_path = output_dir / "training_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"  Training summary saved to {summary_path}")

    return net, trainer


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Axiom Zero - RL Agent Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_rl.py                                         # Heuristic simulator (all datasets)
  python train_rl.py --datasets minif2f,proofnet              # Subset of datasets
  python train_rl.py --datasets builtin                       # Built-in seed only
  python train_rl.py --quick                                  # Quick heuristic smoke test
  python train_rl.py --real-lean                              # Real Lean 4 server (all datasets)
  python train_rl.py --real-lean --lean-path /path/to/lean    # Custom Lean path
  python train_rl.py --real-lean --quick                      # Quick smoke test with real Lean
  python train_rl.py --checkpoint-dir ./my_ckpts              # Custom output dir
        """,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated datasets to use (all, builtin, minif2f, proofnet, workbook). "
             "Default: all available datasets + built-in",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets"),
        help="Path to the datasets directory (default: ./datasets)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of self-play iterations (default: 10)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Episodes per iteration (default: 5)",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=50,
        help="MCTS simulations per move (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=5,
        help="Supervised pre-training epochs (default: 5). Set to 0 to skip.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints"),
        help="Directory for checkpoints and output (default: ./checkpoints)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test mode: 2 iterations, 2 episodes, 10 MCTS sims, 1 pretrain epoch",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=500,
        help="Maximum number of proof traces to load per dataset (default: 500)",
    )
    parser.add_argument(
        "--real-lean",
        action="store_true",
        help="Use a real Lean 4 server instead of the heuristic TacticSimulator",
    )
    parser.add_argument(
        "--lean-path",
        type=str,
        default="lean",
        help="Path to the Lean 4 executable (default: lean, uses PATH)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )

    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────
    datasets_dir = Path(args.datasets_dir)
    output_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Axiom Zero -- RL Agent Training")
    print(f"  Datasets dir: {datasets_dir}")
    print(f"  Output dir:   {output_dir}")
    print(f"{'='*70}")

    # ── Quick mode ──────────────────────────────────────────────────────
    if args.quick:
        args.iterations = 2
        args.episodes = 2
        args.mcts_simulations = 10
        args.pretrain_epochs = 1
        args.batch_size = 8
        if args.datasets == "quick":
            args.datasets = "minif2f"
        print("  [QUICK] Quick smoke test mode enabled")
        print(f"     Iterations: {args.iterations}, Episodes: {args.episodes}, "
              f"MCTS: {args.mcts_simulations}, Pretrain: {args.pretrain_epochs}")

    # ── Step 1: Load proof states from benchmarks ──────────────────────
    print(f"\n-- Step 1: Loading Proof States from Benchmarks -------------------")
    proof_states = load_proof_states_from_benchmarks(verbose=args.verbose)

    if len(proof_states) == 0:
        print("  ! No proof states loaded. Training cannot proceed.")
        print("  Creating minimal synthetic proof states for demonstration...")
        # Create a minimal proof state as fallback
        g = Goal(id="g0", type="x > 0 -> x + 1 > 0", hypotheses=[
            Hypothesis(name="x", type="ℤ")
        ])
        ps = ProofState(
            theorem_name="minimal_example",
            theorem_type="x > 0 -> x + 1 > 0",
            goals=[g],
        )
        proof_states = [ps]

    # ── Step 2: Determine which datasets to load ───────────────────────
    if args.datasets == "all":
        dataset_keys = ["builtin", "minif2f", "proofnet", "workbook"]
    else:
        dataset_keys = [d.strip().lower() for d in args.datasets.split(",")]

    # ── Step 3: Load and convert datasets ──────────────────────────────
    print(f"\n-- Step 2: Loading Datasets ----------------------------------------")
    all_examples: List[TrainingExample] = []

    if "builtin" in dataset_keys:
        print("  Loading built-in seed data...")
        examples = load_builtin_seed(
            proof_states,
            max_traces=min(args.max_traces, 50),
            max_steps=10,
            verbose=args.verbose,
        )
        all_examples.extend(examples)

    if "minif2f" in dataset_keys:
        print("  Loading miniF2F dataset...")
        minif2f_dir = datasets_dir / "minif2f"
        if minif2f_dir.exists():
            examples = load_minif2f(minif2f_dir, verbose=args.verbose)
            all_examples.extend(examples)
        else:
            print(f"  ! miniF2F directory not found at {minif2f_dir}")

    if "proofnet" in dataset_keys:
        print("  Loading ProofNet dataset...")
        proofnet_dir = datasets_dir / "proofnet"
        if proofnet_dir.exists():
            examples = load_proofnet(proofnet_dir, verbose=args.verbose)
            all_examples.extend(examples)
        else:
            print(f"  ! ProofNet directory not found at {proofnet_dir}")

    if "workbook" in dataset_keys:
        print("  Loading Lean Workbook dataset...")
        workbook_dir = datasets_dir / "lean_workbook"
        if workbook_dir.exists():
            examples = load_lean_workbook(workbook_dir, verbose=args.verbose)
            all_examples.extend(examples)
        else:
            print(f"  ! Lean Workbook directory not found at {workbook_dir}")

    print(f"\n  Total training examples collected: {len(all_examples)}")
    if len(all_examples) == 0:
        print("  ! No training examples loaded! Training will proceed with")
        print("     empty buffer -- self-play will generate its own data.")

    # ── Step 4: Configure training ──────────────────────────────────────
    print(f"\n-- Step 3: Configuring Training -------------------------------------")
    config = TrainingConfig(
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        mcts_simulations=args.mcts_simulations,
        batch_size=args.batch_size,
        checkpoint_dir=str(output_dir / "checkpoints"),
        pretrain_on_builtin=False,  # We handle seeding manually
        pretrain_epochs=args.pretrain_epochs,
        pretrain_batch_size=min(args.batch_size, 32),
        pretrain_learning_rate=1e-3,
        verbose=args.verbose,
        temperature_threshold=args.episodes,  # Decay temp after 1 episode
    )

    print(f"  Iterations:          {config.num_iterations}")
    print(f"  Episodes/iteration:  {config.episodes_per_iteration}")
    print(f"  MCTS simulations:    {config.mcts_simulations}")
    print(f"  Batch size:          {config.batch_size}")
    print(f"  Pretrain epochs:     {config.pretrain_epochs}")

    # ── Step 5: Run training ────────────────────────────────────────────
    print(f"\n-- Step 4: Running Training ----------------------------------------")

    lean_env = None
    start_time = time.time()
    try:
        # Optionally set up real Lean server
        sim: Optional[RealTacticSimulator] = None
        if args.real_lean:
            print("  Initializing real Lean 4 server...")
            from proof_engine.lean_env import LeanEnv
            from proof_engine.tactics import TacticExecutor

            lean_env = LeanEnv(lean_path=args.lean_path, verbose=args.verbose)
            lean_env.start()
            executor = TacticExecutor(lean_env)
            sim = RealTacticSimulator(lean_env, executor)
            print(f"  Lean 4 server started ({lean_env.lean_version})\n")

        net, trainer = run_training(
            proof_states=proof_states,
            seed_examples=all_examples,
            config=config,
            output_dir=output_dir,
            verbose=args.verbose,
            simulator=sim,
        )
    finally:
        if lean_env is not None:
            lean_env.stop()
            print("  Lean 4 server stopped.")

    elapsed = time.time() - start_time

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Training Complete!")
    print(f"  Elapsed time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Final network: {output_dir / 'net_final.json'}")
    print(f"  Stats:         {output_dir / 'training_stats.json'}")
    print(f"  Summary:       {output_dir / 'training_summary.json'}")
    print(f"{'='*70}")

    # Print final stats summary
    if trainer.stats:
        print(f"\n  Training Progress:")
        for stat in trainer.stats:
            phase = stat.get("phase", f"iter_{stat.get('iteration', '?')}")
            wr = stat.get("win_rate")
            pl = stat.get("policy_loss")
            vl = stat.get("value_loss")
            bs = stat.get("buffer_size")
            if wr is not None:
                print(f"    {phase:>20s}: win_rate={wr:.1%}, pi_loss={pl:.4f}, v_loss={vl:.4f}, buffer={bs}")
            else:
                print(f"    {phase:>20s}: pi_loss={pl:.4f}, v_loss={vl:.4f}" if pl is not None else f"    {phase}")


if __name__ == "__main__":
    main()
