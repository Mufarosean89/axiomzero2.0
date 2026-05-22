"""
Axiom Zero - Self-Play Training Loop
Phase 3: RL Agent

Implements the AlphaZero self-play cycle:

  1. Generate game: run MCTS to build a proof attempt episode.
  2. Collect training examples: (state_vec, mcts_policy, outcome).
  3. Train the network on a replay buffer of recent examples.
  4. Checkpoint the network.

Episode structure
-----------------
An episode starts with a ProofState (from Phase 1→2 bridge) and ends when:
  - Proof is complete (reward +1)  ✓
  - Max depth reached (reward  0)  ✗
  - Tactic failure with no recovery (reward -1) ✗

Training targets
----------------
  policy target  : MCTS visit-count policy π̂ (length = NUM_ACTIONS)
  value  target  : actual game outcome z ∈ {-1, 0, +1}
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from proof_engine import ProofState, build_proof_state, build_proof_state_collection
from spec_ingestion import extract_specs
from ast_extractor import parse_source, normalize
from abstract_interpreter import analyze

from .encoder import encode, FEATURE_DIM
from .networks import PolicyValueNet
from .mcts import MCTS, TacticSimulator, MAX_DEPTH


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TrainingExample:
    """A single labelled training example from self-play."""
    state_vec: List[float]          # encoded observation
    mcts_policy: List[float]        # MCTS-derived action probabilities
    outcome: float                  # game result: +1 / 0 / -1


@dataclass
class EpisodeResult:
    """Summary of a single self-play episode."""
    theorem_name: str
    success: bool
    num_steps: int
    tactics_applied: List[str]
    outcome: float                  # +1, 0, -1
    examples: List[TrainingExample] = field(default_factory=list)


# ── Self-play episode runner ─────────────────────────────────────────────────

def run_episode(
    initial_state: ProofState,
    net: PolicyValueNet,
    simulator: TacticSimulator | None = None,
    num_simulations: int = 50,
    temperature: float = 1.0,
    verbose: bool = False,
) -> EpisodeResult:
    """
    Run a single self-play episode.

    Args:
        initial_state   : Starting proof state (from Phase 2 bridge).
        net             : Current policy/value network.
        simulator       : Tactic simulator (or real LeanEnv wrapper).
        num_simulations : MCTS simulations per move.
        temperature     : >1 = more exploration; <1 = more greedy; 0 = argmax
        verbose         : Print progress.

    Returns:
        EpisodeResult with collected training examples and outcome.
    """
    mcts = MCTS(net, simulator=simulator, num_simulations=num_simulations)
    state = initial_state
    examples: List[TrainingExample] = []
    tactics_applied: List[str] = []

    if verbose:
        print(f"\n  Episode: {state.theorem_name}")

    for step in range(MAX_DEPTH):
        if state.is_complete:
            outcome = 1.0
            break

        if not state.open_goals:
            outcome = 0.0
            break

        # Run MCTS to get action probabilities
        action_probs, _ = mcts.search(state)

        # Encode state BEFORE action
        obs = state.to_observation()
        state_vec = encode(obs)
        examples.append(TrainingExample(
            state_vec=state_vec,
            mcts_policy=list(action_probs),
            outcome=0.0,  # filled in after episode ends
        ))

        # Select action
        tactic = _sample_action(action_probs, mcts._tactic_list, temperature)
        tactics_applied.append(tactic)

        if verbose:
            print(f"    Step {step + 1}: {tactic}")

        # Apply tactic via simulator
        sim = simulator or TacticSimulator()
        success, new_state, reward = sim.apply(state, tactic)

        if not success:
            outcome = -1.0
            break

        state = new_state
        if state.is_complete:
            outcome = 1.0
            break
    else:
        outcome = 0.0  # max depth reached without proof

    # Back-fill outcome into all examples
    for ex in examples:
        ex.outcome = outcome

    if verbose:
        status = "✓ proved" if outcome > 0 else ("✗ failed" if outcome < 0 else "— timeout")
        print(f"    Result: {status} ({len(examples)} steps)")

    return EpisodeResult(
        theorem_name=state.theorem_name,
        success=outcome > 0,
        num_steps=len(tactics_applied),
        tactics_applied=tactics_applied,
        outcome=outcome,
        examples=examples,
    )


def _sample_action(
    action_probs: List[float],
    tactic_list: List[str],
    temperature: float,
) -> str:
    """Sample a tactic from the MCTS policy with optional temperature."""
    if temperature == 0 or all(p == 0 for p in action_probs):
        # Greedy
        best = max(range(len(action_probs)), key=lambda i: action_probs[i])
        return tactic_list[best]

    # Raise probs to 1/temperature and renormalize
    probs = [p ** (1.0 / temperature) for p in action_probs]
    total = sum(probs) or 1.0
    probs = [p / total for p in probs]

    # Weighted random sample
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return tactic_list[i]
    return tactic_list[-1]


# ── Training loop ────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Configuration for the self-play training loop."""
    num_iterations: int = 10
    episodes_per_iteration: int = 5
    mcts_simulations: int = 50
    replay_buffer_size: int = 1000
    batch_size: int = 32
    learning_rate: float = 5e-4
    checkpoint_dir: str = "checkpoints"
    temperature_threshold: int = 10   # use high temp for first N steps, then greedy
    verbose: bool = False


class SelfPlayTrainer:
    """
    Orchestrates the AlphaZero self-play training loop.

    Usage
    -----
        trainer = SelfPlayTrainer(proof_states, config)
        trainer.run()
    """

    def __init__(
        self,
        proof_states: List[ProofState],
        config: TrainingConfig | None = None,
        net: PolicyValueNet | None = None,
        simulator: TacticSimulator | None = None,
    ) -> None:
        self.proof_states = proof_states
        self.config = config or TrainingConfig()
        self.net = net or PolicyValueNet()
        self.simulator = simulator or TacticSimulator()
        self.replay_buffer: deque[TrainingExample] = deque(
            maxlen=self.config.replay_buffer_size
        )
        self.stats: List[Dict[str, Any]] = []

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def run(self) -> PolicyValueNet:
        """
        Run the full self-play training loop.

        Returns the trained PolicyValueNet.
        """
        cfg = self.config
        print(f"\n{'='*60}")
        print(f"Axiom Zero — Phase 3 Self-Play Training")
        print(f"  Proof states   : {len(self.proof_states)}")
        print(f"  Iterations     : {cfg.num_iterations}")
        print(f"  Episodes/iter  : {cfg.episodes_per_iteration}")
        print(f"  MCTS sims      : {cfg.mcts_simulations}")
        print(f"{'='*60}\n")

        for iteration in range(1, cfg.num_iterations + 1):
            iter_start = time.time()
            print(f"Iteration {iteration}/{cfg.num_iterations}")

            # ── Self-play phase ────────────────────────────────────────
            wins = 0
            for ep_idx in range(cfg.episodes_per_iteration):
                # Pick a random proof state
                initial = random.choice(self.proof_states)
                temperature = 1.0 if ep_idx < cfg.temperature_threshold else 0.1

                result = run_episode(
                    initial,
                    self.net,
                    simulator=self.simulator,
                    num_simulations=cfg.mcts_simulations,
                    temperature=temperature,
                    verbose=cfg.verbose,
                )

                self.replay_buffer.extend(result.examples)
                if result.success:
                    wins += 1

            win_rate = wins / cfg.episodes_per_iteration
            print(f"  Win rate: {win_rate:.1%}  ({wins}/{cfg.episodes_per_iteration})")

            # ── Training phase ─────────────────────────────────────────
            pi_loss, v_loss = self._train_step()
            print(f"  Policy loss: {pi_loss:.4f}  Value loss: {v_loss:.4f}")
            print(f"  Buffer size: {len(self.replay_buffer)}")

            elapsed = time.time() - iter_start
            print(f"  Time: {elapsed:.1f}s\n")

            self.stats.append({
                "iteration": iteration,
                "win_rate": win_rate,
                "policy_loss": pi_loss,
                "value_loss": v_loss,
                "buffer_size": len(self.replay_buffer),
            })

            # ── Checkpoint ─────────────────────────────────────────────
            ckpt_path = os.path.join(
                cfg.checkpoint_dir, f"net_iter_{iteration:03d}.json"
            )
            self.net.save(ckpt_path)

        print("Training complete.")
        return self.net

    def _train_step(self) -> Tuple[float, float]:
        """Sample a mini-batch from the replay buffer and update the network."""
        if len(self.replay_buffer) < self.config.batch_size:
            # Not enough data yet — skip
            return 0.0, 0.0

        batch: List[TrainingExample] = random.sample(
            list(self.replay_buffer), self.config.batch_size
        )

        state_vecs = [ex.state_vec for ex in batch]
        policies = [ex.mcts_policy for ex in batch]
        values = [ex.outcome for ex in batch]

        return self.net.update_weights(
            target_policies=policies,
            target_values=values,
            state_vecs=state_vecs,
            lr=self.config.learning_rate,
        )

    def save_stats(self, path: str = "training_stats.json") -> None:
        """Write training statistics to JSON."""
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"Stats saved to {path}")


# ── Convenience: train from Python source ─────────────────────────────────────

def train_from_source(
    source: str,
    module_name: str = "target",
    config: TrainingConfig | None = None,
) -> Tuple[PolicyValueNet, List[EpisodeResult]]:
    """
    High-level helper: parse Python source → extract specs → train the agent.

    Args:
        source      : Python source string with @requires/@ensures decorators.
        module_name : Module name for the IR.
        config      : Training configuration.

    Returns:
        (trained_net, episode_results_from_last_iteration)
    """
    ir = parse_source(source, module_name)
    ir = normalize(ir)
    state = analyze(ir)
    specs = extract_specs(ir, state)

    if specs.total_count == 0:
        raise ValueError("No proof obligations found in source.")

    proof_states = build_proof_state_collection(specs)
    print(f"Found {specs.total_count} proof obligation(s) → {len(proof_states)} proof state(s).")

    trainer = SelfPlayTrainer(proof_states, config or TrainingConfig())
    net = trainer.run()
    return net, trainer.stats
