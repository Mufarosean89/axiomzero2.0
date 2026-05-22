"""
Axiom Zero - RL Agent Module
Phase 3: AlphaZero-Style Proof Automation

Public API
----------
    Encoder
    -------
    encode(observation)         -> List[float]   (FEATURE_DIM = 256)
    FEATURE_DIM                 -> int

    Networks
    --------
    PolicyValueNet()            -> net
    net.forward(state_vec)      -> (priors: List[float], value: float)
    net.update_weights(...)     -> (policy_loss, value_loss)
    net.save(path)
    PolicyValueNet.load(path)   -> net

    MCTS
    ----
    MCTS(net, simulator, ...)
    mcts.search(state)          -> (action_probs, root_value)
    mcts.best_action(state)     -> tactic_str
    TacticSimulator()           -> simulator

    Self-Play
    ---------
    SelfPlayTrainer(proof_states, config)
    trainer.run()               -> PolicyValueNet
    run_episode(state, net)     -> EpisodeResult
    train_from_source(source)   -> (net, stats)
    TrainingConfig(...)
"""

from .encoder import encode, FEATURE_DIM
from .networks import PolicyValueNet
from .mcts import MCTS, TacticSimulator, MCTSNode
from .self_play import (
    SelfPlayTrainer,
    TrainingConfig,
    TrainingExample,
    EpisodeResult,
    run_episode,
    train_from_source,
)

__all__ = [
    # Encoder
    "encode",
    "FEATURE_DIM",
    # Networks
    "PolicyValueNet",
    # MCTS
    "MCTS",
    "TacticSimulator",
    "MCTSNode",
    # Self-play
    "SelfPlayTrainer",
    "TrainingConfig",
    "TrainingExample",
    "EpisodeResult",
    "run_episode",
    "train_from_source",
]
