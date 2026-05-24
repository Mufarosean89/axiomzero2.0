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

    Cold-Start / Dataset Loaders
    ----------------------------
    parse_leandojo_trace(path)          -> List[RawProofTrace]
    parse_minif2f(path)                 -> List[RawProofTrace]
    parse_proofnet(path)                -> List[RawProofTrace]
    parse_lean_workbook(path)           -> List[RawProofTrace]
    load_dataset(path)                  -> List[RawProofTrace]
    generate_builtin_seed_data(...)     -> List[RawProofTrace]
    convert_to_training_examples(...)   -> List[TrainingExample]
    convert_with_mcts_policy(...)       -> List[TrainingExample]
    save_seed_data(examples, path)
    load_seed_data(path)                -> List[TrainingExample]
    DatasetFormat, RawProofStep, RawProofTrace

    Self-Play
    ---------
    SelfPlayTrainer(proof_states, config)
    trainer.run()                       -> PolicyValueNet
    trainer.seed_buffer(examples)       -> int
    trainer.pretrain_supervised(...)    -> (pi_loss, v_loss)
    run_episode(state, net)             -> EpisodeResult
    train_from_source(source)           -> (net, stats)
    TrainingConfig(...)
"""

from .encoder import encode, FEATURE_DIM
from .networks import PolicyValueNet
from .mcts import MCTS, TacticSimulator, RealTacticSimulator, MCTSNode
from .self_play import (
    SelfPlayTrainer,
    TrainingConfig,
    TrainingExample,
    EpisodeResult,
    run_episode,
    train_from_source,
)
from .dataset_loaders import (
    parse_leandojo_trace,
    parse_minif2f,
    parse_proofnet,
    parse_lean_workbook,
    load_dataset,
    generate_builtin_seed_data,
    convert_to_training_examples,
    convert_with_mcts_policy,
    save_seed_data,
    load_seed_data,
    DatasetFormat,
    RawProofStep,
    RawProofTrace,
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
    "RealTacticSimulator",
    "MCTSNode",
    # Dataset loaders
    "parse_leandojo_trace",
    "parse_minif2f",
    "parse_proofnet",
    "parse_lean_workbook",
    "load_dataset",
    "generate_builtin_seed_data",
    "convert_to_training_examples",
    "convert_with_mcts_policy",
    "save_seed_data",
    "load_seed_data",
    "DatasetFormat",
    "RawProofStep",
    "RawProofTrace",
    # Self-play
    "SelfPlayTrainer",
    "TrainingConfig",
    "TrainingExample",
    "EpisodeResult",
    "run_episode",
    "train_from_source",
]
