"""RL Agent module: encoder, networks, MCTS, dataset loaders, and self-play training."""

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
