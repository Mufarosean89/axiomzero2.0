"""
Axiom Zero - State Encoder
Phase 3: RL Agent

Converts a ProofState observation dict (from ProofState.to_observation()) into
a fixed-size numeric feature vector that the policy and value networks consume.

Design
------
We keep this pure Python / stdlib — no numpy, no torch — so the encoder can
run anywhere Phases 1 & 2 run.  The networks (networks.py) use the same
convention: vectors are plain List[float].

Feature layout (FEATURE_DIM = 256):
  [0:8]    Global proof scalars (normalised)
  [8:47]   Tactic-history bag (39-dim, count of each tactic used, capped at 5)
  [47:87]  Goal-type character n-gram hash (40-dim)
  [87:127] Hypothesis-name bag (40-dim)
  [127:167] Hypothesis-type n-gram hash (40-dim)
  [167:206] Second goal features (same 39-dim tactic bag for goal 2, if any)
  [206:256] Padding / future use (zeros)

Total: 256 floats.
"""

from __future__ import annotations

import math
import hashlib
from typing import Any, Dict, List, Optional

# Must match len(CORE_TACTICS) at import time — we compute lazily.
_TACTIC_LIST: Optional[List[str]] = None
FEATURE_DIM = 256


def _get_tactic_list() -> List[str]:
    global _TACTIC_LIST
    if _TACTIC_LIST is None:
        # Import here to avoid circular imports at module level.
        from proof_engine import CORE_TACTICS
        _TACTIC_LIST = list(CORE_TACTICS.keys())
    return _TACTIC_LIST


def _ngram_hash(text: str, dim: int, n: int = 3) -> List[float]:
    """
    Map a string to a `dim`-dimensional float vector via character n-gram hashing.

    Each n-gram is hashed (sha256) and its index mod `dim` is incremented.
    The result is L2-normalised so magnitudes are comparable across strings.
    """
    vec = [0.0] * dim
    text = text.lower().strip()
    if not text:
        return vec
    grams = [text[i:i + n] for i in range(len(text) - n + 1)] or [text]
    for gram in grams:
        digest = int(hashlib.sha256(gram.encode()).hexdigest(), 16)
        vec[digest % dim] += 1.0
    # L2 normalise
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _tactic_bag(tactic_names: List[str]) -> List[float]:
    """
    Return a bag-of-tactics vector (one entry per CORE_TACTIC).
    Counts are clipped to [0, 5] and then divided by 5 → [0, 1].
    """
    tactics = _get_tactic_list()
    bag = [0.0] * len(tactics)
    for name in tactic_names:
        if name in tactics:
            idx = tactics.index(name)
            bag[idx] = min(bag[idx] + 1.0, 5.0)
    return [v / 5.0 for v in bag]


def _encode_goal(goal_dict: Dict[str, Any]) -> List[float]:
    """
    Encode a single goal dict into a 79-float feature block.

    Layout:
        [0:40]  goal type n-gram hash
        [40:79] hypothesis bag (name 19-dim + type 20-dim, up to 3 hyps)
    """
    type_feat = _ngram_hash(goal_dict.get("type", ""), 40)

    hyp_feat = [0.0] * 39
    hyps = goal_dict.get("hypotheses", [])[:3]  # at most 3 hypotheses
    for i, hyp in enumerate(hyps):
        name_h = _ngram_hash(hyp.get("name", ""), 6)
        type_h = _ngram_hash(hyp.get("type", ""), 7)
        start = i * 13
        hyp_feat[start:start + 6] = name_h
        hyp_feat[start + 6:start + 13] = type_h

    return type_feat + hyp_feat  # 79 floats


def encode(observation: Dict[str, Any]) -> List[float]:
    """
    Convert a ProofState.to_observation() dict into a FEATURE_DIM float vector.

    Args:
        observation: Dict returned by ProofState.to_observation()

    Returns:
        List[float] of length FEATURE_DIM (256).
    """
    vec = [0.0] * FEATURE_DIM

    # ── Global scalars [0:8] ──────────────────────────────────────────────
    max_goals = 10.0
    max_tactics = 50.0
    max_depth = 20.0

    vec[0] = min(observation.get("num_open_goals", 0) / max_goals, 1.0)
    vec[1] = min(observation.get("num_total_goals", 0) / max_goals, 1.0)
    vec[2] = min(observation.get("num_tactics_applied", 0) / max_tactics, 1.0)
    vec[3] = min(observation.get("depth", 0) / max_depth, 1.0)
    vec[4] = 1.0 if observation.get("is_complete") else 0.0
    # vec[5:8]: theorem-name hash (3 floats) — makes each theorem's vector unique
    name_hash = _ngram_hash(observation.get("theorem", ""), 3)
    vec[5:8] = name_hash

    # ── Tactic history bag [8:47] ─────────────────────────────────────────
    history_names = [
        step["tactic"].split()[0]  # first token is the tactic name
        for step in observation.get("tactic_history", [])
        if step.get("success")
    ]
    tactic_bag = _tactic_bag(history_names)  # len = num_tactics (39)
    vec[8:8 + len(tactic_bag)] = tactic_bag  # [8:47]

    # ── Primary goal features [47:126] ────────────────────────────────────
    goals = observation.get("goals", [])
    if goals:
        primary = _encode_goal(goals[0])  # 79 floats
        vec[47:47 + len(primary)] = primary  # [47:126]

    # ── Secondary goal features [126:205] ─────────────────────────────────
    if len(goals) > 1:
        secondary = _encode_goal(goals[1])  # 79 floats
        vec[126:126 + len(secondary)] = secondary  # [126:205]

    # [205:256] remain 0 (padding / future use)
    return vec
