"""Policy & Value Networks: pure-Python MLP (no numpy/torch).

Architecture: FEATURE_DIM → 128 (ReLU) → 64 (ReLU) → policy_head (softmax, 39)
                                                      → value_head (tanh, 1)

Lightweight implementation for correctness testing of MCTS/self-play;
swap for PyTorch equivalents in production.
"""

from __future__ import annotations

import json
import math
import random
from typing import List, Tuple

from .encoder import FEATURE_DIM

# ── Hyper-parameters ────────────────────────────────────────────────────────
HIDDEN1 = 128
HIDDEN2 = 64
_NUM_ACTIONS: int | None = None  # resolved lazily


def _get_num_actions() -> int:
    global _NUM_ACTIONS
    if _NUM_ACTIONS is None:
        from proof_engine import CORE_TACTICS
        _NUM_ACTIONS = len(CORE_TACTICS)
    return _NUM_ACTIONS


# ── Math helpers ─────────────────────────────────────────────────────────────

def _relu(x: float) -> float:
    return max(0.0, x)


def _tanh(x: float) -> float:
    # Clamp to avoid overflow
    x = max(-20.0, min(20.0, x))
    e2x = math.exp(2 * x)
    return (e2x - 1) / (e2x + 1)


def _softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _linear(x: List[float], W: List[List[float]], b: List[float]) -> List[float]:
    """y = W @ x + b  (W is [out_dim x in_dim])"""
    return [
        sum(W[i][j] * x[j] for j in range(len(x))) + b[i]
        for i in range(len(W))
    ]


def _he_init(out_dim: int, in_dim: int) -> Tuple[List[List[float]], List[float]]:
    """He (Kaiming) uniform initialisation for ReLU layers."""
    std = math.sqrt(2.0 / in_dim)
    W = [[random.gauss(0, std) for _ in range(in_dim)] for _ in range(out_dim)]
    b = [0.0] * out_dim
    return W, b


# ── Network ──────────────────────────────────────────────────────────────────

class PolicyValueNet:
    """Shared-trunk policy + value network with manual backprop."""

    def __init__(self, num_actions: int | None = None) -> None:
        na = num_actions or _get_num_actions()
        self.num_actions = na
        self.W1, self.b1 = _he_init(HIDDEN1, FEATURE_DIM)
        self.W2, self.b2 = _he_init(HIDDEN2, HIDDEN1)
        self.Wp, self.bp = _he_init(na, HIDDEN2)
        self.Wv, self.bv = _he_init(1, HIDDEN2)

    def _backbone(self, x: List[float]) -> List[float]:
        h1 = [_relu(v) for v in _linear(x, self.W1, self.b1)]
        h2 = [_relu(v) for v in _linear(h1, self.W2, self.b2)]
        return h2

    def forward(self, state_vec: List[float]) -> Tuple[List[float], float]:
        """Forward pass: returns (priors, value) — softmax policy + tanh value."""
        h = self._backbone(state_vec)
        logits = _linear(h, self.Wp, self.bp)
        priors = _softmax(logits)
        value_logit = _linear(h, self.Wv, self.bv)[0]
        value = _tanh(value_logit)
        return priors, value

    # ── Weight update (used by the training loop) ─────────────────────────

    def update_weights(
        self,
        target_policies: List[List[float]],
        target_values: List[float],
        state_vecs: List[List[float]],
        lr: float = 1e-3,
    ) -> Tuple[float, float]:
        """Single-step batch gradient descent (manual backprop). Returns (pi_loss, v_loss)."""
        assert len(target_policies) == len(target_values) == len(state_vecs)
        n = len(state_vecs)
        if n == 0:
            return 0.0, 0.0

        na = self.num_actions

        # ── Initialise gradient accumulators ─────────────────────────────
        dW1: List[List[float]] = [[0.0] * FEATURE_DIM for _ in range(HIDDEN1)]
        db1: List[float] = [0.0] * HIDDEN1
        dW2: List[List[float]] = [[0.0] * HIDDEN1 for _ in range(HIDDEN2)]
        db2: List[float] = [0.0] * HIDDEN2
        dWp: List[List[float]] = [[0.0] * HIDDEN2 for _ in range(na)]
        dbp: List[float] = [0.0] * na
        dWv: List[List[float]] = [[0.0] * HIDDEN2]
        dbv: List[float] = [0.0]

        total_pi_loss = 0.0
        total_v_loss = 0.0
        eps = 1e-9

        for state_vec, pi_target, v_target in zip(state_vecs, target_policies, target_values):
            # ── Forward pass (save intermediates for backprop) ───────────
            h1_pre = _linear(state_vec, self.W1, self.b1)          # HIDDEN1
            h1 = [_relu(v) for v in h1_pre]                         # HIDDEN1
            h2_pre = _linear(h1, self.W2, self.b2)                  # HIDDEN2
            h2 = [_relu(v) for v in h2_pre]                         # HIDDEN2

            # Policy head forward
            logits = _linear(h2, self.Wp, self.bp)                  # NUM_ACTIONS
            priors = _softmax(logits)
            pi_loss = -sum(pi_target[i] * math.log(priors[i] + eps) for i in range(na))
            total_pi_loss += pi_loss

            # Value head forward
            v_logit = _linear(h2, self.Wv, self.bv)[0]              # 1
            v_pred = _tanh(v_logit)
            v_loss = (v_pred - v_target) ** 2
            total_v_loss += v_loss

            # ── Backward pass ───────────────────────────────────────────

            # Policy head gradient (cross-entropy + softmax)
            d_logits = [priors[i] - pi_target[i] for i in range(na)]

            # Value head gradient (MSE + tanh)
            d_v = 2.0 * (v_pred - v_target) * (1.0 - v_pred * v_pred)

            # Accumulate head gradients
            for i in range(na):
                for j in range(HIDDEN2):
                    dWp[i][j] += d_logits[i] * h2[j]
                dbp[i] += d_logits[i]

            for j in range(HIDDEN2):
                dWv[0][j] += d_v * h2[j]
            dbv[0] += d_v

            # Gradient w.r.t. h2 (shared representation)
            d_h2: List[float] = [0.0] * HIDDEN2
            for j in range(HIDDEN2):
                # Contribution from policy head
                for i in range(na):
                    d_h2[j] += d_logits[i] * self.Wp[i][j]
                # Contribution from value head
                d_h2[j] += d_v * self.Wv[0][j]

            # Backprop through layer 2: ReLU then linear
            d_h2_pre = [d_h2[j] * (1.0 if h2_pre[j] > 0.0 else 0.0) for j in range(HIDDEN2)]
            for j in range(HIDDEN2):
                for k in range(HIDDEN1):
                    dW2[j][k] += d_h2_pre[j] * h1[k]
                db2[j] += d_h2_pre[j]

            # Backprop through layer 2 weights to h1
            d_h1: List[float] = [0.0] * HIDDEN1
            for k in range(HIDDEN1):
                for j in range(HIDDEN2):
                    d_h1[k] += d_h2_pre[j] * self.W2[j][k]

            # Backprop through layer 1: ReLU then linear
            d_h1_pre = [d_h1[k] * (1.0 if h1_pre[k] > 0.0 else 0.0) for k in range(HIDDEN1)]
            for i in range(HIDDEN1):
                for k in range(FEATURE_DIM):
                    dW1[i][k] += d_h1_pre[i] * state_vec[k]
                db1[i] += d_h1_pre[i]

        # ── Apply accumulated gradients ─────────────────────────────────
        inv_n = 1.0 / n
        for i in range(HIDDEN1):
            for k in range(FEATURE_DIM):
                self.W1[i][k] -= lr * dW1[i][k] * inv_n
            self.b1[i] -= lr * db1[i] * inv_n

        for j in range(HIDDEN2):
            for k in range(HIDDEN1):
                self.W2[j][k] -= lr * dW2[j][k] * inv_n
            self.b2[j] -= lr * db2[j] * inv_n

        for i in range(na):
            for j in range(HIDDEN2):
                self.Wp[i][j] -= lr * dWp[i][j] * inv_n
            self.bp[i] -= lr * dbp[i] * inv_n

        for j in range(HIDDEN2):
            self.Wv[0][j] -= lr * dWv[0][j] * inv_n
        self.bv[0] -= lr * dbv[0] * inv_n

        return total_pi_loss / n, total_v_loss / n

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist weights to a JSON checkpoint."""
        data = {
            "num_actions": self.num_actions,
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "Wp": self.Wp, "bp": self.bp,
            "Wv": self.Wv, "bv": self.bv,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "PolicyValueNet":
        """Restore weights from a JSON checkpoint."""
        with open(path, "r") as f:
            data = json.load(f)
        net = cls.__new__(cls)
        net.num_actions = data["num_actions"]
        net.W1 = data["W1"]; net.b1 = data["b1"]
        net.W2 = data["W2"]; net.b2 = data["b2"]
        net.Wp = data["Wp"]; net.bp = data["bp"]
        net.Wv = data["Wv"]; net.bv = data["bv"]
        return net
