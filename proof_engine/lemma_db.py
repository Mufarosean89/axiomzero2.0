"""Axiom Zero — Lemma Database & Embedder

Provides a searchable database of mathlib lemmas with n-gram hash embeddings,
enabling the TacticExecutor to suggest concrete lemma names for ``apply``,
``exact``, ``refine``, ``rw``, and ``rewrite`` holes instead of using bare placeholders.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

LEMMA_EMBED_DIM = 128

# ── Embedding Helpers ─────────────────────────────────────────────────────


def _ngram_hash(text: str, dim: int = LEMMA_EMBED_DIM, n: int = 3) -> List[float]:
    """Map a string to a ``dim``-dimensional float vector via character n-gram hashing."""
    vec = [0.0] * dim
    text = text.lower().strip()
    if not text:
        return vec

    grams = [text[i:i + n] for i in range(len(text) - n + 1)] or [text]
    for gram in grams:
        digest = int(hashlib.sha256(gram.encode()).hexdigest(), 16)
        vec[digest % dim] += 1.0

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# ── Goal Type Classification ──────────────────────────────────────────────

_GOAL_CLASSIFIERS: List[tuple[str, str, callable]] = [
    ("equality", "=", lambda gt: "=" in gt or "≡" in gt),
    ("add_identity", "add_identity", lambda gt: bool(re.search(r'\b[xyz]\s*\+\s*0\b|\b0\s*\+\s*[xyz]\b', gt))),
    ("mul_identity", "mul_identity", lambda gt: bool(re.search(r'\b[xyz]\s*\*\s*1\b|\b1\s*\*\s*[xyz]\b', gt))),
    ("inequality", ">", lambda gt: ">" in gt or "<" in gt or "≥" in gt or "≤" in gt),
    ("arithmetic", "arithmetic", lambda gt: bool(re.search(r'\b[xyz]\s*[+*/\-]\s*[xyz\d]\b', gt))),
    ("boolean", "boolean", lambda gt: "∧" in gt or "∨" in gt or "¬" in gt or "Bool" in gt or "Prop" in gt),
    ("list", "list", lambda gt: "List" in gt or "list" in gt or "length" in gt.lower()),
    ("nat", "nat", lambda gt: "ℕ" in gt or "Nat" in gt),
    ("int", "int", lambda gt: "ℤ" in gt or "Int" in gt),
    ("implication", "implication", lambda gt: "→" in gt or "∀" in gt or "->" in gt),
    ("existential", "existential", lambda gt: "∃" in gt or "Exists" in gt),
    ("tensor", "tensor", lambda gt: "Tensor" in gt or "tensor" in gt.lower()),
]


def _classify_goal_type(goal_type: str) -> List[str]:
    """Return category tags that match the goal type."""
    tags: List[str] = []
    gt = goal_type.strip()
    for tag_name, _, matcher in _GOAL_CLASSIFIERS:
        if matcher(gt):
            if tag_name == "add_identity" or tag_name == "mul_identity":
                _add_if_absent(tags, tag_name)
            else:
                _add_if_absent(tags, tag_name)
    if "=" in gt or "≡" in gt:
        _add_if_absent(tags, "equality")
        if re.search(r'\b\d+\s*[+*/]\s*\d+\b', gt) or re.search(r'\b[xyz]\s*[+*/]\s*\d+\b', gt):
            _add_if_absent(tags, "arithmetic")
    return tags


def _add_if_absent(lst: List[str], item: str) -> None:
    if item not in lst:
        lst.append(item)


# ── Data Structures ───────────────────────────────────────────────────────


@dataclass
class LemmaEntry:
    """A single lemma in the database.

    Attributes:
        name: Fully qualified lemma name (e.g., ``add_comm``).
        type_sig: The type signature as a string.
        category: Rough category tag (e.g., ``arithmetic``, ``list``, ``order``).
        description: Human-readable description.
        embedding: Precomputed 128-dim feature vector (lazy-computed if empty).
        usage_count: How many times this lemma was suggested and selected.
        success_count: How many times it led to a successful tactic application.
    """
    name: str
    type_sig: str = ""
    category: str = "general"
    description: str = ""
    embedding: List[float] = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0

    def __post_init__(self):
        if not self.embedding:
            self.embedding = _ngram_hash(f"{self.name} {self.type_sig}")

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


@dataclass
class LemmaSuggestion:
    """A suggested lemma with its relevance score.

    Attributes:
        lemma_name: The lemma name to use.
        score: Cosine similarity score in [0, 1].
        tactic: The tactic type this is suggested for (``apply``, ``rw``, etc.).
        rendered: The full tactic string (e.g., ``apply add_comm``).
    """
    lemma_name: str
    score: float
    tactic: str = "apply"
    rendered: str = ""

    def __post_init__(self):
        if not self.rendered:
            self._render()

    def _render(self):
        tactic_map = {
            "rw": f"rw [{self.lemma_name}]",
            "rewrite": f"rw [{self.lemma_name}]",
            "exact": f"exact {self.lemma_name}",
            "refine": f"refine {self.lemma_name} ?_",
            "have": f"have h : ?_ := {self.lemma_name}",
        }
        self.rendered = tactic_map.get(self.tactic, f"{self.tactic} {self.lemma_name}")


# ── Seed Lemma Set ─────────────────────────────────────────────────────────

# Each entry: (name, type_sig, category, description)
_SEED_LEMMAS: List[Tuple[str, str, str, str]] = [
    # Arithmetic identities
    ("add_comm",       "∀ {a b : ℕ}, a + b = b + a",                "arithmetic", "Addition is commutative"),
    ("add_assoc",      "∀ {a b c : ℕ}, (a + b) + c = a + (b + c)",  "arithmetic", "Addition is associative"),
    ("add_zero",       "∀ {a : ℕ}, a + 0 = a",                      "arithmetic", "Zero is a right additive identity"),
    ("zero_add",       "∀ {a : ℕ}, 0 + a = a",                      "arithmetic", "Zero is a left additive identity"),
    ("add_left_neg",   "∀ {a : ℤ}, (-a) + a = 0",                   "arithmetic", "Left additive inverse"),
    ("add_right_neg",  "∀ {a : ℤ}, a + (-a) = 0",                   "arithmetic", "Right additive inverse"),
    ("add_sub_cancel", "∀ {a b : ℤ}, a + b - b = a",                "arithmetic", "Adding then subtracting cancels"),
    ("sub_add_cancel", "∀ {a b : ℤ}, a - b + b = a",                "arithmetic", "Subtracting then adding cancels"),
    ("sub_self",       "∀ {a : ℤ}, a - a = 0",                      "arithmetic", "Subtracting a number from itself"),
    ("sub_eq_add_neg", "∀ {a b : ℤ}, a - b = a + (-b)",             "arithmetic", "Rewrite subtraction as addition"),

    # Multiplication
    ("mul_comm",       "∀ {a b : ℕ}, a * b = b * a",                "arithmetic", "Multiplication is commutative"),
    ("mul_assoc",      "∀ {a b c : ℕ}, (a * b) * c = a * (b * c)",  "arithmetic", "Multiplication is associative"),
    ("mul_one",        "∀ {a : ℕ}, a * 1 = a",                      "arithmetic", "One is a right multiplicative identity"),
    ("one_mul",        "∀ {a : ℕ}, 1 * a = a",                      "arithmetic", "One is a left multiplicative identity"),
    ("mul_zero",       "∀ {a : ℕ}, a * 0 = 0",                      "arithmetic", "Multiplying by zero gives zero"),
    ("zero_mul",       "∀ {a : ℕ}, 0 * a = 0",                      "arithmetic", "Multiplying zero by a gives zero"),
    ("add_mul",        "∀ {a b c : ℕ}, (a + b) * c = a * c + b * c","arithmetic", "Distributivity"),
    ("mul_add",        "∀ {a b c : ℕ}, a * (b + c) = a * b + a * c","arithmetic", "Distributivity"),

    # Order / inequalities
    ("le_refl",        "∀ {a : ℕ}, a ≤ a",                          "order",      "≤ is reflexive"),
    ("le_trans",       "∀ {a b c : ℕ}, a ≤ b → b ≤ c → a ≤ c",     "order",      "≤ is transitive"),
    ("le_of_lt",       "∀ {a b : ℕ}, a < b → a ≤ b",               "order",      "Strict inequality implies weak"),
    ("lt_of_lt_of_le", "∀ {a b c : ℕ}, a < b → b ≤ c → a < c",     "order",      "Transitivity of < over ≤"),
    ("lt_of_le_of_lt", "∀ {a b c : ℕ}, a ≤ b → b < c → a < c",     "order",      "Transitivity of ≤ over <"),
    ("lt_irrefl",      "∀ {a : ℕ}, ¬ a < a",                       "order",      "No number is strictly less than itself"),
    ("add_lt_add_right","∀ {a b : ℕ} (c : ℕ), a < b → a + c < b + c","order",    "Adding preserves strict inequality"),
    ("add_le_add_right","∀ {a b : ℕ} (c : ℕ), a ≤ b → a + c ≤ b + c","order",    "Adding preserves weak inequality"),
    ("mul_pos",        "∀ {a b : ℤ}, a > 0 → b > 0 → a * b > 0",   "order",      "Product of two positives is positive"),

    # Natural number specific
    ("Nat.succ_ne_self",  "∀ (n : ℕ), n.succ ≠ n",                 "nat",        "Successor is not equal to original"),
    ("Nat.zero_lt_succ",  "∀ (n : ℕ), 0 < n.succ",                 "nat",        "Zero is less than any successor"),
    ("Nat.lt_succ_self",  "∀ (n : ℕ), n < n.succ",                 "nat",        "Every number is less than its successor"),
    ("Nat.succ_eq_add_one","∀ (n : ℕ), n.succ = n + 1",            "nat",        "Successor equals n + 1"),
    ("Nat.add_succ",      "∀ (n m : ℕ), n + m.succ = (n + m).succ","nat",        "Add with successor"),
    ("Nat.succ_add",      "∀ (n m : ℕ), n.succ + m = (n + m).succ","nat",        "Successor add"),

    # Integer specific
    ("Int.add_comm",   "∀ {a b : ℤ}, a + b = b + a",                "int",        "Addition is commutative on ℤ"),

    # List operations
    ("List.length_append",  "∀ {α : Type} (l₁ l₂ : List α), (l₁ ++ l₂).length = l₁.length + l₂.length", "list", "Length distributes over append"),
    ("List.length_reverse", "∀ {α : Type} (l : List α), l.reverse.length = l.length", "list", "Reverse does not change length"),
    ("List.length_nil",     "∀ {α : Type}, ([] : List α).length = 0", "list",           "Length of nil is 0"),
    ("List.sum_append",     "∀ {α : Type} [AddMonoid α] (l₁ l₂ : List α), (l₁ ++ l₂).sum = l₁.sum + l₂.sum", "list", "Sum distributes over append"),
    ("List.sum_nil",        "∀ {α : Type} [AddMonoid α], ([] : List α).sum = 0", "list", "Sum of nil is 0"),
    ("List.sum_cons",       "∀ {α : Type} [AddMonoid α] (x : α) (xs : List α), (x :: xs).sum = x + xs.sum", "list", "Sum of cons"),
    ("List.length_cons",    "∀ {α : Type} (x : α) (xs : List α), (x :: xs).length = xs.length + 1", "list", "Length of cons"),

    # Boolean / propositional
    ("Bool.not_true",  "¬ true",                                     "boolean",    "Not true is false"),
    ("Bool.not_false", "¬ false",                                    "boolean",    "Not false is true"),
    ("not_not",        "∀ {p : Prop}, ¬¬p ↔ p",                     "boolean",    "Double negation elimination"),
    ("by_contra",      "∀ {p : Prop}, (¬p → False) → p",            "boolean",    "Proof by contradiction"),
    ("and_comm",       "∀ {p q : Prop}, p ∧ q ↔ q ∧ p",             "boolean",    "∧ is commutative"),
    ("or_comm",        "∀ {p q : Prop}, p ∨ q ↔ q ∨ p",             "boolean",    "∨ is commutative"),

    # Tensor / shape (stubs for torch compiler proofs)
    ("Tensor.shape",   "∀ (t : Tensor), shape t = ...",              "tensor",     "Tensor shape property"),
    ("Tensor.add_shape","shape (a + b) = shape a if shape a = shape b", "tensor",  "Tensor add preserves shape"),
]


# ── LemmaDatabase ──────────────────────────────────────────────────────────

class LemmaDatabase:
    """Searchable database of mathlib lemmas with embedding-based retrieval."""

    def __init__(self, include_seed: bool = True) -> None:
        self._lemmas: Dict[str, LemmaEntry] = {}
        self._category_index: Dict[str, List[str]] = {}
        if include_seed:
            self._load_seed_lemmas()

    # ── Public API ────────────────────────────────────────────────────────

    def add_lemma(
        self,
        name: str,
        type_sig: str = "",
        category: str = "general",
        description: str = "",
    ) -> LemmaEntry:
        """Add a lemma to the database."""
        entry = LemmaEntry(name=name, type_sig=type_sig, category=category, description=description)
        self._lemmas[name] = entry
        self._category_index.setdefault(category, []).append(name)
        return entry

    def add_lemmas_from_list(self, lemmas: List[Tuple[str, str, str, str]]) -> None:
        """Bulk-add lemmas from (name, type_sig, category, description) tuples."""
        for name, type_sig, category, desc in lemmas:
            self.add_lemma(name, type_sig, category, desc)

    def search(
        self,
        goal_type: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        min_success_rate: float = 0.0,
    ) -> List[LemmaSuggestion]:
        """Search for lemmas relevant to a given goal type.

        Returns suggestions ordered by relevance (descending).
        """
        if not self._lemmas:
            return []

        goal_tags = _classify_goal_type(goal_type)

        candidates: List[LemmaEntry] = [
            lemma for lemma in self._lemmas.values()
            if not (category_filter and lemma.category != category_filter)
            and not (min_success_rate > 0 and lemma.success_rate < min_success_rate)
        ]

        if not candidates:
            return []

        goal_emb = _ngram_hash(goal_type)
        scored: List[Tuple[float, LemmaEntry]] = []

        for lemma in candidates:
            sim = _cosine_sim(goal_emb, lemma.embedding)
            if lemma.category in goal_tags:
                sim += 0.1
            sim += 0.05 * lemma.success_rate
            scored.append((sim, lemma))

        scored.sort(key=lambda x: -x[0])

        return [
            LemmaSuggestion(lemma_name=lemma.name, score=min(score, 1.0), tactic="apply")
            for score, lemma in scored[:top_k]
        ]

    def suggest_for_goal(
        self,
        goal_type: str,
        tactic: str = "apply",
        category_filter: Optional[str] = None,
    ) -> Optional[LemmaSuggestion]:
        """Get the single best lemma suggestion for a goal and tactic.

        This is the primary entry point for ``TacticExecutor``.
        """
        results = self.search(goal_type, top_k=3, category_filter=category_filter)
        if not results:
            return None
        best = results[0]
        best.tactic = tactic
        best._render()
        return best

    def suggest_for_hypothesis(self, hyp_type: str, tactic: str = "apply") -> Optional[LemmaSuggestion]:
        """Suggest a lemma given a hypothesis type (different from goal type)."""
        return self.suggest_for_goal(hyp_type, tactic=tactic)

    # ── Feedback ─────────────────────────────────────────────────────────

    def record_hit(self, lemma_name: str) -> None:
        """Record that a lemma suggestion led to a successful tactic application."""
        entry = self._lemmas.get(lemma_name)
        if entry:
            entry.usage_count += 1
            entry.success_count += 1

    def record_miss(self, lemma_name: str) -> None:
        """Record that a lemma suggestion led to a tactic failure."""
        entry = self._lemmas.get(lemma_name)
        if entry:
            entry.usage_count += 1

    def record_result(self, lemma_name: str, success: bool) -> None:
        """Record whether a lemma suggestion succeeded or failed."""
        if success:
            self.record_hit(lemma_name)
        else:
            self.record_miss(lemma_name)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the database to a JSON file."""
        data = {
            "lemmas": [
                {
                    "name": e.name,
                    "type_sig": e.type_sig,
                    "category": e.category,
                    "description": e.description,
                    "embedding": e.embedding,
                    "usage_count": e.usage_count,
                    "success_count": e.success_count,
                }
                for e in self._lemmas.values()
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LemmaDatabase":
        """Load a database from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        db = cls.__new__(cls)
        db._lemmas = {}
        db._category_index = {}
        for entry_data in data.get("lemmas", []):
            entry = LemmaEntry(
                name=entry_data["name"],
                type_sig=entry_data.get("type_sig", ""),
                category=entry_data.get("category", "general"),
                description=entry_data.get("description", ""),
                embedding=entry_data.get("embedding", []),
                usage_count=entry_data.get("usage_count", 0),
                success_count=entry_data.get("success_count", 0),
            )
            db._lemmas[entry.name] = entry
            db._category_index.setdefault(entry.category, []).append(entry.name)
        return db

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def lemma_count(self) -> int:
        return len(self._lemmas)

    @property
    def lemma_names(self) -> List[str]:
        return sorted(self._lemmas.keys())

    @property
    def categories(self) -> List[str]:
        return sorted(self._category_index.keys())

    def get_lemma(self, name: str) -> Optional[LemmaEntry]:
        return self._lemmas.get(name)

    def clear(self) -> None:
        self._lemmas.clear()
        self._category_index.clear()

    # ── Internal ─────────────────────────────────────────────────────────

    def _load_seed_lemmas(self) -> None:
        self.add_lemmas_from_list(_SEED_LEMMAS)
