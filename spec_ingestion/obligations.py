"""Proof obligation data structures extracted from formal specifications."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class ObligationKind(Enum):
    PRECONDITION = auto()
    POSTCONDITION = auto()
    LOOP_INVARIANT = auto()
    LOOP_TERMINATION = auto()
    ASSERTION = auto()
    SAFETY = auto()
    TYPE_CONDITION = auto()
    SHAPE_CONDITION = auto()
    EQUALITY = auto()


class ObligationStatus(Enum):
    UNPROVEN = auto()
    IN_PROGRESS = auto()
    PROVEN = auto()
    FAILED = auto()
    ERROR = auto()


@dataclass
class ProofObligation:
    """A single proof obligation the RL agent must prove."""
    kind: ObligationKind
    predicate: str
    location: str = ""
    function: str = ""
    context: Dict[str, str] = field(default_factory=dict)
    hypotheses: List[str] = field(default_factory=list)
    status: ObligationStatus = ObligationStatus.UNPROVEN
    proof: Optional[str] = None
    difficulty: float = 0.5
    dependencies: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        pred_hash = hashlib.md5(self.predicate.encode()).hexdigest()[:4]
        return f"{self.function}:{self.location}:{self.kind.name}:{pred_hash}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.name,
            "predicate": self.predicate,
            "location": self.location,
            "function": self.function,
            "context": self.context,
            "hypotheses": self.hypotheses,
            "status": self.status.name,
            "proof": self.proof,
            "difficulty": self.difficulty,
        }


@dataclass
class SpecCollection:
    """Collection of proof obligations extracted from a program."""
    preconditions: List[ProofObligation] = field(default_factory=list)
    postconditions: List[ProofObligation] = field(default_factory=list)
    loop_invariants: List[ProofObligation] = field(default_factory=list)
    loop_termination: List[ProofObligation] = field(default_factory=list)
    assertions: List[ProofObligation] = field(default_factory=list)
    safety_conditions: List[ProofObligation] = field(default_factory=list)

    @property
    def all(self) -> List[ProofObligation]:
        return (
            self.preconditions + self.postconditions
            + self.loop_invariants + self.loop_termination
            + self.assertions + self.safety_conditions
        )

    @property
    def total_count(self) -> int:
        return len(self.all)

    @property
    def proven_count(self) -> int:
        return sum(1 for o in self.all if o.status == ObligationStatus.PROVEN)

    def add(self, obligation: ProofObligation):
        mapping = {
            ObligationKind.PRECONDITION: self.preconditions,
            ObligationKind.POSTCONDITION: self.postconditions,
            ObligationKind.LOOP_INVARIANT: self.loop_invariants,
            ObligationKind.LOOP_TERMINATION: self.loop_termination,
            ObligationKind.ASSERTION: self.assertions,
            ObligationKind.SAFETY: self.safety_conditions,
            ObligationKind.TYPE_CONDITION: self.safety_conditions,
            ObligationKind.SHAPE_CONDITION: self.safety_conditions,
            ObligationKind.EQUALITY: self.assertions,
        }
        mapping.get(obligation.kind, self.assertions).append(obligation)

    def get_function_obligations(self, function_name: str) -> List[ProofObligation]:
        return [o for o in self.all if o.function == function_name]

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "PROOF OBLIGATIONS SUMMARY",
            "=" * 50,
            f"Total: {self.total_count}",
            f"Proven: {self.proven_count}",
            f"Remaining: {self.total_count - self.proven_count}",
            "",
            f"  Preconditions:    {len(self.preconditions)}",
            f"  Postconditions:   {len(self.postconditions)}",
            f"  Loop invariants:  {len(self.loop_invariants)}",
            f"  Loop termination: {len(self.loop_termination)}",
            f"  Assertions:       {len(self.assertions)}",
            f"  Safety:           {len(self.safety_conditions)}",
        ]
        return "\n".join(lines)
