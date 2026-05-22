"""
Axiom Zero - Proof Obligations

Defines the data structures for representing proof obligations extracted
from formal specifications. A proof obligation is a logical statement that
must be proven for the program to be correct.

Obligations come from:
- @requires / @precondition annotations
- @ensures / @postcondition annotations  
- @invariant annotations on loops
- Loop termination conditions
- Assert statements
- Implicit safety conditions (no divide by zero, in-bounds access, etc.)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


class ObligationKind(Enum):
    """Types of proof obligations."""
    PRECONDITION = auto()       # @requires
    POSTCONDITION = auto()      # @ensures
    LOOP_INVARIANT = auto()     # @invariant on loops
    LOOP_TERMINATION = auto()   # Loop must terminate
    ASSERTION = auto()          # assert statement
    SAFETY = auto()             # Implicit safety (bounds, div by zero)
    TYPE_CONDITION = auto()     # Type compatibility
    SHAPE_CONDITION = auto()    # Tensor shape compatibility
    EQUALITY = auto()           # Equality to prove


class ObligationStatus(Enum):
    """Status of a proof obligation."""
    UNPROVEN = auto()        # Not yet attempted
    IN_PROGRESS = auto()     # Being worked on
    PROVEN = auto()          # Successfully proved
    FAILED = auto()          # Proof attempt failed
    ERROR = auto()           # Error in the obligation itself


@dataclass
class ProofObligation:
    """
    A single proof obligation that the RL agent must prove.
    
    Attributes:
        kind: What kind of obligation this is
        predicate: The logical statement to prove (as a string)
        location: Where in the source code this came from
        function: The function this obligation belongs to
        context: Variables and their types in scope
        hypotheses: Available hypotheses that can be used
        status: Whether this has been proven
        proof: The Lean proof term (once found)
    """
    kind: ObligationKind
    predicate: str                    # e.g., "x + 0 == x"
    location: str = ""                # e.g., "example.py:42"
    function: str = ""                # Function name
    context: Dict[str, str] = field(default_factory=dict)  # var → type
    hypotheses: List[str] = field(default_factory=list)    # Available assumptions
    status: ObligationStatus = ObligationStatus.UNPROVEN
    proof: Optional[str] = None       # Lean proof term
    difficulty: float = 0.5           # Estimated difficulty (0-1)
    dependencies: List[str] = field(default_factory=list)  # Other obligation IDs
    
    @property
    def id(self) -> str:
        """Unique identifier for this obligation."""
        pred_hash = hashlib.md5(self.predicate.encode()).hexdigest()[:4]
        return f"{self.function}:{self.location}:{self.kind.name}:{pred_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
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
    """
    Collection of all proof obligations extracted from a program.
    
    Attributes:
        preconditions: Input constraints for functions
        postconditions: Output guarantees for functions
        loop_invariants: Loop invariants that must hold
        assertions: Assert statements to prove
        safety_conditions: Implicit safety checks
    """
    preconditions: List[ProofObligation] = field(default_factory=list)
    postconditions: List[ProofObligation] = field(default_factory=list)
    loop_invariants: List[ProofObligation] = field(default_factory=list)
    loop_termination: List[ProofObligation] = field(default_factory=list)
    assertions: List[ProofObligation] = field(default_factory=list)
    safety_conditions: List[ProofObligation] = field(default_factory=list)

    @property
    def all(self) -> List[ProofObligation]:
        """Get all obligations."""
        return (
            self.preconditions
            + self.postconditions
            + self.loop_invariants
            + self.loop_termination
            + self.assertions
            + self.safety_conditions
        )

    @property
    def total_count(self) -> int:
        return len(self.all)

    @property
    def proven_count(self) -> int:
        return sum(1 for o in self.all if o.status == ObligationStatus.PROVEN)

    def add(self, obligation: ProofObligation):
        """Add an obligation to the appropriate list."""
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
        """Get all obligations for a specific function."""
        return [o for o in self.all if o.function == function_name]

    def summary(self) -> str:
        """Return a human-readable summary."""
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
