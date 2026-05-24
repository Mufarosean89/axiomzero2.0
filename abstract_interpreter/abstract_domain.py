"""
Axiom Zero - Abstract Domains

Defines the lattice structures used in abstract interpretation for
type inference and tensor shape analysis.

Provides:
- TypeDomain: Type lattice (⊥ → int/float/bool/tensor → ⊤)
- ShapeDimension: Single dimension in a tensor shape
- TensorShape: Symbolic tensor shape representation
- AbstractValue: Type + shape + constraints for a program point
- AbstractState: Complete analysis state for all variables
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class TypeDomain(Enum):
    """Type lattice for abstract interpretation.
    
    Order: ⊥ ⊑ {int, float, bool, string, tensor, list, dict} ⊑ ⊤
    """
    BOTTOM = "Bot"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    TENSOR = "tensor"
    LIST = "list"
    DICT = "dict"
    TOP = "Top"

    @staticmethod
    def join(t1: TypeDomain, t2: TypeDomain) -> TypeDomain:
        """Compute least upper bound (join) in type lattice."""
        if t1 == t2:
            return t1
        if t1 == TypeDomain.BOTTOM:
            return t2
        if t2 == TypeDomain.BOTTOM:
            return t1
        if t1 == TypeDomain.TOP or t2 == TypeDomain.TOP:
            return TypeDomain.TOP

        # Numeric types can join to a common supertype
        if {t1, t2} <= {TypeDomain.INT, TypeDomain.FLOAT}:
            return TypeDomain.FLOAT

        return TypeDomain.TOP

    @staticmethod
    def meet(t1: TypeDomain, t2: TypeDomain) -> TypeDomain:
        """Compute greatest lower bound (meet) in type lattice."""
        if t1 == t2:
            return t1
        if t1 == TypeDomain.TOP:
            return t2
        if t2 == TypeDomain.TOP:
            return t1
        if t1 == TypeDomain.BOTTOM or t2 == TypeDomain.BOTTOM:
            return TypeDomain.BOTTOM
        if {t1, t2} <= {TypeDomain.INT, TypeDomain.FLOAT}:
            # Meet of int and float is int (the more precise)
            if t1 == TypeDomain.INT or t2 == TypeDomain.INT:
                return TypeDomain.INT
            return TypeDomain.FLOAT
        return TypeDomain.BOTTOM

    @staticmethod
    def from_python_type(py_type: str) -> TypeDomain:
        """Map a Python type string to a type domain."""
        mapping = {
            "int": TypeDomain.INT,
            "float": TypeDomain.FLOAT,
            "bool": TypeDomain.BOOL,
            "str": TypeDomain.STRING,
            "string": TypeDomain.STRING,
            "Tensor": TypeDomain.TENSOR,
            "torch.Tensor": TypeDomain.TENSOR,
            "List": TypeDomain.LIST,
            "list": TypeDomain.LIST,
            "Dict": TypeDomain.DICT,
            "dict": TypeDomain.DICT,
        }
        return mapping.get(py_type, TypeDomain.TOP)


class ShapeDimension:
    """Represents a single dimension in a tensor shape."""

    def __init__(self, value: Optional[int] = None, symbolic: Optional[str] = None):
        self.value = value
        self.symbolic = symbolic

    @property
    def is_concrete(self) -> bool:
        return self.value is not None

    @property
    def is_symbolic(self) -> bool:
        return self.symbolic is not None

    @property
    def is_unknown(self) -> bool:
        return self.value is None and self.symbolic is None

    def matches(self, other: ShapeDimension) -> bool:
        """Check if two dimensions are compatible."""
        if self.is_concrete and other.is_concrete:
            return self.value == other.value
        # Symbolic dimensions can match any concrete or same symbolic
        if self.is_symbolic and other.is_symbolic:
            return self.symbolic == other.symbolic
        # Symbolic can match concrete
        return True

    def join(self, other: ShapeDimension) -> ShapeDimension:
        """Compute the join (LUB) of two dimensions."""
        if self == other:
            return ShapeDimension(self.value, self.symbolic)
        if self.is_unknown:
            return ShapeDimension(other.value, other.symbolic)
        if other.is_unknown:
            return ShapeDimension(self.value, self.symbolic)
        if self.is_concrete and other.is_concrete:
            if self.value == other.value:
                return ShapeDimension(value=self.value)
            return ShapeDimension(symbolic=f"dim")
        if self.is_symbolic and other.is_concrete:
            return ShapeDimension(symbolic=self.symbolic)
        if self.is_concrete and other.is_symbolic:
            return ShapeDimension(symbolic=other.symbolic)
        # Both symbolic
        if self.symbolic == other.symbolic:
            return ShapeDimension(symbolic=self.symbolic)
        return ShapeDimension(symbolic="dim")

    def __eq__(self, other):
        if not isinstance(other, ShapeDimension):
            return False
        return self.value == other.value and self.symbolic == other.symbolic

    def __hash__(self):
        return hash((self.value, self.symbolic))

    def __repr__(self):
        if self.is_concrete:
            return str(self.value)
        if self.is_symbolic:
            return self.symbolic
        return "?"


@dataclass
class TensorShape:
    """Symbolic tensor shape representation."""
    dimensions: List[ShapeDimension] = field(default_factory=list)
    rank: Optional[int] = None

    def __post_init__(self):
        if self.rank is None and self.dimensions:
            self.rank = len(self.dimensions)

    @staticmethod
    def from_list(shape_list: List[Union[int, str]]) -> TensorShape:
        """Create TensorShape from a list of ints or symbolic strings."""
        dims = []
        for dim in shape_list:
            if isinstance(dim, int):
                dims.append(ShapeDimension(value=dim))
            elif isinstance(dim, str):
                dims.append(ShapeDimension(symbolic=dim))
            else:
                dims.append(ShapeDimension())
        return TensorShape(dimensions=dims)

    @staticmethod
    def unknown(rank: Optional[int] = None) -> TensorShape:
        """Create an unknown tensor shape."""
        if rank is not None:
            dims = [ShapeDimension() for _ in range(rank)]
            return TensorShape(dimensions=dims, rank=rank)
        return TensorShape(rank=None)

    @property
    def is_fully_known(self) -> bool:
        return all(d.is_concrete for d in self.dimensions) if self.dimensions else False

    @property
    def has_symbolic_dims(self) -> bool:
        return any(d.is_symbolic for d in self.dimensions)

    def get_symbolic_dims(self) -> Dict[str, int]:
        """Get mapping of symbolic names to their positions."""
        return {d.symbolic: i for i, d in enumerate(self.dimensions) if d.is_symbolic}

    def compatible_with(self, other: TensorShape) -> bool:
        """Check if shapes are compatible for operations."""
        if self.rank is None or other.rank is None:
            return True
        if len(self.dimensions) != len(other.dimensions):
            return False
        return all(d1.matches(d2) for d1, d2 in zip(self.dimensions, other.dimensions))

    def join(self, other: TensorShape) -> TensorShape:
        """Compute the join (LUB) of two tensor shapes."""
        if self.rank is None:
            return TensorShape(rank=other.rank, dimensions=list(other.dimensions))
        if other.rank is None:
            return TensorShape(rank=self.rank, dimensions=list(self.dimensions))
        if self.rank != other.rank:
            return TensorShape.unknown()
        dims = [d1.join(d2) for d1, d2 in zip(self.dimensions, other.dimensions)]
        return TensorShape(dimensions=dims)

    def num_elements(self) -> Optional[int]:
        """Total number of elements if fully concrete."""
        if not self.is_fully_known:
            return None
        total = 1
        for d in self.dimensions:
            total *= d.value
        return total

    def __repr__(self):
        if not self.dimensions:
            return "Tensor[?]"
        dims_str = ", ".join(str(d) for d in self.dimensions)
        return f"Tensor[{dims_str}]"


@dataclass
class AbstractValue:
    """
    Abstract value combining type and shape information at a program point.
    
    Attributes:
        type_domain: The inferred type (part of type lattice)
        tensor_shape: Shape information for tensors
        concrete_value: Known concrete value (for constants)
        symbolic_constraints: Logical constraints on this value
        is_top: Whether this is the top element (any value)
    """
    type_domain: TypeDomain = TypeDomain.BOTTOM
    tensor_shape: Optional[TensorShape] = None
    concrete_value: Any = None
    symbolic_constraints: List[str] = field(default_factory=list)
    is_top: bool = False

    @property
    def is_tensor(self) -> bool:
        return self.type_domain == TypeDomain.TENSOR

    @property
    def has_shape(self) -> bool:
        return self.is_tensor and self.tensor_shape is not None

    def get_shape(self) -> Optional[TensorShape]:
        return self.tensor_shape if self.is_tensor else None

    @staticmethod
    def top() -> AbstractValue:
        return AbstractValue(type_domain=TypeDomain.TOP, is_top=True)

    @staticmethod
    def bottom() -> AbstractValue:
        return AbstractValue(type_domain=TypeDomain.BOTTOM)

    @staticmethod
    def from_type(type_domain: TypeDomain) -> AbstractValue:
        return AbstractValue(type_domain=type_domain)

    @staticmethod
    def from_tensor(shape: TensorShape) -> AbstractValue:
        return AbstractValue(type_domain=TypeDomain.TENSOR, tensor_shape=shape)

    @staticmethod
    def constant(value: Any, type_domain: TypeDomain) -> AbstractValue:
        return AbstractValue(type_domain=type_domain, concrete_value=value)

    def join(self, other: AbstractValue) -> AbstractValue:
        """Compute the join (LUB) with another abstract value."""
        if self.is_top:
            return self
        if other.is_top:
            return other
        if self.type_domain == TypeDomain.BOTTOM:
            return AbstractValue(
                type_domain=other.type_domain,
                tensor_shape=other.tensor_shape,
                concrete_value=other.concrete_value,
                symbolic_constraints=list(self.symbolic_constraints + other.symbolic_constraints),
            )
        if other.type_domain == TypeDomain.BOTTOM:
            return AbstractValue(
                type_domain=self.type_domain,
                tensor_shape=self.tensor_shape,
                concrete_value=self.concrete_value,
                symbolic_constraints=list(self.symbolic_constraints + other.symbolic_constraints),
            )

        joined_type = TypeDomain.join(self.type_domain, other.type_domain)

        # Merge tensor shapes
        joined_shape = None
        if self.is_tensor and other.is_tensor:
            if self.tensor_shape and other.tensor_shape:
                joined_shape = self.tensor_shape.join(other.tensor_shape)
            elif self.tensor_shape:
                joined_shape = self.tensor_shape
            else:
                joined_shape = other.tensor_shape

        # Keep concrete value only if both are the same
        concrete = self.concrete_value if self.concrete_value == other.concrete_value else None

        return AbstractValue(
            type_domain=joined_type,
            tensor_shape=joined_shape,
            concrete_value=concrete,
            symbolic_constraints=list(set(self.symbolic_constraints + other.symbolic_constraints)),
        )

    def __repr__(self):
        parts = [self.type_domain.value]
        if self.has_shape:
            parts.append(str(self.tensor_shape))
        if self.concrete_value is not None:
            parts.append(f"={self.concrete_value}")
        if self.symbolic_constraints:
            parts.append(f"{{{', '.join(self.symbolic_constraints)}}}")
        if self.is_top:
            parts.append("(⊤)")
        return " ".join(parts)


@dataclass
class AbstractState:
    """
    Complete abstract state after analysis.
    Contains inferred types, shapes, symbolic facts, and data flow info
    for all variables across the entire program.
    """

    # Variable environments: var_name → AbstractValue
    global_env: Dict[str, AbstractValue] = field(default_factory=dict)
    function_envs: Dict[str, Dict[str, AbstractValue]] = field(default_factory=dict)

    # Shape constraints and facts
    shape_facts: List[str] = field(default_factory=list)
    type_constraints: List[str] = field(default_factory=list)

    # Data flow information: var_name → list of dependencies
    data_flow_graph: Dict[str, List[str]] = field(default_factory=dict)

    # Function signatures with inferred types
    function_signatures: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Tensor operations metadata
    tensor_ops_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Analysis metadata
    analysis_complete: bool = False
    warnings: List[str] = field(default_factory=list)

    def add_shape_fact(self, fact: str):
        """Add a shape constraint fact."""
        if fact not in self.shape_facts:
            self.shape_facts.append(fact)

    def add_type_constraint(self, constraint: str):
        """Add a type constraint."""
        if constraint not in self.type_constraints:
            self.type_constraints.append(constraint)

    def get_variable_type(self, var_name: str, function: Optional[str] = None) -> Optional[AbstractValue]:
        """Get the inferred abstract value for a variable."""
        if function and function in self.function_envs:
            return self.function_envs[function].get(var_name)
        return self.global_env.get(var_name)

    def set_variable_type(self, var_name: str, value: AbstractValue, function: Optional[str] = None):
        """Set the inferred abstract value for a variable."""
        if function:
            if function not in self.function_envs:
                self.function_envs[function] = {}
            self.function_envs[function][var_name] = value
        else:
            self.global_env[var_name] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "global_env": {k: repr(v) for k, v in self.global_env.items()},
            "function_envs": {
                func: {k: repr(v) for k, v in env.items()}
                for func, env in self.function_envs.items()
            },
            "shape_facts": self.shape_facts,
            "type_constraints": self.type_constraints,
            "function_signatures": self.function_signatures,
            "tensor_ops_metadata": self.tensor_ops_metadata,
            "analysis_complete": self.analysis_complete,
            "warnings": self.warnings,
        }
