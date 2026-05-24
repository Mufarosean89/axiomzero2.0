"""Axiom Zero - Intermediate Representation (IR)

Defines the normalized intermediate representation used throughout the pipeline.
The IR strips Python-isms and keeps only semantically meaningful constructs:
function signatures, loop bounds, conditionals, tensor operations, and type annotations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ─── Parser Configuration & Warnings ─────────────────────────────────────────

@dataclass
class ParserWarning:
    """Warning produced when an unsupported Python construct is encountered during parsing."""

    construct: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: Unsupported construct '{self.construct}' — {self.message}"


@dataclass
class ParserConfig:
    """Configuration for the Python source parser.

    Attributes:
        strict_mode: If True, unsupported constructs raise ParserError instead of warnings.
        collect_warnings: If True, warnings are collected and returned with the IR.
    """

    strict_mode: bool = False
    collect_warnings: bool = True


# ─── Type Representations ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class TypeAnnotation:
    """A type annotation on a variable or expression."""

    name: str
    params: List["TypeAnnotation"] = field(default_factory=list)

    def to_string(self) -> str:
        if not self.params:
            return self.name
        inner = ", ".join(p.to_string() for p in self.params)
        return f"{self.name}[{inner}]"

    @staticmethod
    def int_type() -> "TypeAnnotation":
        return TypeAnnotation("int")

    @staticmethod
    def float_type() -> "TypeAnnotation":
        return TypeAnnotation("float")

    @staticmethod
    def bool_type() -> "TypeAnnotation":
        return TypeAnnotation("bool")

    @staticmethod
    def tensor_type(dtype: str = "float32") -> "TypeAnnotation":
        return TypeAnnotation("Tensor", [TypeAnnotation(dtype)])

    @staticmethod
    def list_type(elem: "TypeAnnotation") -> "TypeAnnotation":
        return TypeAnnotation("List", [elem])

    @staticmethod
    def from_string(s: str) -> "TypeAnnotation":
        """Parse a simple type string into a TypeAnnotation."""
        s = s.strip()
        bracket_pos = s.find("[")
        if bracket_pos == -1:
            return TypeAnnotation(s)

        name = s[:bracket_pos]
        inner = s[bracket_pos + 1: s.rindex("]")]
        params = [TypeAnnotation.from_string(p.strip()) for p in _split_params(inner)]
        return TypeAnnotation(name, params)


def _split_params(s: str) -> List[str]:
    """Split comma-separated type params, respecting nested brackets."""
    parts = []
    depth = 0
    current: List[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


# ─── Tensor Operations ─────────────────────────────────────────────────────────

class TensorOpKind(Enum):
    """Kinds of tensor operations tracked by the analyzer."""

    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MATMUL = auto()
    RELU = auto()
    SIGMOID = auto()
    TANH = auto()
    SOFTMAX = auto()
    RESHAPE = auto()
    VIEW = auto()
    TRANSPOSE = auto()
    PERMUTE = auto()
    CONCAT = auto()
    STACK = auto()
    SLICE = auto()
    SUM = auto()
    MEAN = auto()
    MAX = auto()
    MIN = auto()
    DROPOUT = auto()
    BATCH_NORM = auto()
    LAYER_NORM = auto()
    CONV2D = auto()
    MAX_POOL2D = auto()
    FLATTEN = auto()
    LINEAR = auto()
    EMBEDDING = auto()
    UNKNOWN = auto()


# ─── Expression IR ─────────────────────────────────────────────────────────────

@dataclass
class ExpressionIR:
    """IR node for an expression."""

    expr_type: str
    value: Any = None
    name: str = ""
    op: str = ""
    left: Optional["ExpressionIR"] = None
    right: Optional["ExpressionIR"] = None
    operand: Optional["ExpressionIR"] = None
    func: Optional["ExpressionIR"] = None
    args: List["ExpressionIR"] = field(default_factory=list)
    kwargs: Dict[str, "ExpressionIR"] = field(default_factory=dict)
    target: Optional["ExpressionIR"] = None
    attr: str = ""
    index: Optional["ExpressionIR"] = None
    elements: List["ExpressionIR"] = field(default_factory=list)
    keys: List["ExpressionIR"] = field(default_factory=list)
    tensor_op_kind: TensorOpKind = TensorOpKind.UNKNOWN
    type: Optional[TypeAnnotation] = None
    source_loc: Optional[str] = None

    @staticmethod
    def constant(value: Any, source_loc: str = None) -> "ExpressionIR":
        return ExpressionIR(expr_type="constant", value=value, source_loc=source_loc)

    @staticmethod
    def variable(name: str, source_loc: str = None) -> "ExpressionIR":
        return ExpressionIR(expr_type="variable", name=name, source_loc=source_loc)

    @staticmethod
    def binary_op(left: "ExpressionIR", op: str, right: "ExpressionIR", source_loc: str = None) -> "ExpressionIR":
        return ExpressionIR(expr_type="binary_op", op=op, left=left, right=right, source_loc=source_loc)

    @staticmethod
    def unary_op(op: str, operand: "ExpressionIR", source_loc: str = None) -> "ExpressionIR":
        return ExpressionIR(expr_type="unary_op", op=op, operand=operand, source_loc=source_loc)

    @staticmethod
    def call(func: "ExpressionIR", args: List["ExpressionIR"] = None, kwargs: Dict[str, "ExpressionIR"] = None, source_loc: str = None) -> "ExpressionIR":
        return ExpressionIR(
            expr_type="call", func=func,
            args=args or [], kwargs=kwargs or {},
            source_loc=source_loc,
        )

    @staticmethod
    def tensor_op(kind: TensorOpKind, args: List["ExpressionIR"] = None, kwargs: Dict[str, "ExpressionIR"] = None, source_loc: str = None) -> "ExpressionIR":
        return ExpressionIR(
            expr_type="tensor_op", tensor_op_kind=kind,
            args=args or [], kwargs=kwargs or {},
            source_loc=source_loc,
        )

    @staticmethod
    def attribute(target: "ExpressionIR", attr: str, source_loc: str = None) -> "ExpressionIR":
        return ExpressionIR(expr_type="attribute", target=target, attr=attr, source_loc=source_loc)

    @staticmethod
    def subscript(target: "ExpressionIR", index: "ExpressionIR", source_loc: str = None) -> "ExpressionIR":
        return ExpressionIR(expr_type="subscript", target=target, index=index, source_loc=source_loc)


# ─── Statement IR ──────────────────────────────────────────────────────────────

@dataclass
class StatementIR:
    """IR node for a single statement."""

    stmt_type: str
    target: Optional[ExpressionIR] = None
    expression: Optional[ExpressionIR] = None
    value: Optional[ExpressionIR] = None
    condition: Optional[ExpressionIR] = None
    body: List["StatementIR"] = field(default_factory=list)
    orelse: List["StatementIR"] = field(default_factory=list)
    iter_var: Optional[ExpressionIR] = None
    iterable: Optional[ExpressionIR] = None
    decorator_name: str = ""
    decorator_args: List[ExpressionIR] = field(default_factory=list)
    decorator_kwargs: Dict[str, ExpressionIR] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)
    source_loc: Optional[str] = None


# ─── Function IR ───────────────────────────────────────────────────────────────

@dataclass
class ParameterIR:
    """A function parameter."""

    name: str
    type: Optional[TypeAnnotation] = None
    default: Optional[ExpressionIR] = None


@dataclass
class FunctionSignature:
    """Function signature with parameter and return type info."""

    name: str
    parameters: List[ParameterIR] = field(default_factory=list)
    return_type: Optional[TypeAnnotation] = None
    decorators: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FunctionIR:
    """IR node for a function definition."""

    signature: FunctionSignature
    body: List[StatementIR] = field(default_factory=list)
    tensor_operations: List[ExpressionIR] = field(default_factory=list)
    nested_functions: List["FunctionIR"] = field(default_factory=list)
    source_loc: Optional[str] = None
    is_method: bool = False
    is_async: bool = False


# ─── Class IR ──────────────────────────────────────────────────────────────────

@dataclass
class ClassIR:
    """IR node for a class definition."""

    name: str
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionIR] = field(default_factory=list)
    class_variables: List[StatementIR] = field(default_factory=list)
    decorators: List[Dict[str, Any]] = field(default_factory=list)
    source_loc: Optional[str] = None


# ─── Loop & Conditional IR ────────────────────────────────────────────────────

@dataclass
class LoopIR:
    """Represents a loop with its bounds and body."""

    loop_type: str
    variable: Optional[ExpressionIR] = None
    iterable: Optional[ExpressionIR] = None
    condition: Optional[ExpressionIR] = None
    body: List[StatementIR] = field(default_factory=list)
    invariant: Optional[str] = None
    source_loc: Optional[str] = None


@dataclass
class ConditionalIR:
    """Represents a conditional branch."""

    condition: ExpressionIR
    then_branch: List[StatementIR] = field(default_factory=list)
    else_branch: List[StatementIR] = field(default_factory=list)
    source_loc: Optional[str] = None


# ─── Module-Level IR ───────────────────────────────────────────────────────────

@dataclass
class NormalizedIR:
    """Top-level IR for a Python module: output of AST extraction, input to
    the abstract interpreter and spec ingestion phases."""

    module_name: str = ""
    imports: List[str] = field(default_factory=list)
    functions: List[FunctionIR] = field(default_factory=list)
    classes: List[ClassIR] = field(default_factory=list)
    loops: List[LoopIR] = field(default_factory=list)
    conditionals: List[ConditionalIR] = field(default_factory=list)
    global_statements: List[StatementIR] = field(default_factory=list)
    source_path: Optional[str] = None
    python_version: Optional[str] = None
    warnings: List[ParserWarning] = field(default_factory=list)

    @property
    def all_functions(self) -> List[FunctionIR]:
        """Get all functions including methods."""
        result = list(self.functions)
        for cls in self.classes:
            result.extend(cls.methods)
        return result

    def get_function(self, name: str) -> Optional[FunctionIR]:
        """Find a function by name, searching methods as well."""
        for f in self.functions:
            if f.signature.name == name:
                return f
        for cls in self.classes:
            for m in cls.methods:
                if m.signature.name == name:
                    return m
        return None

    def has_tensor_operations(self) -> bool:
        """Check if any function uses tensor operations."""
        return any(func.tensor_operations for func in self.all_functions)
