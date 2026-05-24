"""
Axiom Zero - Tensor Shape Analyzer

Performs symbolic shape analysis over tensor operations.
Tracks tensor shapes through the program, handling:
- Shape inference from operations
- Symbolic dimension constraints
- Shape compatibility checking
- Shape facts extraction for the proof state
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
from ast_extractor.ir import (
    FunctionIR,
    StatementIR,
    ExpressionIR,
    NormalizedIR,
    TensorOpKind,
)
from .abstract_domain import (
    AbstractState,
    AbstractValue,
    TypeDomain,
    TensorShape,
    ShapeDimension,
)
from .type_inference import TypeInferenceEngine


class TensorShapeAnalyzer:
    """
    Analyzes tensor shapes through the program.
    
    Key responsibilities:
    - Track tensor shapes through operations
    - Generate shape constraints (e.g., "dim M == dim N")
    - Detect shape mismatches
    - Extract shape facts for the proof state
    """

    def __init__(self):
        self._type_engine = TypeInferenceEngine()

    def analyze(
        self,
        ir: NormalizedIR,
        state: AbstractState,
    ):
        """
        Run shape analysis on the entire IR.
        
        Args:
            ir: Normalized IR
            state: Abstract state to populate with shape facts
        """
        # Analyze each function's tensor operations
        for func in ir.all_functions:
            self._analyze_function_shapes(func, state)

        # Extract shape facts
        for func in ir.all_functions:
            self._extract_shape_facts(func, state)

    def _analyze_function_shapes(self, func: FunctionIR, state: AbstractState):
        """Analyze tensor shapes within a single function."""
        func_name = func.signature.name
        ops_metadata: Dict[str, Any] = {}

        for op_expr in func.tensor_operations:
            op_info = self._analyze_tensor_op_shape(op_expr, state)
            if op_info:
                op_key = f"{func_name}_op_{len(ops_metadata)}"
                ops_metadata[op_key] = op_info

        if ops_metadata:
            state.tensor_ops_metadata[func_name] = ops_metadata

    def _analyze_tensor_op_shape(
        self,
        expr: ExpressionIR,
        state: AbstractState,
    ) -> Optional[Dict[str, Any]]:
        """Analyze the shape behavior of a single tensor operation."""
        info: Dict[str, Any] = {
            "kind": expr.tensor_op_kind.name,
            "input_shapes": [],
            "output_shape": None,
            "constraints": [],
        }

        # Infer shapes of input arguments
        for arg in expr.args:
            if arg.expr_type == "variable":
                var_name = arg.name
                # Look up in function envs
                for func_name, func_env in state.function_envs.items():
                    if var_name in func_env:
                        val = func_env[var_name]
                        if val.is_tensor and val.tensor_shape:
                            info["input_shapes"].append(self._shape_to_dict(val.tensor_shape))
                            break
                else:
                    info["input_shapes"].append({"rank": None})

        # Infer output shape based on operation kind
        if expr.tensor_op_kind in (
            TensorOpKind.RELU, TensorOpKind.SIGMOID, TensorOpKind.TANH,
            TensorOpKind.DROPOUT,
        ):
            # Element-wise: preserves input shape
            if info["input_shapes"]:
                info["output_shape"] = dict(info["input_shapes"][0])
                info["constraints"].append("output_shape == input_shape")

        elif expr.tensor_op_kind == TensorOpKind.MATMUL:
            # (M, K) @ (K, N) → (M, N)
            if len(info["input_shapes"]) >= 2:
                s1 = info["input_shapes"][0]
                s2 = info["input_shapes"][1]
                if s1.get("dimensions") and s2.get("dimensions"):
                    info["constraints"].append("dim[-1]_input1 == dim[-2]_input2")
                    info["output_shape"] = {
                        "dimensions": [
                            {"value": s1["dimensions"][0].get("value")},
                            {"value": s2["dimensions"][-1].get("value")},
                        ]
                    }

        elif expr.tensor_op_kind == TensorOpKind.RESHAPE:
            # Output shape from the reshape arguments
            new_dims = []
            for arg in expr.args[1:]:
                if arg.expr_type == "constant" and isinstance(arg.value, int):
                    new_dims.append(ShapeDimension(value=arg.value))
                else:
                    new_dims.append(ShapeDimension(symbolic="N"))
            if new_dims:
                info["output_shape"] = self._shape_to_dict(TensorShape(dimensions=new_dims))
                info["constraints"].append("num_elements_preserved")

        return info if info["output_shape"] else None

    def _extract_shape_facts(self, func: FunctionIR, state: AbstractState):
        """Extract shape constraints as facts from a function."""
        func_name = func.signature.name
        env = state.function_envs.get(func_name, {})

        for var_name, value in env.items():
            if value.is_tensor and value.tensor_shape:
                shape = value.tensor_shape
                for i, dim in enumerate(shape.dimensions):
                    if dim.is_concrete:
                        state.add_shape_fact(f"{var_name}.dim[{i}] == {dim.value}")
                    elif dim.is_symbolic:
                        state.add_shape_fact(f"{var_name}.dim[{i}] == {dim.symbolic}")

                # Add rank fact
                if shape.rank is not None:
                    state.add_shape_fact(f"rank({var_name}) == {shape.rank}")

    def check_shape_compatibility(
        self,
        shapes: List[TensorShape],
        operation: str,
    ) -> bool:
        """Check if shapes are compatible for a given operation."""
        if not shapes:
            return True

        if operation == "matmul":
            if len(shapes) < 2:
                return True
            a, b = shapes[0], shapes[1]
            if a.rank is None or b.rank is None:
                return True
            if a.rank != 2 or b.rank != 2:
                return False
            # Check inner dimensions match
            if len(a.dimensions) >= 1 and len(b.dimensions) >= 1:
                return a.dimensions[-1].matches(b.dimensions[-2])
            return True

        if operation == "add":
            # Broadcasting: shapes must be compatible
            if len(shapes) < 2:
                return True
            return shapes[0].compatible_with(shapes[1])

        if operation == "concat":
            if len(shapes) < 2:
                return True
            # All dimensions except concat axis must match
            return True  # Simplified

        return True

    def _shape_to_dict(self, shape: TensorShape) -> Dict[str, Any]:
        """Convert a TensorShape to a serializable dict."""
        return {
            "rank": shape.rank,
            "dimensions": [
                {
                    "value": d.value if d.is_concrete else None,
                    "symbolic": d.symbolic if d.is_symbolic else None,
                }
                for d in shape.dimensions
            ]
        }

    def infer_shape_from_source(self, source_line: str) -> Optional[TensorShape]:
        """
        Infer a tensor shape from a source string like 'torch.randn(3, 224, 224)'.
        Useful for quick analysis without full parsing.
        """
        import re

        # Match shape arguments in function calls
        shape_pattern = r"(?:torch\.)?(?:zeros|ones|randn|rand|empty)\(([^)]+)\)"
        match = re.search(shape_pattern, source_line)
        if match:
            args_str = match.group(1)
            dims = []
            for arg in args_str.split(","):
                arg = arg.strip()
                try:
                    dims.append(ShapeDimension(value=int(arg)))
                except ValueError:
                    dims.append(ShapeDimension(symbolic="N"))
            if dims:
                return TensorShape(dimensions=dims)

        return None
