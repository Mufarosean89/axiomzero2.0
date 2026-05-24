"""Axiom Zero - Tensor Shape Analyzer

Performs symbolic shape analysis over tensor operations, tracking shapes
through the program and generating shape constraints for the proof state.
"""

import re
from typing import Any, Dict, List, Optional

from ast_extractor.ir import (
    FunctionIR, StatementIR, ExpressionIR, NormalizedIR, TensorOpKind,
)
from .abstract_domain import (
    AbstractState, AbstractValue, TypeDomain, TensorShape, ShapeDimension,
)
from .type_inference import TypeInferenceEngine


class TensorShapeAnalyzer:
    """Analyzes tensor shapes through the program.

    Tracks tensor shapes through operations, generates shape constraints,
    detects shape mismatches, and extracts shape facts for the proof state.
    """

    def __init__(self) -> None:
        self._type_engine = TypeInferenceEngine()

    def analyze(self, ir: NormalizedIR, state: AbstractState) -> None:
        """Run shape analysis on the entire IR, populating the state."""
        for func in ir.all_functions:
            self._analyze_function_shapes(func, state)
        for func in ir.all_functions:
            self._extract_shape_facts(func, state)

    def _analyze_function_shapes(self, func: FunctionIR, state: AbstractState) -> None:
        func_name = func.signature.name
        ops_meta: Dict[str, Any] = {}

        for i, op_expr in enumerate(func.tensor_operations):
            op_info = self._analyze_tensor_op_shape(op_expr, state)
            if op_info:
                ops_meta[f"{func_name}_op_{i}"] = op_info

        if ops_meta:
            state.tensor_ops_metadata[func_name] = ops_meta

    def _analyze_tensor_op_shape(self, expr: ExpressionIR, state: AbstractState) -> Optional[Dict[str, Any]]:
        info: Dict[str, Any] = {
            "kind": expr.tensor_op_kind.name,
            "input_shapes": [],
            "output_shape": None,
            "constraints": [],
        }

        # Infer shapes of input arguments
        for arg in expr.args:
            if arg.expr_type == "variable":
                found = False
                for func_name, func_env in state.function_envs.items():
                    if arg.name in func_env:
                        val = func_env[arg.name]
                        if val.is_tensor and val.tensor_shape:
                            info["input_shapes"].append(self._shape_to_dict(val.tensor_shape))
                            found = True
                            break
                if not found:
                    info["input_shapes"].append({"rank": None})

        # Infer output shape based on operation kind
        kind = expr.tensor_op_kind

        if kind in (TensorOpKind.RELU, TensorOpKind.SIGMOID, TensorOpKind.TANH, TensorOpKind.DROPOUT):
            if info["input_shapes"]:
                info["output_shape"] = dict(info["input_shapes"][0])
                info["constraints"].append("output_shape == input_shape")

        elif kind == TensorOpKind.MATMUL and len(info["input_shapes"]) >= 2:
            s1, s2 = info["input_shapes"][0], info["input_shapes"][1]
            dims1, dims2 = s1.get("dimensions"), s2.get("dimensions")
            if dims1 and dims2:
                info["constraints"].append("dim[-1]_input1 == dim[-2]_input2")
                info["output_shape"] = {
                    "dimensions": [
                        {"value": dims1[0].get("value")},
                        {"value": dims2[-1].get("value")},
                    ]
                }

        elif kind in (TensorOpKind.RESHAPE, TensorOpKind.VIEW, TensorOpKind.FLATTEN):
            new_dims = []
            for arg in expr.args[1:]:
                if arg.expr_type == "constant" and isinstance(arg.value, int):
                    new_dims.append(ShapeDimension(value=arg.value))
                else:
                    new_dims.append(ShapeDimension(symbolic="N"))
            if new_dims:
                info["output_shape"] = self._shape_to_dict(TensorShape(dimensions=new_dims))
                info["constraints"].append("num_elements_preserved")

        return info if info.get("output_shape") else None

    def _extract_shape_facts(self, func: FunctionIR, state: AbstractState) -> None:
        env = state.function_envs.get(func.signature.name, {})
        for var_name, value in env.items():
            if value.is_tensor and value.tensor_shape:
                shape = value.tensor_shape
                for i, dim in enumerate(shape.dimensions):
                    if dim.is_concrete:
                        state.add_shape_fact(f"{var_name}.dim[{i}] == {dim.value}")
                    elif dim.is_symbolic:
                        state.add_shape_fact(f"{var_name}.dim[{i}] == {dim.symbolic}")
                if shape.rank is not None:
                    state.add_shape_fact(f"rank({var_name}) == {shape.rank}")

    def check_shape_compatibility(self, shapes: List[TensorShape], operation: str) -> bool:
        if len(shapes) < 2:
            return True

        if operation == "matmul":
            a, b = shapes[0], shapes[1]
            if a.rank is None or b.rank is None:
                return True
            if a.rank != 2 or b.rank != 2:
                return False
            return a.dimensions[-1].matches(b.dimensions[-2]) if a.dimensions and b.dimensions else True

        if operation in ("add", "concat"):
            return shapes[0].compatible_with(shapes[1])

        return True

    def _shape_to_dict(self, shape: TensorShape) -> Dict[str, Any]:
        return {
            "rank": shape.rank,
            "dimensions": [
                {"value": d.value if d.is_concrete else None, "symbolic": d.symbolic if d.is_symbolic else None}
                for d in shape.dimensions
            ],
        }

    def infer_shape_from_source(self, source_line: str) -> Optional[TensorShape]:
        """Infer a tensor shape from a source string like 'torch.randn(3, 224, 224)'."""
        match = re.search(r"(?:torch\.)?(?:zeros|ones|randn|rand|empty)\(([^)]+)\)", source_line)
        if not match:
            return None

        dims = []
        for arg in match.group(1).split(","):
            arg = arg.strip()
            try:
                dims.append(ShapeDimension(value=int(arg)))
            except ValueError:
                dims.append(ShapeDimension(symbolic="N"))
        return TensorShape(dimensions=dims) if dims else None
