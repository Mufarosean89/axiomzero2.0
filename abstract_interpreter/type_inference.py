"""
Axiom Zero - Type Inference Engine

Performs forward type inference over the normalized IR.
Tracks variable types through assignments, function calls, and control flow.
Handles tensor types specially by coordinating with the shape analyzer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
from ast_extractor.ir import (
    FunctionIR,
    StatementIR,
    ExpressionIR,
    TypeAnnotation,
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


class TypeInferenceEngine:
    """
    Performs forward type inference over the normalized IR.
    
    Tracks:
    - Variable types through assignments
    - Return types of functions
    - Element types of lists/dicts
    - Tensor dtypes
    - Type constraints from operations
    """

    def __init__(self):
        self._builtin_types: Dict[str, TypeAnnotation] = {
            "int": TypeAnnotation.int_type(),
            "float": TypeAnnotation.float_type(),
            "bool": TypeAnnotation.bool_type(),
            "str": TypeAnnotation("str"),
            "Tensor": TypeAnnotation.tensor_type(),
            "List": TypeAnnotation("List"),
            "Dict": TypeAnnotation("Dict"),
            "Optional": TypeAnnotation("Optional"),
        }

        # Common type signatures for known functions
        self._known_signatures: Dict[str, Tuple[List[TypeDomain], TypeDomain]] = {
            "len": ([TypeDomain.TOP], TypeDomain.INT),
            "range": ([TypeDomain.INT], TypeDomain.LIST),
            "int": ([TypeDomain.TOP], TypeDomain.INT),
            "float": ([TypeDomain.TOP], TypeDomain.FLOAT),
            "str": ([TypeDomain.TOP], TypeDomain.STRING),
            "bool": ([TypeDomain.TOP], TypeDomain.BOOL),
            "abs": ([TypeDomain.INT], TypeDomain.INT),
            "min": ([TypeDomain.TOP], TypeDomain.TOP),
            "max": ([TypeDomain.TOP], TypeDomain.TOP),
            "sum": ([TypeDomain.LIST], TypeDomain.TOP),
            "print": ([TypeDomain.TOP], TypeDomain.BOTTOM),
        }

    def infer_function_types(
        self,
        func: FunctionIR,
        global_env: Dict[str, AbstractValue],
    ) -> Dict[str, AbstractValue]:
        """
        Run type inference on a single function.
        
        Args:
            func: FunctionIR to analyze
            global_env: Global variable environments
            
        Returns:
            Local variable environment with inferred types
        """
        local_env: Dict[str, AbstractValue] = dict(global_env)

        # Seed parameters with their declared types
        for param in func.signature.parameters:
            if param.type:
                abstract_type = self._type_annotation_to_abstract(param.type)
                local_env[param.name] = abstract_type
            else:
                # Unknown parameter type
                local_env[param.name] = AbstractValue(type_domain=TypeDomain.TOP)

        # Infer body types
        self._infer_statements(func.body, local_env)

        return local_env

    def _infer_statements(
        self,
        stmts: List[StatementIR],
        env: Dict[str, AbstractValue],
    ):
        """Infer types through a list of statements."""
        for stmt in stmts:
            self._infer_statement(stmt, env)

    def _infer_statement(
        self,
        stmt: StatementIR,
        env: Dict[str, AbstractValue],
    ):
        """Infer types through a single statement."""
        if stmt.stmt_type == "assignment":
            self._infer_assignment(stmt, env)
        elif stmt.stmt_type == "return":
            if stmt.value:
                stmt.value.type = self._infer_expression(stmt.value, env)
        elif stmt.stmt_type == "if":
            if stmt.condition:
                stmt.condition.type = self._infer_expression(stmt.condition, env)
            # Analyze both branches
            then_env = dict(env)
            self._infer_statements(stmt.body, then_env)
            else_env = dict(env)
            self._infer_statements(stmt.orelse, else_env)
            # Join environments
            for var in set(list(then_env.keys()) + list(else_env.keys())):
                t_val = then_env.get(var, AbstractValue.bottom())
                e_val = else_env.get(var, AbstractValue.bottom())
                env[var] = t_val.join(e_val)
        elif stmt.stmt_type == "for":
            if stmt.iterable:
                iter_type = self._infer_expression(stmt.iterable, env)
                # If iterating over a range, loop variable is int
                if stmt.iter_var and stmt.iter_var.expr_type == "variable":
                    env[stmt.iter_var.name] = AbstractValue.from_type(TypeDomain.INT)
            self._infer_statements(stmt.body, env)
        elif stmt.stmt_type == "while":
            if stmt.condition:
                stmt.condition.type = self._infer_expression(stmt.condition, env)
            self._infer_statements(stmt.body, env)
        elif stmt.stmt_type == "expression":
            if stmt.expression:
                stmt.expression.type = self._infer_expression(stmt.expression, env)

    def _infer_assignment(
        self,
        stmt: StatementIR,
        env: Dict[str, AbstractValue],
    ):
        """Infer types for an assignment statement."""
        if stmt.expression:
            inferred_type = self._infer_expression(stmt.expression, env)
            stmt.expression.type = inferred_type

            if stmt.target and stmt.target.expr_type == "variable":
                # Check for annotation override
                if "type" in stmt.annotations:
                    ann_type = self._type_annotation_to_abstract(stmt.annotations["type"])
                    env[stmt.target.name] = ann_type
                else:
                    env[stmt.target.name] = inferred_type

        # Handle tuple unpacking: a, b = ...
        if stmt.target and stmt.target.expr_type == "list":
            target_elements = stmt.target.elements
            if stmt.expression and stmt.expression.expr_type == "list":
                for target, value in zip(target_elements, stmt.expression.elements):
                    if target and target.expr_type == "variable" and value:
                        env[target.name] = value.type or AbstractValue.top()

    def _infer_expression(
        self,
        expr: ExpressionIR,
        env: Dict[str, AbstractValue],
    ) -> AbstractValue:
        """Infer the type of an expression."""
        if expr is None:
            return AbstractValue.bottom()

        if expr.expr_type == "constant":
            return self._infer_constant(expr)

        if expr.expr_type == "variable":
            return env.get(expr.name, AbstractValue.top())

        if expr.expr_type == "binary_op":
            return self._infer_binary_op(expr, env)

        if expr.expr_type == "unary_op":
            return self._infer_unary_op(expr, env)

        if expr.expr_type == "call":
            return self._infer_call(expr, env)

        if expr.expr_type == "tensor_op":
            return self._infer_tensor_op(expr, env)

        if expr.expr_type == "attribute":
            return self._infer_attribute(expr, env)

        if expr.expr_type == "subscript":
            return self._infer_subscript(expr, env)

        if expr.expr_type == "list":
            if expr.elements:
                elem_types = [self._infer_expression(e, env) for e in expr.elements]
                joined = elem_types[0]
                for t in elem_types[1:]:
                    joined = joined.join(t)
                return joined
            return AbstractValue.from_type(TypeDomain.LIST)

        return AbstractValue.top()

    def _infer_constant(self, expr: ExpressionIR) -> AbstractValue:
        """Infer the type of a constant."""
        value = expr.value
        if isinstance(value, bool):
            return AbstractValue.constant(value, TypeDomain.BOOL)
        if isinstance(value, int):
            return AbstractValue.constant(value, TypeDomain.INT)
        if isinstance(value, float):
            return AbstractValue.constant(value, TypeDomain.FLOAT)
        if isinstance(value, str):
            return AbstractValue.constant(value, TypeDomain.STRING)
        if value is None:
            return AbstractValue(type_domain=TypeDomain.BOTTOM)
        return AbstractValue.top()

    def _infer_binary_op(self, expr: ExpressionIR, env: Dict[str, AbstractValue]) -> AbstractValue:
        """Infer the type of a binary operation."""
        left_type = self._infer_expression(expr.left, env) if expr.left else AbstractValue.bottom()
        right_type = self._infer_expression(expr.right, env) if expr.right else AbstractValue.bottom()

        # Infer tensor operations on tensors
        if left_type.is_tensor or right_type.is_tensor:
            result_shape = None
            if left_type.tensor_shape and right_type.tensor_shape:
                result_shape = left_type.tensor_shape.join(right_type.tensor_shape)
            return AbstractValue.from_tensor(result_shape) if result_shape else AbstractValue.from_type(TypeDomain.TENSOR)

        # Arithmetic operations on numerics
        if expr.op in ("+", "-", "*", "/", "//", "%", "**"):
            joined = left_type.join(right_type)
            if joined.type_domain in (TypeDomain.INT, TypeDomain.FLOAT):
                return joined
            return AbstractValue.from_type(TypeDomain.FLOAT)

        # Comparison always yields bool
        if expr.op in ("==", "!=", "<", "<=", ">", ">=", "is", "is not", "in", "not in"):
            return AbstractValue.from_type(TypeDomain.BOOL)

        # Boolean operators
        if expr.op in ("and", "or"):
            return AbstractValue.from_type(TypeDomain.BOOL)

        return AbstractValue.top()

    def _infer_unary_op(self, expr: ExpressionIR, env: Dict[str, AbstractValue]) -> AbstractValue:
        """Infer the type of a unary operation."""
        operand_type = self._infer_expression(expr.operand, env) if expr.operand else AbstractValue.bottom()

        if expr.op == "not":
            return AbstractValue.from_type(TypeDomain.BOOL)
        if expr.op in ("+", "-"):
            return operand_type if operand_type.type_domain in (TypeDomain.INT, TypeDomain.FLOAT) else AbstractValue.top()

        return operand_type

    def _infer_call(self, expr: ExpressionIR, env: Dict[str, AbstractValue]) -> AbstractValue:
        """Infer the return type of a function call."""
        func_name = ""
        if expr.func and expr.func.expr_type == "variable":
            func_name = expr.func.name
        elif expr.func and expr.func.expr_type == "attribute":
            func_name = f"{expr.func.target.name}.{expr.func.attr}" if expr.func.target else expr.func.attr

        # Infer argument types
        arg_types = [self._infer_expression(a, env) for a in expr.args]

        # Check known signatures
        if func_name in self._known_signatures:
            sig = self._known_signatures[func_name]
            # Return type
            return_type = sig[1]
            if return_type == TypeDomain.TOP:
                # Infer from argument
                return arg_types[0] if arg_types else AbstractValue.top()
            return AbstractValue.from_type(return_type)

        # torch.tensor([...]) — creates a tensor
        if "torch.tensor" in func_name or func_name == "tensor":
            return AbstractValue.from_tensor(TensorShape.unknown())

        # torch.zeros, torch.ones, torch.randn
        if func_name in ("torch.zeros", "torch.ones", "torch.randn", "zeros", "ones", "randn"):
            shape_dims = []
            for a in expr.args:
                arg_type = self._infer_expression(a, env)
                if arg_type.concrete_value is not None and isinstance(arg_type.concrete_value, int):
                    shape_dims.append(ShapeDimension(value=arg_type.concrete_value))
                else:
                    shape_dims.append(ShapeDimension(symbolic="N"))
            return AbstractValue.from_tensor(TensorShape(dimensions=shape_dims))

        return AbstractValue.top()

    def _infer_tensor_op(self, expr: ExpressionIR, env: Dict[str, AbstractValue]) -> AbstractValue:
        """Infer the result type/shape of a tensor operation."""
        arg_types = [self._infer_expression(a, env) for a in expr.args]

        if expr.tensor_op_kind in (
            TensorOpKind.RELU, TensorOpKind.SIGMOID, TensorOpKind.TANH,
            TensorOpKind.DROPOUT, TensorOpKind.SOFTMAX,
        ):
            # Element-wise: preserves shape
            if arg_types and arg_types[0].tensor_shape:
                return AbstractValue.from_tensor(arg_types[0].tensor_shape)
            return AbstractValue.from_type(TypeDomain.TENSOR)

        if expr.tensor_op_kind == TensorOpKind.MATMUL:
            # (M, K) @ (K, N) → (M, N)
            if len(arg_types) >= 2:
                s1 = arg_types[0].tensor_shape
                s2 = arg_types[1].tensor_shape
                if s1 and s2 and len(s1.dimensions) >= 2 and len(s2.dimensions) >= 2:
                    result_dims = [s1.dimensions[0], s2.dimensions[-1]]
                    return AbstractValue.from_tensor(TensorShape(dimensions=result_dims))
            return AbstractValue.from_type(TypeDomain.TENSOR)

        if expr.tensor_op_kind in (TensorOpKind.SUM, TensorOpKind.MEAN, TensorOpKind.MAX, TensorOpKind.MIN):
            # Reduction ops: rank reduces
            if arg_types and arg_types[0].tensor_shape:
                shape = arg_types[0].tensor_shape
                if shape.dimensions:
                    reduced = TensorShape(dimensions=list(shape.dimensions[:-1])) if len(shape.dimensions) > 1 else TensorShape(dimensions=[ShapeDimension(value=1)])
                    return AbstractValue.from_tensor(reduced)
            return AbstractValue.from_type(TypeDomain.TENSOR)

        if expr.tensor_op_kind in (TensorOpKind.RESHAPE, TensorOpKind.VIEW, TensorOpKind.FLATTEN):
            # Shape-changing ops: shape becomes the specified one
            if expr.args and len(expr.args) > 1:
                # New shape specified as additional args
                new_dims = []
                for a in expr.args[1:]:
                    arg_type = self._infer_expression(a, env)
                    if arg_type.concrete_value is not None and isinstance(arg_type.concrete_value, int):
                        new_dims.append(ShapeDimension(value=arg_type.concrete_value))
                    else:
                        new_dims.append(ShapeDimension(symbolic="N"))
                return AbstractValue.from_tensor(TensorShape(dimensions=new_dims))
            return AbstractValue.from_type(TypeDomain.TENSOR)

        if expr.tensor_op_kind in (TensorOpKind.CONCAT, TensorOpKind.STACK):
            # Concatenation/stacking
            if arg_types and arg_types[0].tensor_shape:
                return AbstractValue.from_tensor(arg_types[0].tensor_shape)
            return AbstractValue.from_type(TypeDomain.TENSOR)

        return AbstractValue.from_type(TypeDomain.TENSOR)

    def _infer_attribute(self, expr: ExpressionIR, env: Dict[str, AbstractValue]) -> AbstractValue:
        """Infer the type of an attribute access."""
        target_type = self._infer_expression(expr.target, env) if expr.target else AbstractValue.bottom()

        # Tensor.shape or Tensor.size()
        if expr.attr in ("shape", "size", "dtype", "device"):
            if expr.attr == "shape":
                return AbstractValue.from_type(TypeDomain.LIST)
            return AbstractValue.from_type(TypeDomain.INT)

        # .T for transpose
        if expr.attr == "T":
            if target_type.tensor_shape:
                return AbstractValue.from_tensor(target_type.tensor_shape)
            return AbstractValue.from_type(TypeDomain.TENSOR)

        # .item() on a tensor
        if expr.attr == "item":
            return AbstractValue.from_type(TypeDomain.FLOAT)

        return AbstractValue.top()

    def _infer_subscript(self, expr: ExpressionIR, env: Dict[str, AbstractValue]) -> AbstractValue:
        """Infer the type of a subscript/index operation."""
        target_type = self._infer_expression(expr.target, env) if expr.target else AbstractValue.bottom()
        index_type = self._infer_expression(expr.index, env) if expr.index else AbstractValue.bottom()

        # Tensor indexing: t[i] → reduces rank
        if target_type.is_tensor and target_type.tensor_shape:
            shape = target_type.tensor_shape
            if shape.dimensions:
                new_dims = list(shape.dimensions[1:])
                if not new_dims:
                    # Scalar from tensor
                    return AbstractValue.from_type(TypeDomain.FLOAT)
                return AbstractValue.from_tensor(TensorShape(dimensions=new_dims))
            return AbstractValue.from_type(TypeDomain.TENSOR)

        # List indexing: x[i] → element type
        if target_type.type_domain == TypeDomain.LIST:
            return AbstractValue.from_type(TypeDomain.TOP)

        return AbstractValue.top()

    def _type_annotation_to_abstract(self, ann: TypeAnnotation) -> AbstractValue:
        """Convert a parsed type annotation to an AbstractValue."""
        domain = TypeDomain.from_python_type(ann.name)

        if domain == TypeDomain.TENSOR:
            # Try to extract shape from type annotation like Tensor[float32, [B, 768]]
            return AbstractValue.from_tensor(TensorShape.unknown())

        if domain == TypeDomain.LIST and ann.params:
            elem = ann.params[0]
            return AbstractValue(
                type_domain=TypeDomain.LIST,
                concrete_value={"element_type": elem.to_string()},
            )

        return AbstractValue.from_type(domain)
