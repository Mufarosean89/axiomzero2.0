"""Axiom Zero - Type Inference Engine

Performs forward type inference over the normalized IR. Tracks variable
types through assignments, function calls, and control flow. Handles
tensor types specially by coordinating with the shape analyzer.
"""

from typing import Any, Dict, List, Optional, Tuple

from ast_extractor.ir import (
    FunctionIR, StatementIR, ExpressionIR,
    TypeAnnotation, NormalizedIR, TensorOpKind,
)
from .abstract_domain import (
    AbstractState, AbstractValue, TypeDomain, TensorShape, ShapeDimension,
)

# Known function type signatures: (param_types, return_type)
_KnownSignatures: Dict[str, Tuple[List[TypeDomain], TypeDomain]] = {
    "len":    ([TypeDomain.TOP], TypeDomain.INT),
    "range":  ([TypeDomain.INT], TypeDomain.LIST),
    "int":    ([TypeDomain.TOP], TypeDomain.INT),
    "float":  ([TypeDomain.TOP], TypeDomain.FLOAT),
    "str":    ([TypeDomain.TOP], TypeDomain.STRING),
    "bool":   ([TypeDomain.TOP], TypeDomain.BOOL),
    "abs":    ([TypeDomain.INT], TypeDomain.INT),
    "min":    ([TypeDomain.TOP], TypeDomain.TOP),
    "max":    ([TypeDomain.TOP], TypeDomain.TOP),
    "sum":    ([TypeDomain.LIST], TypeDomain.TOP),
    "print":  ([TypeDomain.TOP], TypeDomain.BOTTOM),
}

_TORCH_CREATORS = {"torch.zeros", "torch.ones", "torch.randn", "zeros", "ones", "randn"}

# Tensor ops that preserve shape (element-wise)
_SHAPE_PRESERVING_OPS = {
    TensorOpKind.RELU, TensorOpKind.SIGMOID, TensorOpKind.TANH,
    TensorOpKind.DROPOUT, TensorOpKind.SOFTMAX,
}

# Tensor ops that reduce rank
_REDUCTION_OPS = {
    TensorOpKind.SUM, TensorOpKind.MEAN, TensorOpKind.MAX, TensorOpKind.MIN,
}

# Shape-changing ops
_SHAPE_CHANGING_OPS = {
    TensorOpKind.RESHAPE, TensorOpKind.VIEW, TensorOpKind.FLATTEN,
}

# Ops that combine tensors
_COMBINATION_OPS = {
    TensorOpKind.CONCAT, TensorOpKind.STACK,
}


class TypeInferenceEngine:
    """Performs forward type inference over the normalized IR.

    Tracks variable types through assignments, return types, element types
    of lists/dicts, tensor dtypes, and type constraints from operations.
    """

    def __init__(self) -> None:
        self._builtin_types = {
            "int": TypeAnnotation.int_type(),
            "float": TypeAnnotation.float_type(),
            "bool": TypeAnnotation.bool_type(),
            "str": TypeAnnotation("str"),
            "Tensor": TypeAnnotation.tensor_type(),
            "List": TypeAnnotation("List"),
            "Dict": TypeAnnotation("Dict"),
            "Optional": TypeAnnotation("Optional"),
        }

    def infer_function_types(
        self, func: FunctionIR, global_env: Dict[str, AbstractValue],
    ) -> Dict[str, AbstractValue]:
        """Run type inference on a single function, returning the local env."""
        env: Dict[str, AbstractValue] = dict(global_env)

        for param in func.signature.parameters:
            if param.type:
                env[param.name] = self._type_annotation_to_abstract(param.type)
            else:
                env[param.name] = AbstractValue(type_domain=TypeDomain.TOP)

        self._infer_statements(func.body, env)
        return env

    def _infer_statements(self, stmts: List[StatementIR], env: Dict[str, AbstractValue]) -> None:
        for stmt in stmts:
            self._infer_statement(stmt, env)

    def _infer_statement(self, stmt: StatementIR, env: Dict[str, AbstractValue]) -> None:
        if stmt.stmt_type == "assignment":
            self._infer_assignment(stmt, env)
        elif stmt.stmt_type == "return":
            if stmt.value:
                stmt.value.type = self._infer_expression(stmt.value, env)
        elif stmt.stmt_type == "if":
            if stmt.condition:
                stmt.condition.type = self._infer_expression(stmt.condition, env)
            then_env, else_env = dict(env), dict(env)
            self._infer_statements(stmt.body, then_env)
            self._infer_statements(stmt.orelse, else_env)
            for var in set(then_env) | set(else_env):
                env[var] = then_env.get(var, AbstractValue.bottom()).join(
                    else_env.get(var, AbstractValue.bottom())
                )
        elif stmt.stmt_type == "for":
            if stmt.iterable:
                self._infer_expression(stmt.iterable, env)
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

    def _infer_assignment(self, stmt: StatementIR, env: Dict[str, AbstractValue]) -> None:
        if stmt.expression:
            inferred = self._infer_expression(stmt.expression, env)
            stmt.expression.type = inferred

            if stmt.target and stmt.target.expr_type == "variable":
                if "type" in stmt.annotations:
                    env[stmt.target.name] = self._type_annotation_to_abstract(stmt.annotations["type"])
                else:
                    env[stmt.target.name] = inferred

        # Tuple unpacking: a, b = ...
        if stmt.target and stmt.target.expr_type == "list" and stmt.expression:
            for target, value in zip(stmt.target.elements, stmt.expression.elements):
                if target and target.expr_type == "variable" and value:
                    env[target.name] = value.type or AbstractValue.top()

    def _infer_expression(self, expr: ExpressionIR, env: Dict[str, AbstractValue]) -> AbstractValue:
        if expr is None:
            return AbstractValue.bottom()

        dispatchers = {
            "constant": lambda: self._infer_constant(expr),
            "variable": lambda: env.get(expr.name, AbstractValue.top()),
            "binary_op": lambda: self._infer_binary_op(expr, env),
            "unary_op": lambda: self._infer_unary_op(expr, env),
            "call": lambda: self._infer_call(expr, env),
            "tensor_op": lambda: self._infer_tensor_op(expr, env),
            "attribute": lambda: self._infer_attribute(expr, env),
            "subscript": lambda: self._infer_subscript(expr, env),
            "list": lambda: self._infer_list_literal(expr, env),
        }
        handler = dispatchers.get(expr.expr_type)
        return handler() if handler else AbstractValue.top()

    def _infer_list_literal(self, expr: ExpressionIR, env: Dict[str, AbstractValue]) -> AbstractValue:
        if not expr.elements:
            return AbstractValue.from_type(TypeDomain.LIST)
        joined = self._infer_expression(expr.elements[0], env)
        for e in expr.elements[1:]:
            joined = joined.join(self._infer_expression(e, env))
        return joined

    def _infer_constant(self, expr: ExpressionIR) -> AbstractValue:
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
        left = self._infer_expression(expr.left, env) if expr.left else AbstractValue.bottom()
        right = self._infer_expression(expr.right, env) if expr.right else AbstractValue.bottom()

        if left.is_tensor or right.is_tensor:
            result_shape = None
            if left.tensor_shape and right.tensor_shape:
                result_shape = left.tensor_shape.join(right.tensor_shape)
            return AbstractValue.from_tensor(result_shape) if result_shape else AbstractValue.from_type(TypeDomain.TENSOR)

        if expr.op in ("+", "-", "*", "/", "//", "%", "**"):
            joined = left.join(right)
            if joined.type_domain in (TypeDomain.INT, TypeDomain.FLOAT):
                return joined
            return AbstractValue.from_type(TypeDomain.FLOAT)

        if expr.op in ("==", "!=", "<", "<=", ">", ">=", "is", "is not", "in", "not in", "and", "or"):
            return AbstractValue.from_type(TypeDomain.BOOL)

        return AbstractValue.top()

    def _infer_unary_op(self, expr: ExpressionIR, env: Dict[str, AbstractValue]) -> AbstractValue:
        operand = self._infer_expression(expr.operand, env) if expr.operand else AbstractValue.bottom()
        if expr.op == "not":
            return AbstractValue.from_type(TypeDomain.BOOL)
        if expr.op in ("+", "-") and operand.type_domain in (TypeDomain.INT, TypeDomain.FLOAT):
            return operand
        return operand

    def _infer_call(self, expr: ExpressionIR, env: Dict[str, AbstractValue]) -> AbstractValue:
        func_name = ""
        if expr.func:
            if expr.func.expr_type == "variable":
                func_name = expr.func.name
            elif expr.func.expr_type == "attribute" and expr.func.target:
                func_name = f"{expr.func.target.name}.{expr.func.attr}"

        arg_types = [self._infer_expression(a, env) for a in expr.args]

        if func_name in _KnownSignatures:
            sig = _KnownSignatures[func_name]
            if sig[1] == TypeDomain.TOP:
                return arg_types[0] if arg_types else AbstractValue.top()
            return AbstractValue.from_type(sig[1])

        if "torch.tensor" in func_name or func_name == "tensor":
            return AbstractValue.from_tensor(TensorShape.unknown())

        if func_name in _TORCH_CREATORS:
            dims = []
            for a in expr.args:
                at = self._infer_expression(a, env)
                if at.concrete_value is not None and isinstance(at.concrete_value, int):
                    dims.append(ShapeDimension(value=at.concrete_value))
                else:
                    dims.append(ShapeDimension(symbolic="N"))
            return AbstractValue.from_tensor(TensorShape(dimensions=dims))

        return AbstractValue.top()

    def _infer_tensor_op(self, expr: ExpressionIR, env: Dict[str, AbstractValue]) -> AbstractValue:
        arg_types = [self._infer_expression(a, env) for a in expr.args]
        kind = expr.tensor_op_kind

        if kind in _SHAPE_PRESERVING_OPS:
            if arg_types and arg_types[0].tensor_shape:
                return AbstractValue.from_tensor(arg_types[0].tensor_shape)
            return AbstractValue.from_type(TypeDomain.TENSOR)

        if kind == TensorOpKind.MATMUL and len(arg_types) >= 2:
            s1, s2 = arg_types[0].tensor_shape, arg_types[1].tensor_shape
            if s1 and s2 and len(s1.dimensions) >= 2 and len(s2.dimensions) >= 2:
                return AbstractValue.from_tensor(TensorShape(dimensions=[s1.dimensions[0], s2.dimensions[-1]]))
            return AbstractValue.from_type(TypeDomain.TENSOR)

        if kind in _REDUCTION_OPS and arg_types and arg_types[0].tensor_shape:
            shape = arg_types[0].tensor_shape
            if shape.dimensions:
                reduced_dims = list(shape.dimensions[:-1]) if len(shape.dimensions) > 1 else [ShapeDimension(value=1)]
                return AbstractValue.from_tensor(TensorShape(dimensions=reduced_dims))
            return AbstractValue.from_type(TypeDomain.TENSOR)

        if kind in _SHAPE_CHANGING_OPS and len(expr.args) > 1:
            dims = []
            for a in expr.args[1:]:
                at = self._infer_expression(a, env)
                if at.concrete_value is not None and isinstance(at.concrete_value, int):
                    dims.append(ShapeDimension(value=at.concrete_value))
                else:
                    dims.append(ShapeDimension(symbolic="N"))
            return AbstractValue.from_tensor(TensorShape(dimensions=dims))

        if kind in _COMBINATION_OPS and arg_types and arg_types[0].tensor_shape:
            return AbstractValue.from_tensor(arg_types[0].tensor_shape)

        return AbstractValue.from_type(TypeDomain.TENSOR)

    def _infer_attribute(self, expr: ExpressionIR, env: Dict[str, AbstractValue]) -> AbstractValue:
        target = self._infer_expression(expr.target, env) if expr.target else AbstractValue.bottom()

        if expr.attr == "shape":
            return AbstractValue.from_type(TypeDomain.LIST)
        if expr.attr in ("size", "dtype", "device"):
            return AbstractValue.from_type(TypeDomain.INT)
        if expr.attr == "T" and target.tensor_shape:
            return AbstractValue.from_tensor(target.tensor_shape)
        if expr.attr == "item":
            return AbstractValue.from_type(TypeDomain.FLOAT)

        return AbstractValue.top()

    def _infer_subscript(self, expr: ExpressionIR, env: Dict[str, AbstractValue]) -> AbstractValue:
        target = self._infer_expression(expr.target, env) if expr.target else AbstractValue.bottom()
        self._infer_expression(expr.index, env) if expr.index else None

        if target.is_tensor and target.tensor_shape and target.tensor_shape.dimensions:
            new_dims = list(target.tensor_shape.dimensions[1:])
            if not new_dims:
                return AbstractValue.from_type(TypeDomain.FLOAT)
            return AbstractValue.from_tensor(TensorShape(dimensions=new_dims))

        if target.type_domain == TypeDomain.LIST:
            return AbstractValue.from_type(TypeDomain.TOP)

        return AbstractValue.top()

    def _type_annotation_to_abstract(self, ann: TypeAnnotation) -> AbstractValue:
        domain = TypeDomain.from_python_type(ann.name)

        if domain == TypeDomain.TENSOR:
            return AbstractValue.from_tensor(TensorShape.unknown())

        if domain == TypeDomain.LIST and ann.params:
            return AbstractValue(
                type_domain=TypeDomain.LIST,
                concrete_value={"element_type": ann.params[0].to_string()},
            )

        return AbstractValue.from_type(domain)
