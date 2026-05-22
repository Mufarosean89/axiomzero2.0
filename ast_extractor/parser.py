"""
Axiom Zero - AST Extractor Parser

Parses Python source code into Axiom Zero's normalized intermediate representation (IR).
Uses Python's built-in `ast` module to parse source and walks the AST to produce
FunctionIR, ClassIR, LoopIR, ConditionalIR, and StatementIR/ExpressionIR nodes.

Special attention is paid to:
- Tensor operations (torch.*, F.*, nn.*)
- Type annotations
- Decorators (especially @requires, @ensures, @invariant)
- Loop structures with bounds
"""

from __future__ import annotations

import ast
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .ir import (
    NormalizedIR,
    FunctionIR,
    FunctionSignature,
    ParameterIR,
    ClassIR,
    LoopIR,
    ConditionalIR,
    StatementIR,
    ExpressionIR,
    TypeAnnotation,
    TensorOpKind,
    ParserConfig,
    ParserWarning,
)


# ─── Known Tensor Operations ──────────────────────────────────────────────────

TENSOR_MODULES: Set[str] = {"torch", "torch.nn", "torch.nn.functional", "F", "nn"}
TENSOR_FUNCTIONS: Dict[str, TensorOpKind] = {
    "torch.add": TensorOpKind.ADD,
    "torch.sub": TensorOpKind.SUB,
    "torch.mul": TensorOpKind.MUL,
    "torch.div": TensorOpKind.DIV,
    "torch.matmul": TensorOpKind.MATMUL,
    "torch.mm": TensorOpKind.MATMUL,
    "torch.relu": TensorOpKind.RELU,
    "torch.sigmoid": TensorOpKind.SIGMOID,
    "torch.tanh": TensorOpKind.TANH,
    "torch.softmax": TensorOpKind.SOFTMAX,
    "torch.reshape": TensorOpKind.RESHAPE,
    "torch.view": TensorOpKind.VIEW,
    "torch.transpose": TensorOpKind.TRANSPOSE,
    "torch.permute": TensorOpKind.PERMUTE,
    "torch.cat": TensorOpKind.CONCAT,
    "torch.stack": TensorOpKind.STACK,
    "torch.sum": TensorOpKind.SUM,
    "torch.mean": TensorOpKind.MEAN,
    "torch.max": TensorOpKind.MAX,
    "torch.min": TensorOpKind.MIN,
    "torch.dropout": TensorOpKind.DROPOUT,
    "torch.flatten": TensorOpKind.FLATTEN,
    "torch.nn.functional.relu": TensorOpKind.RELU,
    "torch.nn.functional.sigmoid": TensorOpKind.SIGMOID,
    "torch.nn.functional.tanh": TensorOpKind.TANH,
    "torch.nn.functional.softmax": TensorOpKind.SOFTMAX,
    "torch.nn.functional.dropout": TensorOpKind.DROPOUT,
    "F.relu": TensorOpKind.RELU,
    "F.sigmoid": TensorOpKind.SIGMOID,
    "F.tanh": TensorOpKind.TANH,
    "F.softmax": TensorOpKind.SOFTMAX,
    "F.dropout": TensorOpKind.DROPOUT,
}

NN_MODULE_OPS: Dict[str, TensorOpKind] = {
    "Linear": TensorOpKind.LINEAR,
    "Conv2d": TensorOpKind.CONV2D,
    "MaxPool2d": TensorOpKind.MAX_POOL2D,
    "BatchNorm2d": TensorOpKind.BATCH_NORM,
    "LayerNorm": TensorOpKind.LAYER_NORM,
    "Embedding": TensorOpKind.EMBEDDING,
    "Dropout": TensorOpKind.DROPOUT,
    "ReLU": TensorOpKind.RELU,
    "Sigmoid": TensorOpKind.SIGMOID,
    "Tanh": TensorOpKind.TANH,
    "Flatten": TensorOpKind.FLATTEN,
}


# ─── Known Spec Decorators ───────────────────────────────────────────────────

SPEC_DECORATORS: Set[str] = {"requires", "ensures", "invariant", "precondition", "postcondition"}


class ParserError(Exception):
    """Error raised during AST parsing."""
    pass


# ─── Known Mutating Methods ────────────────────────────────────────────────────

_MUTATING_METHODS: Set[str] = {
    # List methods
    "append", "extend", "insert", "remove", "pop", "sort", "reverse", "clear",
    # Dict methods
    "update", "setdefault", "popitem",
    # Set methods
    "add", "discard", "difference_update", "intersection_update", "symmetric_difference_update",
}


# ─── Known Unsupported Constructs ─────────────────────────────────────────────

# Statement types that are known to be unsupported (silently dropped otherwise).
# Maps ast node type → (construct_name, explanation).
_UNSUPPORTED_STATEMENTS: Dict[Any, tuple[str, str]] = {
    ast.Try: ("try/except/finally", "Exception handling is not supported. Use @requires/@ensures instead."),
    ast.Delete: ("del", "Delete statements are not supported."),
    ast.Global: ("global", "Global declarations are not supported."),
    ast.Nonlocal: ("nonlocal", "Nonlocal declarations are not supported."),
    ast.AsyncFor: ("async for", "Async iteration is not supported."),
    ast.AsyncWith: ("async with", "Async context managers are not supported."),
}

# Register Python-version-dependent unsupported statements safely
_TRYSTAR = getattr(ast, "TryStar", None)
if _TRYSTAR is not None:
    _UNSUPPORTED_STATEMENTS[_TRYSTAR] = ("try*", "Star import exception handling is not supported.")

_MATCH = getattr(ast, "Match", None)
if _MATCH is not None:
    _UNSUPPORTED_STATEMENTS[_MATCH] = ("match/case", "Pattern matching is not supported.")

# Expression types that are known to be unsupported (silently dropped otherwise).
_UNSUPPORTED_EXPRESSIONS: Dict[Any, tuple[str, str]] = {
    ast.GeneratorExp: ("generator expression", "Generator comprehensions are not supported; use list comprehension instead."),
    ast.SetComp: ("set comprehension", "Set comprehensions are not supported."),
    ast.DictComp: ("dict comprehension", "Dict comprehensions are not supported."),
    ast.Yield: ("yield", "Generators/yield are not supported."),
    ast.YieldFrom: ("yield from", "Generators/yield from are not supported."),
    ast.Await: ("await", "Async/await expressions are not supported."),
    ast.Starred: ("starred expression *args", "Starred unpacking expressions are not supported."),
    ast.JoinedStr: ("f-string", "f-strings with interpolated expressions are not supported; use string concatenation instead."),
    ast.FormattedValue: ("f-string expression", "f-string expressions are not supported; use string concatenation instead."),
}

# Register Python-version-dependent unsupported expressions safely
_NAMEDEXPR = getattr(ast, "NamedExpr", None)
if _NAMEDEXPR is not None:
    _UNSUPPORTED_EXPRESSIONS[_NAMEDEXPR] = ("walrus operator :=", "Assignment expressions (:=) are not supported.")


class Parser:
    """
    Parses Python source code into Axiom Zero's NormalizedIR.
    
    Usage:
        parser = Parser()
        ir = parser.parse("path/to/file.py")
        # or
        ir = parser.parse_source("def foo(x): return x + 1", module_name="example")
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        self._config = config or ParserConfig()
        self._current_source_path: Optional[str] = None

    def parse(self, source_path: str) -> NormalizedIR:
        """
        Parse a Python source file into NormalizedIR.
        
        Args:
            source_path: Path to the .py file
            
        Returns:
            NormalizedIR representation of the module
        """
        self._current_source_path = source_path
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                source = f.read()
        except FileNotFoundError:
            raise ParserError(f"File not found: {source_path}")
        return self.parse_source(source, module_name=source_path)

    def parse_source(self, source: str, module_name: str = "<string>") -> NormalizedIR:
        """
        Parse Python source string into NormalizedIR.
        
        Args:
            source: Python source code as a string
            module_name: Name for the module (used in error messages)
            
        Returns:
            NormalizedIR representation
        """
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise ParserError(f"Syntax error in {module_name}: {e}")

        self._current_ir = NormalizedIR(
            module_name=module_name,
            source_path=self._current_source_path,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        )

        self._walk_module(tree, self._current_ir)
        return self._current_ir

    def _warn_unsupported(self, node: ast.AST, construct: str, message: str):
        """
        Record a warning (or raise an error in strict mode) for an unsupported construct.
        
        Args:
            node: The AST node that triggered the warning
            construct: Name of the unsupported construct
            message: Explanation of why it's unsupported
            
        Raises:
            ParserError: If strict_mode is enabled
        """
        location = self._make_loc(node)
        warning = ParserWarning(construct=construct, location=location, message=message)

        if self._config.strict_mode:
            raise ParserError(str(warning))

        if self._config.collect_warnings:
            self._current_ir.warnings.append(warning)

    def _make_loc(self, node: ast.AST) -> str:
        """Create a source location string from an AST node."""
        if hasattr(node, "lineno"):
            path = self._current_source_path or "<unknown>"
            return f"{path}:{node.lineno}"
        return ""

    # ─── Module Level ──────────────────────────────────────────────────────

    def _walk_module(self, node: ast.Module, ir: NormalizedIR):
        """Walk the top-level module body."""
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) or isinstance(stmt, ast.AsyncFunctionDef):
                func_ir = self._parse_function(stmt)
                if func_ir:
                    ir.functions.append(func_ir)
            elif isinstance(stmt, ast.ClassDef):
                class_ir = self._parse_class(stmt)
                if class_ir:
                    ir.classes.append(class_ir)
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    ir.imports.append(alias.name)
            elif isinstance(stmt, ast.ImportFrom):
                module = stmt.module or ""
                for alias in stmt.names:
                    full_import = f"{module}.{alias.name}" if module else alias.name
                    ir.imports.append(full_import)
            else:
                stmt_ir = self._parse_statement(stmt)
                if stmt_ir:
                    ir.global_statements.append(stmt_ir)

    # ─── Functions ─────────────────────────────────────────────────────────

    def _parse_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[FunctionIR]:
        """Parse a function definition into FunctionIR."""
        loc = self._make_loc(node)

        # Parse decorators
        decorators = []
        for dec in node.decorator_list:
            dec_info = self._parse_decorator(dec)
            if dec_info:
                decorators.append(dec_info)

        # Parse signature
        params = []
        for arg in node.args.args:
            param = ParameterIR(
                name=arg.arg,
                type=self._parse_type_annotation(arg.annotation) if arg.annotation else None,
            )
            params.append(param)

        # Handle *args, **kwargs
        if node.args.vararg:
            params.append(ParameterIR(name=f"*{node.args.vararg.arg}"))
        if node.args.kwonlyargs:
            for arg in node.args.kwonlyargs:
                params.append(ParameterIR(name=arg.arg, type=self._parse_type_annotation(arg.annotation) if arg.annotation else None))
        if node.args.kwarg:
            params.append(ParameterIR(name=f"**{node.args.kwarg.arg}"))

        sig = FunctionSignature(
            name=node.name,
            parameters=params,
            return_type=self._parse_type_annotation(node.returns) if node.returns else None,
            decorators=decorators,
        )

        # Parse body
        body = []
        tensor_ops = []
        for stmt in node.body:
            stmt_ir = self._parse_statement(stmt)
            if stmt_ir:
                body.append(stmt_ir)
                self._collect_tensor_ops(stmt_ir, tensor_ops)

        # Parse nested functions
        nested = []
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested_func = self._parse_function(stmt)
                if nested_func:
                    nested.append(nested_func)

        # Check for recursion (self-recursive calls — termination can't be verified)
        if self._ast_has_recursive_call(node.body, node.name):
            self._warn_unsupported(
                node,
                "recursion",
                f"Recursive call to '{node.name}' — termination cannot be verified automatically.",
            )

        return FunctionIR(
            signature=sig,
            body=body,
            tensor_operations=tensor_ops,
            nested_functions=nested,
            source_loc=loc,
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )

    def _parse_decorator(self, node: ast.expr) -> Optional[Dict[str, Any]]:
        """Parse a decorator expression into a structured dict."""
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                name = f"{self._expr_to_name(func.target)}.{func.attr}"
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                return None

            args = [self._parse_expression(a) for a in node.args]
            kwargs = {kw.arg: self._parse_expression(kw.value) for kw in node.keywords if kw.arg}

            info: Dict[str, Any] = {
                "name": name,
                "args": [a for a in args if a is not None],
                "kwargs": {k: v for k, v in kwargs.items() if v is not None},
            }

            # Handle string arguments specially (spec predicates)
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    info.setdefault("predicates", []).append(a.value)
                elif isinstance(a, ast.Str):
                    info.setdefault("predicates", []).append(a.s)

            return info

        elif isinstance(node, (ast.Name, ast.Attribute)):
            name = (
                node.id
                if isinstance(node, ast.Name)
                else f"{self._expr_to_name(node.target)}.{node.attr}"
            )
            return {"name": name, "args": [], "kwargs": {}}

        return None

    # ─── Classes ───────────────────────────────────────────────────────────

    def _parse_class(self, node: ast.ClassDef) -> Optional[ClassIR]:
        """Parse a class definition into ClassIR."""
        loc = self._make_loc(node)

        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._expr_to_name(base.target)}.{base.attr}")

        decorators = []
        for dec in node.decorator_list:
            dec_info = self._parse_decorator(dec)
            if dec_info:
                decorators.append(dec_info)

        methods = []
        class_vars = []
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_ir = self._parse_function(stmt)
                if func_ir:
                    func_ir.is_method = True
                    methods.append(func_ir)
            else:
                stmt_ir = self._parse_statement(stmt)
                if stmt_ir:
                    class_vars.append(stmt_ir)

        return ClassIR(
            name=node.name,
            bases=bases,
            methods=methods,
            class_variables=class_vars,
            decorators=decorators,
            source_loc=loc,
        )

    # ─── Statements ────────────────────────────────────────────────────────

    def _parse_statement(self, node: ast.stmt) -> Optional[StatementIR]:
        """Parse a statement into StatementIR."""
        loc = self._make_loc(node)

        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Handled at module level, skip here
            return None

        if isinstance(node, ast.Assign):
            # Check for mutable state via attribute/subscript assignment
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    self._warn_unsupported(
                        target,
                        "attribute assignment",
                        "Assigning to an object attribute (obj.attr = val) mutates state and is not supported. Use a pure functional pattern.",
                    )
                if isinstance(target, ast.Subscript):
                    self._warn_unsupported(
                        target,
                        "subscript assignment",
                        "Assigning to a subscript (list[i] = val) mutates state and is not supported. Use a pure functional pattern.",
                    )
            # Multiple targets: a = b = expr
            if len(node.targets) == 1:
                target = self._parse_expression(node.targets[0])
                expr = self._parse_expression(node.value)
                return StatementIR(stmt_type="assignment", target=target, expression=expr, source_loc=loc)
            else:
                # Handle multi-target assignment by returning only the last assignment
                target = self._parse_expression(node.targets[-1])
                expr = self._parse_expression(node.value)
                return StatementIR(stmt_type="assignment", target=target, expression=expr, source_loc=loc)

        if isinstance(node, ast.AnnAssign):
            target = self._parse_expression(node.target)
            expr = self._parse_expression(node.value) if node.value else None
            ann = self._parse_type_annotation(node.annotation)
            stmt = StatementIR(stmt_type="assignment", target=target, expression=expr, source_loc=loc)
            if ann:
                stmt.annotations["type"] = ann
            return stmt

        if isinstance(node, ast.AugAssign):
            target = self._parse_expression(node.target)
            op_map = {
                ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
                ast.LShift: "<<", ast.RShift: ">>", ast.BitOr: "|",
                ast.BitXor: "^", ast.BitAnd: "&",
            }
            op = op_map.get(type(node.op), "?")
            expr = self._parse_expression(node.value)
            # Convert a += 1 to a = a + 1
            full_expr = ExpressionIR.binary_op(target, op, expr, loc)
            return StatementIR(stmt_type="assignment", target=target, expression=full_expr, source_loc=loc)

        if isinstance(node, ast.Return):
            value = self._parse_expression(node.value) if node.value else None
            return StatementIR(stmt_type="return", value=value, source_loc=loc)

        if isinstance(node, ast.Expr):
            expr = self._parse_expression(node.value)
            return StatementIR(stmt_type="expression", expression=expr, source_loc=loc)

        if isinstance(node, ast.If):
            condition = self._parse_expression(node.test)
            body = [s for s in (self._parse_statement(s) for s in node.body) if s is not None]
            orelse = [s for s in (self._parse_statement(s) for s in node.orelse) if s is not None]
            return StatementIR(stmt_type="if", condition=condition, body=body, orelse=orelse, source_loc=loc)

        if isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                var = ExpressionIR.variable(node.target.id, loc)
            else:
                var = self._parse_expression(node.target)
            iterable = self._parse_expression(node.iter)
            body = [s for s in (self._parse_statement(s) for s in node.body) if s is not None]
            orelse = [s for s in (self._parse_statement(s) for s in node.orelse) if s is not None]
            return StatementIR(stmt_type="for", iter_var=var, iterable=iterable, body=body, orelse=orelse, source_loc=loc)

        if isinstance(node, ast.While):
            # Check for potentially non-terminating loop (while True with no break)
            is_while_true = (
                isinstance(node.test, ast.Constant) and node.test.value is True
            )
            if is_while_true and not self._ast_has_break(node.body):
                self._warn_unsupported(
                    node,
                    "infinite loop",
                    "while True with no break — loop termination cannot be verified.",
                )
            condition = self._parse_expression(node.test)
            body = [s for s in (self._parse_statement(s) for s in node.body) if s is not None]
            orelse = [s for s in (self._parse_statement(s) for s in node.orelse) if s is not None]
            return StatementIR(stmt_type="while", condition=condition, body=body, orelse=orelse, source_loc=loc)

        if isinstance(node, ast.Assert):
            test = self._parse_expression(node.test)
            msg = self._parse_expression(node.msg) if node.msg else None
            return StatementIR(stmt_type="assert", condition=test, expression=msg, source_loc=loc)

        if isinstance(node, ast.Pass):
            return StatementIR(stmt_type="pass", source_loc=loc)

        if isinstance(node, ast.Break):
            return StatementIR(stmt_type="break", source_loc=loc)

        if isinstance(node, ast.Continue):
            return StatementIR(stmt_type="continue", source_loc=loc)

        if isinstance(node, ast.Raise):
            exc = self._parse_expression(node.exc) if node.exc else None
            return StatementIR(stmt_type="raise", expression=exc, source_loc=loc)

        if isinstance(node, ast.With):
            items = []
            for item in node.items:
                items.append(self._parse_expression(item.context_expr))
            body = [s for s in (self._parse_statement(s) for s in node.body) if s is not None]
            return StatementIR(stmt_type="with", expression=items[0] if items else None, body=body, source_loc=loc)

        # ── Detect known unsupported statements ───────────────────────────
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._warn_unsupported(
                node,
                "import",
                "Import statements inside functions are not supported. Move them to module level.",
            )
            return None

        if isinstance(node, ast.ClassDef):
            self._warn_unsupported(
                node,
                "class",
                "Nested class definitions inside functions are not supported.",
            )
            return None

        # Check the unsupported statement registry
        unsupported = _UNSUPPORTED_STATEMENTS.get(type(node))
        if unsupported:
            construct, message = unsupported
            self._warn_unsupported(node, construct, message)
            return None

        return None

    # ─── Expressions ──────────────────────────────────────────────────────

    def _parse_expression(self, node: ast.expr) -> Optional[ExpressionIR]:
        """Parse an expression into ExpressionIR."""
        loc = self._make_loc(node)

        if isinstance(node, ast.Constant):
            return ExpressionIR.constant(node.value, loc)

        if isinstance(node, ast.Name):
            return ExpressionIR.variable(node.id, loc)

        if isinstance(node, ast.Num):
            return ExpressionIR.constant(node.n, loc)

        if isinstance(node, ast.Str):
            return ExpressionIR.constant(node.s, loc)

        if isinstance(node, ast.NameConstant):
            return ExpressionIR.constant(node.value, loc)

        if isinstance(node, ast.BinOp):
            left = self._parse_expression(node.left)
            right = self._parse_expression(node.right)
            op_map = {
                ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
                ast.LShift: "<<", ast.RShift: ">>", ast.BitOr: "|",
                ast.BitXor: "^", ast.BitAnd: "&",
            }
            op = op_map.get(type(node.op), "?")
            if left and right:
                return ExpressionIR.binary_op(left, op, right, loc)
            return None

        if isinstance(node, ast.UnaryOp):
            operand = self._parse_expression(node.operand)
            op_map = {ast.UAdd: "+", ast.USub: "-", ast.Not: "not", ast.Invert: "~"}
            op = op_map.get(type(node.op), "?")
            if operand:
                return ExpressionIR.unary_op(op, operand, loc)
            return None

        if isinstance(node, ast.BoolOp):
            # Chain a == b and c == d
            left = self._parse_expression(node.values[0])
            for val in node.values[1:]:
                right = self._parse_expression(val)
                op = "and" if isinstance(node.op, ast.And) else "or"
                if left and right:
                    left = ExpressionIR.binary_op(left, op, right, loc)
            return left

        if isinstance(node, ast.Compare):
            left = self._parse_expression(node.left)
            result = left
            for op, comparator in zip(node.ops, node.comparators):
                right = self._parse_expression(comparator)
                op_map = {
                    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
                    ast.Gt: ">", ast.GtE: ">=", ast.Is: "is", ast.IsNot: "is not",
                    ast.In: "in", ast.NotIn: "not in",
                }
                op_str = op_map.get(type(op), "?")
                if result and right:
                    result = ExpressionIR.binary_op(result, op_str, right, loc)
            return result

        if isinstance(node, ast.Call):
            func = self._parse_expression(node.func)
            args = [self._parse_expression(a) for a in node.args]
            kwargs = {}
            for kw in node.keywords:
                if kw.arg:
                    arg_expr = self._parse_expression(kw.value)
                    if arg_expr:
                        kwargs[kw.arg] = arg_expr

            args = [a for a in args if a is not None]

            if func is None:
                return None

            # Check if this is a tensor operation
            tensor_kind = self._check_tensor_op(func, args)
            if tensor_kind is not None:
                return ExpressionIR.tensor_op(tensor_kind, args, kwargs, loc)

            # Check if this is a mutating method call
            if isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATING_METHODS:
                self._warn_unsupported(
                    node,
                    f"mutating method '{node.func.attr}'",
                    f"Method '{node.func.attr}()' mutates the object in-place, which is not supported. Use a pure functional pattern instead.",
                )

            return ExpressionIR.call(func, args, kwargs, loc)

        if isinstance(node, ast.Attribute):
            target = self._parse_expression(node.value)
            if target:
                return ExpressionIR.attribute(target, node.attr, loc)
            return None

        if isinstance(node, ast.Subscript):
            target = self._parse_expression(node.value)
            index = self._parse_expression(node.slice)
            if target and index:
                return ExpressionIR.subscript(target, index, loc)
            return None

        if isinstance(node, ast.List):
            elements = [self._parse_expression(e) for e in node.elts]
            return ExpressionIR(expr_type="list", elements=[e for e in elements if e is not None], source_loc=loc)

        if isinstance(node, ast.Dict):
            keys = [self._parse_expression(k) for k in node.keys]
            values = [self._parse_expression(v) for v in node.values]
            return ExpressionIR(
                expr_type="dict",
                keys=[k for k in keys if k is not None],
                elements=[v for v in values if v is not None],
                source_loc=loc,
            )

        if isinstance(node, ast.Tuple):
            elements = [self._parse_expression(e) for e in node.elts]
            return ExpressionIR(expr_type="list", elements=[e for e in elements if e is not None], source_loc=loc)

        if isinstance(node, ast.Lambda):
            body = self._parse_expression(node.body)
            return ExpressionIR(expr_type="lambda", value=body, source_loc=loc)

        if isinstance(node, ast.IfExp):
            test = self._parse_expression(node.test)
            body = self._parse_expression(node.body)
            orelse = self._parse_expression(node.orelse)
            return ExpressionIR(expr_type="if_exp", left=test, right=body, operand=orelse, source_loc=loc)

        if isinstance(node, ast.Slice):
            lower = self._parse_expression(node.lower) if node.lower else None
            upper = self._parse_expression(node.upper) if node.upper else None
            step = self._parse_expression(node.step) if node.step else None
            return ExpressionIR(expr_type="slice", left=lower, right=upper, operand=step, source_loc=loc)

        if isinstance(node, ast.ListComp):
            elt = self._parse_expression(node.elt)
            return ExpressionIR(expr_type="list_comp", value=elt, source_loc=loc)

        if isinstance(node, ast.Set):
            elements = [self._parse_expression(e) for e in node.elts]
            return ExpressionIR(expr_type="set", elements=[e for e in elements if e is not None], source_loc=loc)

        # ── Detect known unsupported expressions ──────────────────────────
        unsupported = _UNSUPPORTED_EXPRESSIONS.get(type(node))
        if unsupported:
            construct, message = unsupported
            self._warn_unsupported(node, construct, message)
            return None

        return None

    # ─── Tensor Operation Detection ───────────────────────────────────────

    def _check_tensor_op(self, func: ExpressionIR, args: List[ExpressionIR]) -> Optional[TensorOpKind]:
        """Check if a function call is a known tensor operation."""
        if func.expr_type != "attribute":
            return None

        # Build the full qualified name
        parts = []
        current = func
        while current.expr_type == "attribute":
            parts.append(current.attr)
            current = current.target if current.target else ExpressionIR.variable("")
        if current.expr_type == "variable":
            parts.append(current.name)
        parts.reverse()
        qualified = ".".join(parts)

        # Check tensor functions
        if qualified in TENSOR_FUNCTIONS:
            return TENSOR_FUNCTIONS[qualified]

        # Check method calls on tensors (e.g., x.relu(), x.reshape())
        if len(parts) >= 2 and parts[-1] in ("relu", "sigmoid", "tanh", "reshape", "view", "transpose", "permute", "sum", "mean", "max", "min", "flatten"):
            method_to_op = {
                "relu": TensorOpKind.RELU,
                "sigmoid": TensorOpKind.SIGMOID,
                "tanh": TensorOpKind.TANH,
                "reshape": TensorOpKind.RESHAPE,
                "view": TensorOpKind.VIEW,
                "transpose": TensorOpKind.TRANSPOSE,
                "permute": TensorOpKind.PERMUTE,
                "sum": TensorOpKind.SUM,
                "mean": TensorOpKind.MEAN,
                "max": TensorOpKind.MAX,
                "min": TensorOpKind.MIN,
                "flatten": TensorOpKind.FLATTEN,
            }
            return method_to_op.get(parts[-1])

        # Check nn.Module instantiations
        if len(parts) >= 2 and parts[-1] in NN_MODULE_OPS:
            return NN_MODULE_OPS[parts[-1]]

        return None

    # ─── Helpers for termination / mutation detection ────────────────────

    def _ast_has_break(self, body: List[ast.stmt]) -> bool:
        """Check if a list of AST statements contains a break statement."""
        for stmt in body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Break):
                    return True
        return False

    def _ast_has_recursive_call(self, body: List[ast.stmt], func_name: str) -> bool:
        """Check if a function body contains a direct recursive call to itself."""
        for stmt in body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    if child.func.id == func_name:
                        return True
        return False

    # ─── Expression to Name ───────────────────────────────────────────────

    def _expr_to_name(self, node: ast.expr) -> str:
        """Convert an expression node to a string name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._expr_to_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Num):
            return str(node.n)
        elif isinstance(node, ast.Str):
            return node.s
        return "?"

    # ─── Type Annotations ────────────────────────────────────────────────

    def _parse_type_annotation(self, node: ast.expr) -> Optional[TypeAnnotation]:
        """Parse a type annotation into a TypeAnnotation."""
        if node is None:
            return None

        if isinstance(node, ast.Name):
            return TypeAnnotation(node.id)

        if isinstance(node, ast.Attribute):
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            parts.reverse()
            return TypeAnnotation(".".join(parts))

        if isinstance(node, ast.Subscript):
            if isinstance(node.value, (ast.Name, ast.Attribute)):
                base = self._parse_type_annotation(node.value)
                if isinstance(node.slice, ast.Tuple):
                    params = [self._parse_type_annotation(e) for e in node.slice.elts]
                else:
                    params = [self._parse_type_annotation(node.slice)]
                if base:
                    return TypeAnnotation(base.name, [p for p in params if p is not None])
            return None

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Forward reference (string annotation)
            return TypeAnnotation.from_string(node.value)

        if isinstance(node, ast.Str):
            return TypeAnnotation.from_string(node.s)

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Optional[X] = X | None
            left = self._parse_type_annotation(node.left)
            right = self._parse_type_annotation(node.right)
            if left and right and right.name == "None":
                return TypeAnnotation("Optional", [left])
            return None

        return None

    # ─── Tensor Op Collection ────────────────────────────────────────────

    def _collect_tensor_ops(self, stmt: StatementIR, tensor_ops: List[ExpressionIR]):
        """Recursively collect tensor operations from a statement."""
        if stmt.expression and stmt.expression.expr_type == "tensor_op":
            tensor_ops.append(stmt.expression)
        if stmt.condition and stmt.condition.expr_type == "tensor_op":
            tensor_ops.append(stmt.condition)
        for s in stmt.body:
            self._collect_tensor_ops(s, tensor_ops)
        for s in stmt.orelse:
            self._collect_tensor_ops(s, tensor_ops)
