"""Axiom Zero - AST Extractor Parser

Parses Python source into Axiom Zero's normalized IR using Python's built-in `ast` module.
Special attention is paid to:
- Tensor operations (torch.*, F.*, nn.*)
- Type annotations
- Decorators (especially @requires, @ensures, @invariant)
- Loop structures with bounds
"""

import ast
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

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

# ─── Operator Maps ────────────────────────────────────────────────────────────

_BINARY_OP_MAP: Dict[type, str] = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
    ast.LShift: "<<", ast.RShift: ">>", ast.BitOr: "|",
    ast.BitXor: "^", ast.BitAnd: "&",
}

_UNARY_OP_MAP: Dict[type, str] = {
    ast.UAdd: "+", ast.USub: "-", ast.Not: "not", ast.Invert: "~",
}

_COMPARE_OP_MAP: Dict[type, str] = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
    ast.Gt: ">", ast.GtE: ">=", ast.Is: "is", ast.IsNot: "is not",
    ast.In: "in", ast.NotIn: "not in",
}

# ─── Tensor Operation Maps ────────────────────────────────────────────────────

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

_TENSOR_METHODS: Dict[str, TensorOpKind] = {
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

# ─── Spec Decorators ──────────────────────────────────────────────────────────

SPEC_DECORATORS: Set[str] = {"requires", "ensures", "invariant", "precondition", "postcondition"}

# ─── Mutating Methods (stateful calls the analyzer rejects) ────────────────────

_MUTATING_METHODS: Set[str] = {
    "append", "extend", "insert", "remove", "pop", "sort", "reverse", "clear",
    "update", "setdefault", "popitem",
    "add", "discard", "difference_update", "intersection_update", "symmetric_difference_update",
}

# ─── Unsupported Constructs ──────────────────────────────────────────────────

_UNSUPPORTED_STATEMENTS: Dict[type, Tuple[str, str]] = {
    ast.Try: ("try/except/finally", "Exception handling is not supported. Use @requires/@ensures instead."),
    ast.Delete: ("del", "Delete statements are not supported."),
    ast.Global: ("global", "Global declarations are not supported."),
    ast.Nonlocal: ("nonlocal", "Nonlocal declarations are not supported."),
    ast.AsyncFor: ("async for", "Async iteration is not supported."),
    ast.AsyncWith: ("async with", "Async context managers are not supported."),
}

_UNSUPPORTED_EXPRESSIONS: Dict[type, Tuple[str, str]] = {
    ast.GeneratorExp: ("generator expression", "Use list comprehension instead."),
    ast.SetComp: ("set comprehension", "Set comprehensions are not supported."),
    ast.DictComp: ("dict comprehension", "Dict comprehensions are not supported."),
    ast.Yield: ("yield", "Generators/yield are not supported."),
    ast.YieldFrom: ("yield from", "Generators/yield from are not supported."),
    ast.Await: ("await", "Async/await expressions are not supported."),
    ast.Starred: ("starred expression *args", "Starred unpacking expressions are not supported."),
    ast.JoinedStr: ("f-string", "f-strings with interpolated expressions are not supported; use string concatenation instead."),
    ast.FormattedValue: ("f-string expression", "f-string expressions are not supported; use string concatenation instead."),
}

# Register version-dependent unsupported constructs
if hasattr(ast, "TryStar"):
    _UNSUPPORTED_STATEMENTS[ast.TryStar] = ("try*", "Star import exception handling is not supported.")
if hasattr(ast, "Match"):
    _UNSUPPORTED_STATEMENTS[ast.Match] = ("match/case", "Pattern matching is not supported.")
if hasattr(ast, "NamedExpr"):
    _UNSUPPORTED_EXPRESSIONS[ast.NamedExpr] = ("walrus operator :=", "Assignment expressions (:=) are not supported.")


class ParserError(Exception):
    """Error raised during AST parsing."""
    pass


class Parser:
    """Parses Python source code into Axiom Zero's NormalizedIR.

    Usage:
        parser = Parser()
        ir = parser.parse("path/to/file.py")
        # or
        ir = parser.parse_source("def foo(x): return x + 1", module_name="example")
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        self._config = config or ParserConfig()
        self._current_source_path: Optional[str] = None
        self._current_ir: Optional[NormalizedIR] = None

    def parse(self, source_path: str) -> NormalizedIR:
        """Parse a Python source file into NormalizedIR."""
        self._current_source_path = source_path
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                source = f.read()
        except FileNotFoundError:
            raise ParserError(f"File not found: {source_path}")
        return self.parse_source(source, module_name=source_path)

    def parse_source(self, source: str, module_name: str = "<string>") -> NormalizedIR:
        """Parse Python source string into NormalizedIR."""
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

    def _warn(self, node: ast.AST, construct: str, message: str):
        """Record a warning (or raise in strict mode) for an unsupported construct."""
        location = self._loc(node)
        warning = ParserWarning(construct=construct, location=location, message=message)

        if self._config.strict_mode:
            raise ParserError(str(warning))

        if self._config.collect_warnings:
            self._current_ir.warnings.append(warning)

    def _loc(self, node: ast.AST) -> str:
        """Create a source location string from an AST node."""
        if hasattr(node, "lineno"):
            path = self._current_source_path or "<unknown>"
            return f"{path}:{node.lineno}"
        return ""

    def _expr_to_str(self, node: ast.expr) -> str:
        """Convert an expression node to a dotted string name."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._expr_to_str(node.value)}.{node.attr}"
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.Num):
            return str(node.n)
        if isinstance(node, ast.Str):
            return node.s
        return "?"

    # ─── Module Level ──────────────────────────────────────────────────────

    def _walk_module(self, node: ast.Module, ir: NormalizedIR):
        """Walk the top-level module body."""
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_ir = self._parse_function(stmt)
                if func_ir:
                    ir.functions.append(func_ir)
            elif isinstance(stmt, ast.ClassDef):
                class_ir = self._parse_class(stmt)
                if class_ir:
                    ir.classes.append(class_ir)
            elif isinstance(stmt, ast.Import):
                ir.imports.extend(alias.name for alias in stmt.names)
            elif isinstance(stmt, ast.ImportFrom):
                module = stmt.module or ""
                for alias in stmt.names:
                    full = f"{module}.{alias.name}" if module else alias.name
                    ir.imports.append(full)
            else:
                stmt_ir = self._parse_statement(stmt)
                if stmt_ir:
                    ir.global_statements.append(stmt_ir)

    # ─── Functions ─────────────────────────────────────────────────────────

    def _parse_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[FunctionIR]:
        """Parse a function definition into FunctionIR."""
        loc = self._loc(node)
        decorators = [d for d in (self._parse_decorator(d) for d in node.decorator_list) if d]

        sig = FunctionSignature(
            name=node.name,
            parameters=self._parse_parameters(node),
            return_type=self._parse_type_annotation(node.returns) if node.returns else None,
            decorators=decorators,
        )

        body = []
        tensor_ops = []
        for stmt in node.body:
            stmt_ir = self._parse_statement(stmt)
            if stmt_ir:
                body.append(stmt_ir)
                self._collect_tensor_ops(stmt_ir, tensor_ops)

        nested = [
            func_ir for s in node.body
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (func_ir := self._parse_function(s)) is not None
        ]

        if self._has_recursive_call(node.body, node.name):
            self._warn(node, "recursion",
                       f"Recursive call to '{node.name}' — termination cannot be verified automatically.")

        return FunctionIR(
            signature=sig, body=body, tensor_operations=tensor_ops,
            nested_functions=nested, source_loc=loc,
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )

    def _parse_parameters(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[ParameterIR]:
        """Extract parameters from a function definition."""
        params = []
        for arg in node.args.args:
            params.append(ParameterIR(
                name=arg.arg,
                type=self._parse_type_annotation(arg.annotation) if arg.annotation else None,
            ))
        if node.args.vararg:
            params.append(ParameterIR(name=f"*{node.args.vararg.arg}"))
        for arg in node.args.kwonlyargs:
            params.append(ParameterIR(
                name=arg.arg,
                type=self._parse_type_annotation(arg.annotation) if arg.annotation else None,
            ))
        if node.args.kwarg:
            params.append(ParameterIR(name=f"**{node.args.kwarg.arg}"))
        return params

    def _parse_decorator(self, node: ast.expr) -> Optional[Dict[str, Any]]:
        """Parse a decorator expression into a structured dict."""
        if isinstance(node, ast.Call):
            func_name = self._expr_to_str(node.func)
            args = [a for a in (self._parse_expression(a) for a in node.args) if a]
            kwargs = {kw.arg: v for kw in node.keywords if kw.arg and (v := self._parse_expression(kw.value))}

            info: Dict[str, Any] = {"name": func_name, "args": args, "kwargs": kwargs}
            # Collect string predicates from decorator arguments
            predicates = []
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    predicates.append(a.value)
                elif isinstance(a, ast.Str):
                    predicates.append(a.s)
            if predicates:
                info["predicates"] = predicates
            return info

        if isinstance(node, (ast.Name, ast.Attribute)):
            return {"name": self._expr_to_str(node), "args": [], "kwargs": {}}

        return None

    # ─── Classes ───────────────────────────────────────────────────────────

    def _parse_class(self, node: ast.ClassDef) -> Optional[ClassIR]:
        """Parse a class definition into ClassIR."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._expr_to_str(base))

        decorators = [d for d in (self._parse_decorator(d) for d in node.decorator_list) if d]

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
            name=node.name, bases=bases, methods=methods,
            class_variables=class_vars, decorators=decorators,
            source_loc=self._loc(node),
        )

    # ─── Statements ────────────────────────────────────────────────────────

    def _parse_statement(self, node: ast.stmt) -> Optional[StatementIR]:
        """Parse a statement into StatementIR."""
        loc = self._loc(node)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None  # Handled at module/class level

        if isinstance(node, ast.Assign):
            return self._parse_assign(node, loc)

        if isinstance(node, ast.AnnAssign):
            target = self._parse_expression(node.target)
            expr = self._parse_expression(node.value) if node.value else None
            stmt = StatementIR(stmt_type="assignment", target=target, expression=expr, source_loc=loc)
            if (ann := self._parse_type_annotation(node.annotation)) is not None:
                stmt.annotations["type"] = ann
            return stmt

        if isinstance(node, ast.AugAssign):
            return self._parse_aug_assign(node, loc)

        if isinstance(node, ast.Return):
            return StatementIR(
                stmt_type="return",
                value=self._parse_expression(node.value) if node.value else None,
                source_loc=loc,
            )

        if isinstance(node, ast.Expr):
            return StatementIR(
                stmt_type="expression",
                expression=self._parse_expression(node.value),
                source_loc=loc,
            )

        if isinstance(node, ast.If):
            condition = self._parse_expression(node.test)
            body = self._parse_stmts(node.body)
            orelse = self._parse_stmts(node.orelse)
            return StatementIR(stmt_type="if", condition=condition, body=body, orelse=orelse, source_loc=loc)

        if isinstance(node, ast.For):
            var = ExpressionIR.variable(node.target.id, loc) if isinstance(node.target, ast.Name) else self._parse_expression(node.target)
            return StatementIR(
                stmt_type="for", iter_var=var,
                iterable=self._parse_expression(node.iter),
                body=self._parse_stmts(node.body),
                orelse=self._parse_stmts(node.orelse),
                source_loc=loc,
            )

        if isinstance(node, ast.While):
            if isinstance(node.test, ast.Constant) and node.test.value is True and not self._has_break(node.body):
                self._warn(node, "infinite loop", "while True with no break — loop termination cannot be verified.")
            return StatementIR(
                stmt_type="while",
                condition=self._parse_expression(node.test),
                body=self._parse_stmts(node.body),
                orelse=self._parse_stmts(node.orelse),
                source_loc=loc,
            )

        if isinstance(node, ast.Assert):
            return StatementIR(
                stmt_type="assert",
                condition=self._parse_expression(node.test),
                expression=self._parse_expression(node.msg) if node.msg else None,
                source_loc=loc,
            )

        if isinstance(node, (ast.Pass, ast.Break, ast.Continue)):
            return StatementIR(stmt_type=type(node).__name__.lower(), source_loc=loc)

        if isinstance(node, ast.Raise):
            return StatementIR(
                stmt_type="raise",
                expression=self._parse_expression(node.exc) if node.exc else None,
                source_loc=loc,
            )

        if isinstance(node, ast.With):
            items = [self._parse_expression(item.context_expr) for item in node.items]
            return StatementIR(
                stmt_type="with",
                expression=items[0] if items else None,
                body=self._parse_stmts(node.body),
                source_loc=loc,
            )

        # Reject imports and class definitions inside functions
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._warn(node, "import", "Import statements inside functions are not supported. Move them to module level.")
            return None
        if isinstance(node, ast.ClassDef):
            self._warn(node, "class", "Nested class definitions inside functions are not supported.")
            return None

        unsupported = _UNSUPPORTED_STATEMENTS.get(type(node))
        if unsupported:
            self._warn(node, *unsupported)
        return None

    def _parse_assign(self, node: ast.Assign, loc: str) -> StatementIR:
        """Parse an assignment statement."""
        # Warn about mutable state assignments
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                self._warn(target, "attribute assignment",
                           "Assigning to an object attribute mutates state and is not supported. Use a pure functional pattern.")
            if isinstance(target, ast.Subscript):
                self._warn(target, "subscript assignment",
                           "Assigning to a subscript mutates state and is not supported. Use a pure functional pattern.")

        # Multi-target: a = b = expr — keep only the last assignment
        target = self._parse_expression(node.targets[-1])
        expr = self._parse_expression(node.value)
        return StatementIR(stmt_type="assignment", target=target, expression=expr, source_loc=loc)

    def _parse_aug_assign(self, node: ast.AugAssign, loc: str) -> StatementIR:
        """Parse an augmented assignment like a += 1 → a = a + 1."""
        op = _BINARY_OP_MAP.get(type(node.op), "?")
        target = self._parse_expression(node.target)
        expr = self._parse_expression(node.value)
        full = ExpressionIR.binary_op(target, op, expr, loc)
        return StatementIR(stmt_type="assignment", target=target, expression=full, source_loc=loc)

    def _parse_stmts(self, stmts: List[ast.stmt]) -> List[StatementIR]:
        """Parse a list of AST statements, filtering out None results."""
        return [s for s in (self._parse_statement(s) for s in stmts) if s is not None]

    # ─── Expressions ──────────────────────────────────────────────────────

    def _parse_expression(self, node: ast.expr) -> Optional[ExpressionIR]:
        """Parse an expression into ExpressionIR."""
        if node is None:
            return None

        loc = self._loc(node)

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
            op = _BINARY_OP_MAP.get(type(node.op), "?")
            return ExpressionIR.binary_op(left, op, right, loc) if left and right else None

        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OP_MAP.get(type(node.op), "?")
            operand = self._parse_expression(node.operand)
            return ExpressionIR.unary_op(op, operand, loc) if operand else None

        if isinstance(node, ast.BoolOp):
            return self._parse_bool_op(node, loc)

        if isinstance(node, ast.Compare):
            return self._parse_compare(node, loc)

        if isinstance(node, ast.Call):
            return self._parse_call(node, loc)

        if isinstance(node, ast.Attribute):
            target = self._parse_expression(node.value)
            return ExpressionIR.attribute(target, node.attr, loc) if target else None

        if isinstance(node, ast.Subscript):
            target = self._parse_expression(node.value)
            index = self._parse_expression(node.slice)
            return ExpressionIR.subscript(target, index, loc) if target and index else None

        if isinstance(node, ast.List):
            elements = [e for e in (self._parse_expression(e) for e in node.elts) if e]
            return ExpressionIR(expr_type="list", elements=elements, source_loc=loc)

        if isinstance(node, ast.Tuple):
            elements = [e for e in (self._parse_expression(e) for e in node.elts) if e]
            return ExpressionIR(expr_type="list", elements=elements, source_loc=loc)

        if isinstance(node, ast.Dict):
            keys = [k for k in (self._parse_expression(k) for k in node.keys) if k]
            values = [v for v in (self._parse_expression(v) for v in node.values) if v]
            return ExpressionIR(expr_type="dict", keys=keys, elements=values, source_loc=loc)

        if isinstance(node, ast.Lambda):
            return ExpressionIR(expr_type="lambda", value=self._parse_expression(node.body), source_loc=loc)

        if isinstance(node, ast.IfExp):
            return ExpressionIR(
                expr_type="if_exp",
                left=self._parse_expression(node.test),
                right=self._parse_expression(node.body),
                operand=self._parse_expression(node.orelse),
                source_loc=loc,
            )

        if isinstance(node, ast.Slice):
            return ExpressionIR(
                expr_type="slice",
                left=self._parse_expression(node.lower) if node.lower else None,
                right=self._parse_expression(node.upper) if node.upper else None,
                operand=self._parse_expression(node.step) if node.step else None,
                source_loc=loc,
            )

        if isinstance(node, ast.ListComp):
            return ExpressionIR(expr_type="list_comp", value=self._parse_expression(node.elt), source_loc=loc)

        if isinstance(node, ast.Set):
            elements = [e for e in (self._parse_expression(e) for e in node.elts) if e]
            return ExpressionIR(expr_type="set", elements=elements, source_loc=loc)

        unsupported = _UNSUPPORTED_EXPRESSIONS.get(type(node))
        if unsupported:
            self._warn(node, *unsupported)
        return None

    def _parse_bool_op(self, node: ast.BoolOp, loc: str) -> Optional[ExpressionIR]:
        """Parse a boolean operator expression (and/or) into chained binary ops."""
        op_str = "and" if isinstance(node.op, ast.And) else "or"
        left = self._parse_expression(node.values[0])
        for val in node.values[1:]:
            right = self._parse_expression(val)
            if left and right:
                left = ExpressionIR.binary_op(left, op_str, right, loc)
        return left

    def _parse_compare(self, node: ast.Compare, loc: str) -> Optional[ExpressionIR]:
        """Parse a comparison chain (a < b < c) into chained binary ops."""
        result = self._parse_expression(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self._parse_expression(comparator)
            op_str = _COMPARE_OP_MAP.get(type(op), "?")
            if result and right:
                result = ExpressionIR.binary_op(result, op_str, right, loc)
        return result

    def _parse_call(self, node: ast.Call, loc: str) -> Optional[ExpressionIR]:
        """Parse a function/method call expression."""
        func = self._parse_expression(node.func)
        if func is None:
            return None

        args = [a for a in (self._parse_expression(a) for a in node.args) if a]

        kwargs = {}
        for kw in node.keywords:
            if kw.arg and (v := self._parse_expression(kw.value)) is not None:
                kwargs[kw.arg] = v

        # Detect tensor operations
        tensor_kind = self._detect_tensor_op(func, args)
        if tensor_kind is not None:
            return ExpressionIR.tensor_op(tensor_kind, args, kwargs, loc)

        # Warn about mutating method calls
        if isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATING_METHODS:
            self._warn(node, f"mutating method '{node.func.attr}'",
                       f"Method '{node.func.attr}()' mutates the object in-place, which is not supported. Use a pure functional pattern.")

        return ExpressionIR.call(func, args, kwargs, loc)

    # ─── Tensor Operation Detection ───────────────────────────────────────

    def _detect_tensor_op(self, func: ExpressionIR, args: List[ExpressionIR]) -> Optional[TensorOpKind]:
        """Check if a function call is a known tensor operation."""
        if func.expr_type != "attribute":
            return None

        qualified = self._build_qualified_name(func)

        if qualified in TENSOR_FUNCTIONS:
            return TENSOR_FUNCTIONS[qualified]

        parts = qualified.split(".")
        if len(parts) >= 2 and parts[-1] in _TENSOR_METHODS:
            return _TENSOR_METHODS[parts[-1]]

        if len(parts) >= 2 and parts[-1] in NN_MODULE_OPS:
            return NN_MODULE_OPS[parts[-1]]

        return None

    def _build_qualified_name(self, expr: ExpressionIR) -> str:
        """Build a dotted qualified name from an attribute chain."""
        parts = []
        current = expr
        while current.expr_type == "attribute":
            parts.append(current.attr)
            current = current.target if current.target else ExpressionIR.variable("")
        if current.expr_type == "variable":
            parts.append(current.name)
        return ".".join(reversed(parts))

    # ─── Helpers: termination / mutation detection ─────────────────────────

    def _has_break(self, body: List[ast.stmt]) -> bool:
        """Check if AST statements contain a break statement."""
        for stmt in body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Break):
                    return True
        return False

    def _has_recursive_call(self, body: List[ast.stmt], func_name: str) -> bool:
        """Check if a function body contains a direct recursive call to itself."""
        for stmt in body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == func_name:
                    return True
        return False

    # ─── Type Annotations ────────────────────────────────────────────────

    def _parse_type_annotation(self, node: ast.expr) -> Optional[TypeAnnotation]:
        """Parse a type annotation AST node into TypeAnnotation."""
        if node is None:
            return None

        if isinstance(node, ast.Name):
            return TypeAnnotation(node.id)

        if isinstance(node, ast.Attribute):
            return TypeAnnotation(self._expr_to_str(node))

        if isinstance(node, ast.Subscript):
            base = self._parse_type_annotation(node.value)
            if base is None:
                return None
            params_slice = node.slice
            if isinstance(params_slice, ast.Tuple):
                params = [p for p in (self._parse_type_annotation(e) for e in params_slice.elts) if p]
            else:
                params = [p for p in (self._parse_type_annotation(params_slice),) if p]
            return TypeAnnotation(base.name, params)

        if isinstance(node, (ast.Constant, ast.Str)):
            raw = node.value if isinstance(node, ast.Constant) else node.s
            if isinstance(raw, str):
                return TypeAnnotation.from_string(raw)
            return None

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._parse_type_annotation(node.left)
            right = self._parse_type_annotation(node.right)
            if left and right and right.name == "None":
                return TypeAnnotation("Optional", [left])
            return None

        return None

    # ─── Tensor Op Collection ────────────────────────────────────────────

    def _collect_tensor_ops(self, stmt: StatementIR, tensor_ops: List[ExpressionIR]):
        """Recursively collect tensor operations from a statement."""
        for field_name in ("expression", "condition"):
            expr = getattr(stmt, field_name, None)
            if expr and expr.expr_type == "tensor_op":
                tensor_ops.append(expr)
        for s in stmt.body:
            self._collect_tensor_ops(s, tensor_ops)
        for s in stmt.orelse:
            self._collect_tensor_ops(s, tensor_ops)
