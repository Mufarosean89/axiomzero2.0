"""Axiom Zero - Abstract Interpreter

Orchestrates type inference and shape analysis over normalized IR.
Coordinates TypeInferenceEngine and TensorShapeAnalyzer to produce a complete AbstractState.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from ast_extractor.ir import (
    NormalizedIR,
    FunctionIR,
    StatementIR,
    ExpressionIR,
)
from .abstract_domain import AbstractState, AbstractValue, TypeDomain
from .type_inference import TypeInferenceEngine
from .shape_analysis import TensorShapeAnalyzer


class AbstractInterpreter:
    """Main abstract interpreter orchestrating type inference, shape analysis, and fact extraction."""

    def __init__(self):
        self.type_engine = TypeInferenceEngine()
        self.shape_analyzer = TensorShapeAnalyzer()
        self._warnings: List[str] = []

    def analyze(self, ir: NormalizedIR) -> AbstractState:
        state = AbstractState()
        self._warnings = []
        self._analyze_globals(ir, state)
        self._analyze_functions(ir, state)
        self._analyze_shapes(ir, state)
        self._extract_facts(ir, state)
        state.analysis_complete = True
        state.warnings = self._warnings
        return state

    def _analyze_globals(self, ir: NormalizedIR, state: AbstractState):
        for imp in ir.imports:
            if "torch" in imp:
                state.add_type_constraint("torch.tensors_available == True")
                state.global_env["torch"] = AbstractValue(type_domain=TypeDomain.TOP)
            if "typing" in imp:
                state.add_type_constraint("typing.annotations_enabled == True")
            if "math" in imp:
                state.add_type_constraint("math_functions_available == True")

        for stmt in ir.global_statements:
            if stmt.stmt_type == "assignment" and stmt.target and stmt.expression:
                if stmt.target.expr_type == "variable":
                    inferred = self.type_engine._infer_expression(stmt.expression, state.global_env)
                    state.global_env[stmt.target.name] = inferred

    def _analyze_functions(self, ir: NormalizedIR, state: AbstractState):
        for func in ir.functions:
            self._analyze_single_function(func, state)
        for cls in ir.classes:
            for method in cls.methods:
                self._analyze_single_function(method, state, class_name=cls.name)

    def _analyze_single_function(self, func: FunctionIR, state: AbstractState, class_name: Optional[str] = None):
        func_name = func.signature.name
        full_name = f"{class_name}.{func_name}" if class_name else func_name

        try:
            local_env = self.type_engine.infer_function_types(func, state.global_env)
            state.function_envs[full_name] = local_env

            sig_info: Dict[str, Any] = {
                "name": full_name,
                "parameters": [],
                "return_type": None,
                "has_tensor_ops": len(func.tensor_operations) > 0,
                "is_method": func.is_method,
            }

            for param in func.signature.parameters:
                param_info: Dict[str, Any] = {"name": param.name}
                if param.type:
                    param_info["type"] = param.type.to_string()
                    param_info["abstract_type"] = repr(self.type_engine._type_annotation_to_abstract(param.type))
                elif param.name in local_env:
                    param_info["inferred_type"] = repr(local_env[param.name])
                sig_info["parameters"].append(param_info)

            return_type = self._infer_return_type(func, local_env)
            sig_info["return_type"] = repr(return_type) if return_type else None
            state.function_signatures[full_name] = sig_info

        except Exception as e:
            self._warnings.append(f"Error analyzing function {full_name}: {e}")

    def _infer_return_type(self, func: FunctionIR, env: Dict[str, AbstractValue]) -> Optional[AbstractValue]:
        return_types = []
        for stmt in func.body:
            return_types.extend(self._collect_return_types(stmt, env))
        if not return_types:
            if func.signature.return_type:
                return self.type_engine._type_annotation_to_abstract(func.signature.return_type)
            return AbstractValue.from_type(TypeDomain.BOTTOM)
        result = return_types[0]
        for t in return_types[1:]:
            result = result.join(t)
        return result

    def _collect_return_types(self, stmt: StatementIR, env: Dict[str, AbstractValue]) -> List[AbstractValue]:
        types = []
        if stmt.stmt_type == "return" and stmt.value and stmt.value.type:
            types.append(stmt.value.type)
        for s in stmt.body:
            types.extend(self._collect_return_types(s, env))
        for s in stmt.orelse:
            types.extend(self._collect_return_types(s, env))
        return types

    def _analyze_shapes(self, ir: NormalizedIR, state: AbstractState):
        self.shape_analyzer.analyze(ir, state)

    def _extract_facts(self, ir: NormalizedIR, state: AbstractState):
        for func_name, env in state.function_envs.items():
            for var_name, value in env.items():
                if value.type_domain != TypeDomain.TOP:
                    state.add_type_constraint(f"type({var_name}) == {value.type_domain.value}")

        for func in ir.all_functions:
            self._extract_data_flow(func, state)

    def _extract_data_flow(self, func: FunctionIR, state: AbstractState):
        func_name = func.signature.name

        def walk(stmts: List[StatementIR]):
            for stmt in stmts:
                if stmt.stmt_type == "assignment":
                    targets = []
                    if stmt.target and stmt.target.expr_type == "variable":
                        targets.append(stmt.target.name)
                    elif stmt.target and stmt.target.expr_type == "list":
                        for elem in stmt.target.elements:
                            if elem and elem.expr_type == "variable":
                                targets.append(elem.name)

                    deps = self._collect_var_deps(stmt.expression) if stmt.expression else []
                    for target in targets:
                        key = f"{func_name}.{target}"
                        state.data_flow_graph.setdefault(key, []).extend(deps)

                walk(stmt.body)
                walk(stmt.orelse)

        walk(func.body)

    def _collect_var_deps(self, expr: ExpressionIR) -> List[str]:
        deps = []
        if expr is None:
            return deps
        if expr.expr_type == "variable":
            deps.append(expr.name)
        for sub_expr in [expr.left, expr.right, expr.operand, expr.target, expr.func, expr.index]:
            if sub_expr:
                deps.extend(self._collect_var_deps(sub_expr))
        for arg in expr.args:
            if arg:
                deps.extend(self._collect_var_deps(arg))
        return deps

    @staticmethod
    def pretty_print(state: AbstractState) -> str:
        lines = ["=" * 60, "ABSTRACT INTERPRETATION RESULTS", "=" * 60, ""]
        lines.append("--- GLOBAL ENVIRONMENT ---")
        for var, value in state.global_env.items():
            lines.append(f"  {var}: {value}")
        lines.append("")
        lines.append("--- FUNCTION ENVIRONMENTS ---")
        for func_name, env in state.function_envs.items():
            lines.append(f"\n  Function: {func_name}")
            for var, value in env.items():
                lines.append(f"    {var}: {value}")
        lines.append("")
        lines.append("--- SHAPE FACTS ---")
        for fact in state.shape_facts:
            lines.append(f"  {fact}")
        lines.append("")
        lines.append("--- TYPE CONSTRAINTS ---")
        for constr in state.type_constraints:
            lines.append(f"  {constr}")
        lines.append("")
        lines.append("--- FUNCTION SIGNATURES ---")
        for fname, sig in state.function_signatures.items():
            lines.append(f"  {fname}: {sig}")
        lines.append("")
        if state.warnings:
            lines.append("--- WARNINGS ---")
            for w in state.warnings:
                lines.append(f"  WARN: {w}")
            lines.append("")
        lines.append(f"Analysis complete: {state.analysis_complete}")
        return "\n".join(lines)
