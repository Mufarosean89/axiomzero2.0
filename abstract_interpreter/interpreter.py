"""
Axiom Zero - Abstract Interpreter

Main orchestrator that runs type inference and shape analysis over the normalized IR.
Coordinates the TypeInferenceEngine and TensorShapeAnalyzer to produce a complete
AbstractState with inferred types, shapes, symbolic facts, and data flow information.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from ast_extractor.ir import (
    NormalizedIR,
    FunctionIR,
    ClassIR,
    StatementIR,
    ExpressionIR,
    TensorOpKind,
)
from .abstract_domain import AbstractState, AbstractValue, TypeDomain, TensorShape
from .type_inference import TypeInferenceEngine
from .shape_analysis import TensorShapeAnalyzer


class AbstractInterpreter:
    """
    Main abstract interpreter for Axiom Zero.
    
    Orchestrates:
    1. Type inference (what types do variables have?)
    2. Shape analysis (what shapes do tensors have?)
    3. Fact extraction (what constraints can we deduce?)
    
    Usage:
        interpreter = AbstractInterpreter()
        state = interpreter.analyze(normalized_ir)
    """

    def __init__(self):
        self.type_engine = TypeInferenceEngine()
        self.shape_analyzer = TensorShapeAnalyzer()
        self._warnings: List[str] = []

    def analyze(self, ir: NormalizedIR) -> AbstractState:
        """
        Run complete abstract interpretation over the normalized IR.
        
        Args:
            ir: Normalized IR from AST extraction + normalization
            
        Returns:
            AbstractState with inferred types, shapes, constraints, and facts
        """
        state = AbstractState()
        self._warnings = []

        # Phase 1: Global analysis (imports, global variables)
        self._analyze_globals(ir, state)

        # Phase 2: Function-level analysis (type inference)
        self._analyze_functions(ir, state)

        # Phase 3: Shape analysis (tensor shapes through operations)
        self._analyze_shapes(ir, state)

        # Phase 4: Extract constraints and facts
        self._extract_facts(ir, state)

        state.analysis_complete = True
        state.warnings = self._warnings

        return state

    def _analyze_globals(self, ir: NormalizedIR, state: AbstractState):
        """Analyze global imports and top-level statements."""
        # Process imports for type information
        for imp in ir.imports:
            if "torch" in imp:
                state.add_type_constraint("torch.tensors_available == True")
                # Seed torch type info
                state.global_env["torch"] = AbstractValue(type_domain=TypeDomain.TOP)
            if "typing" in imp:
                state.add_type_constraint("typing.annotations_enabled == True")
            if "math" in imp:
                state.add_type_constraint("math_functions_available == True")

        # Analyze global statements (assignments, etc.)
        for stmt in ir.global_statements:
            if stmt.stmt_type == "assignment" and stmt.target and stmt.expression:
                if stmt.target.expr_type == "variable":
                    # Infer type of global
                    inferred = self.type_engine._infer_expression(
                        stmt.expression, state.global_env
                    )
                    state.global_env[stmt.target.name] = inferred

    def _analyze_functions(self, ir: NormalizedIR, state: AbstractState):
        """Analyze all functions for type information."""
        for func in ir.functions:
            self._analyze_single_function(func, state)

        for cls in ir.classes:
            for method in cls.methods:
                self._analyze_single_function(method, state, class_name=cls.name)

    def _analyze_single_function(
        self,
        func: FunctionIR,
        state: AbstractState,
        class_name: Optional[str] = None,
    ):
        """
        Analyze a single function for types and shapes.
        
        Args:
            func: FunctionIR to analyze
            state: Abstract state to update
            class_name: Name of the containing class (if a method)
        """
        func_name = func.signature.name
        full_name = f"{class_name}.{func_name}" if class_name else func_name

        try:
            # Step 1: Run type inference
            local_env = self.type_engine.infer_function_types(func, state.global_env)
            state.function_envs[full_name] = local_env

            # Step 2: Build function signature info
            sig_info: Dict[str, Any] = {
                "name": full_name,
                "parameters": [],
                "return_type": None,
                "has_tensor_ops": len(func.tensor_operations) > 0,
                "is_method": func.is_method,
            }

            for param in func.signature.parameters:
                param_info: Dict[str, Any] = {
                    "name": param.name,
                }
                if param.type:
                    param_info["type"] = param.type.to_string()
                    param_info["abstract_type"] = repr(
                        self.type_engine._type_annotation_to_abstract(param.type)
                    )
                elif param.name in local_env:
                    param_info["inferred_type"] = repr(local_env[param.name])
                sig_info["parameters"].append(param_info)

            # Infer return type from return statements
            return_type = self._infer_return_type(func, local_env)
            sig_info["return_type"] = repr(return_type) if return_type else None

            state.function_signatures[full_name] = sig_info

        except Exception as e:
            self._warnings.append(f"Error analyzing function {full_name}: {e}")

    def _infer_return_type(
        self,
        func: FunctionIR,
        env: Dict[str, AbstractValue],
    ) -> Optional[AbstractValue]:
        """Infer the return type by analyzing return statements."""
        return_types = []

        for stmt in func.body:
            types = self._collect_return_types(stmt, env)
            return_types.extend(types)

        if not return_types:
            # Check for declared return type
            if func.signature.return_type:
                return self.type_engine._type_annotation_to_abstract(
                    func.signature.return_type
                )
            return AbstractValue.from_type(TypeDomain.BOTTOM)

        # Join all return types
        result = return_types[0]
        for t in return_types[1:]:
            result = result.join(t)
        return result

    def _collect_return_types(
        self,
        stmt: StatementIR,
        env: Dict[str, AbstractValue],
    ) -> List[AbstractValue]:
        """Collect return types from a statement and its children."""
        types = []
        if stmt.stmt_type == "return" and stmt.value and stmt.value.type:
            types.append(stmt.value.type)
        for s in stmt.body:
            types.extend(self._collect_return_types(s, env))
        for s in stmt.orelse:
            types.extend(self._collect_return_types(s, env))
        return types

    def _analyze_shapes(self, ir: NormalizedIR, state: AbstractState):
        """Run tensor shape analysis."""
        self.shape_analyzer.analyze(ir, state)

    def _extract_facts(self, ir: NormalizedIR, state: AbstractState):
        """
        Extract additional facts and constraints from the analysis.
        
        Generates:
        - Type constraints (variable types)
        - Shape constraints (tensor dimensions)
        - Data flow edges (variable dependencies)
        """
        # Extract type constraints from all function environments
        for func_name, env in state.function_envs.items():
            for var_name, value in env.items():
                if value.type_domain != TypeDomain.TOP:
                    state.add_type_constraint(
                        f"type({var_name}) == {value.type_domain.value}"
                    )

        # Extract data flow graph
        for func in ir.all_functions:
            self._extract_data_flow(func, state)

    def _extract_data_flow(self, func: FunctionIR, state: AbstractState):
        """Extract data flow dependencies from a function."""
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

                    deps = []
                    if stmt.expression:
                        deps = self._collect_var_deps(stmt.expression)

                    for target in targets:
                        key = f"{func_name}.{target}"
                        if key not in state.data_flow_graph:
                            state.data_flow_graph[key] = []
                        state.data_flow_graph[key].extend(deps)

                walk(stmt.body)
                walk(stmt.orelse)

        walk(func.body)

    def _collect_var_deps(self, expr: ExpressionIR) -> List[str]:
        """Collect all variable dependencies from an expression."""
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
        """Return a human-readable summary of the abstract state."""
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
