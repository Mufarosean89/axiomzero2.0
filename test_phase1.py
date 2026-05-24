"""Phase 1 tests: Parser Pipeline (AST extractor → abstract interpreter → spec ingestion)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_ast_extractor_basic():
    """Parse a simple function and verify IR structure."""
    from ast_extractor import parse_source, normalize

    source = """
def add(x: int, y: int) -> int:
    result = x + y
    return result
"""
    ir = parse_source(source, "test_basic")
    assert ir is not None
    assert len(ir.functions) == 1

    func = ir.functions[0]
    assert func.signature.name == "add"
    assert len(func.signature.parameters) == 2
    assert func.signature.parameters[0].name == "x"
    assert func.signature.parameters[1].name == "y"
    assert func.signature.return_type is not None
    assert func.signature.return_type.name == "int"
    assert len(func.body) == 2
    assert func.body[0].stmt_type == "assignment"
    assert func.body[1].stmt_type == "return"

    normalized = normalize(ir)
    assert normalized is not None

    print("  PASS: test_ast_extractor_basic")


def test_ast_extractor_tensor_ops():
    """Detect tensor operations in torch-using code."""
    from ast_extractor import parse_source

    source = """
import torch
import torch.nn.functional as F

def forward(x):
    z = torch.relu(x)
    y = F.sigmoid(z)
    w = x.view(-1, 768)
    return w
"""
    ir = parse_source(source, "test_tensor")
    assert len(ir.imports) >= 2
    assert "torch" in ir.imports

    func = ir.functions[0]
    assert len(func.tensor_operations) > 0

    op_kinds = [op.tensor_op_kind.name for op in func.tensor_operations]
    print(f"  Tensor ops detected: {op_kinds}")
    print("  PASS: test_ast_extractor_tensor_ops")


def test_ast_extractor_conditionals():
    """Parse if/else conditionals."""
    from ast_extractor import parse_source

    source = """
def max_val(a: int, b: int) -> int:
    if a > b:
        return a
    else:
        return b
"""
    ir = parse_source(source, "test_cond")
    func = ir.functions[0]
    assert func.body[0].stmt_type == "if"
    assert func.body[0].condition is not None
    assert len(func.body[0].body) == 1
    assert len(func.body[0].orelse) == 1
    print("  PASS: test_ast_extractor_conditionals")


def test_ast_extractor_loops():
    """Parse for/while loops."""
    from ast_extractor import parse_source

    source = """
def sum_to_n(n: int) -> int:
    total = 0
    for i in range(n):
        total = total + i
    return total
"""
    ir = parse_source(source, "test_loop")
    func = ir.functions[0]
    loop_stmt = func.body[1]
    assert loop_stmt.stmt_type == "for"
    assert loop_stmt.iter_var is not None
    assert loop_stmt.iter_var.name == "i"
    assert len(loop_stmt.body) == 1
    print("  PASS: test_ast_extractor_loops")


def test_ast_extractor_classes():
    """Parse class definitions."""
    from ast_extractor import parse_source

    source = """
class LinearLayer:
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.randn(in_features, out_features)

    def forward(self, x):
        return x @ self.weight.T
"""
    ir = parse_source(source, "test_class")
    assert len(ir.classes) == 1
    cls = ir.classes[0]
    assert cls.name == "LinearLayer"
    assert len(cls.methods) == 2
    assert cls.methods[0].is_method is True
    print("  PASS: test_ast_extractor_classes")


def test_ast_extractor_type_annotations():
    """Parse complex type annotations like List[int], Optional[float]."""
    from ast_extractor import parse_source

    source = """
from typing import List, Optional

def process(items: List[int]) -> Optional[float]:
    return None
"""
    ir = parse_source(source, "test_types")
    func = ir.functions[0]
    param = func.signature.parameters[0]
    assert param.type is not None
    assert param.type.name == "List"
    assert len(param.type.params) == 1
    assert param.type.params[0].name == "int"
    print("  PASS: test_ast_extractor_type_annotations")


def test_abstract_interpreter():
    """Run abstract interpretation on a simple function."""
    from ast_extractor import parse_source, normalize
    from abstract_interpreter import analyze

    source = """
def add(x: int, y: int) -> int:
    result = x + y
    return result
"""
    ir = parse_source(source, "test_ai")
    ir = normalize(ir)

    state = analyze(ir)
    assert state is not None
    assert state.analysis_complete is True
    assert "add" in state.function_envs
    env = state.function_envs["add"]
    assert "x" in env
    assert "y" in env

    print(f"  Types inferred: x={env.get('x')}, y={env.get('y')}")
    print(f"  Shape facts: {state.shape_facts}")
    print(f"  Type constraints: {state.type_constraints}")
    print("  PASS: test_abstract_interpreter")


def test_abstract_interpreter_tensors():
    """Run abstract interpretation with tensor operations."""
    from ast_extractor import parse_source, normalize
    from abstract_interpreter import analyze

    source = """
import torch

def forward(x: torch.Tensor):
    z = torch.relu(x)
    return z
"""
    ir = parse_source(source, "test_tensor_ai")
    ir = normalize(ir)

    state = analyze(ir)
    has_tensor_ops = any(
        sig.get("has_tensor_ops", False)
        for sig in state.function_signatures.values()
    )
    assert has_tensor_ops

    print(f"  Function signatures: {state.function_signatures}")
    print(f"  Tensor ops metadata: {state.tensor_ops_metadata}")
    print("  PASS: test_abstract_interpreter_tensors")


def test_spec_ingestion():
    """Extract specs from a simple function with docstring."""
    from ast_extractor import parse_source, normalize
    from spec_ingestion import extract_specs

    source = """
def add(x: int, y: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    result = x + y
    return result
"""
    ir = parse_source(source, "test_spec")
    ir = normalize(ir)

    specs = extract_specs(ir)
    assert specs is not None
    print(f"  Total obligations: {specs.total_count}")
    print(specs.summary())
    print("  PASS: test_spec_ingestion")


def test_unsupported_construct_warnings():
    """Verify unsupported Python constructs generate warnings."""
    from ast_extractor import parse_source

    # try/except
    source = """
def unsafe_divide(x: int, y: int) -> int:
    try:
        return x // y
    except:
        return 0
"""
    ir = parse_source(source, "test_try")
    assert len(ir.warnings) >= 1
    assert any("try/except" in w.construct for w in ir.warnings)

    # generator expression
    source = """
def get_squares(n: int) -> list:
    return (x * x for x in range(n))
"""
    ir = parse_source(source, "test_gen")
    assert any("generator" in w.construct for w in ir.warnings)

    # walrus operator
    source = """
def process(x: int) -> bool:
    if (y := x + 1) > 0:
        return True
    return False
"""
    ir = parse_source(source, "test_walrus")
    assert any("walrus" in w.construct for w in ir.warnings)

    # f-string
    source = """
def greet(name: str) -> str:
    return f"Hello, {name}!"
"""
    ir = parse_source(source, "test_fstring")
    assert any("f-string" in w.construct for w in ir.warnings)

    print("  PASS: test_unsupported_construct_warnings")


def test_strict_mode():
    """Verify strict mode raises ParserError on unsupported constructs."""
    from ast_extractor import parse_source, ParserConfig, ParserError

    source = """
def unsafe_divide(x: int, y: int) -> int:
    try:
        return x // y
    except:
        return 0
"""
    config = ParserConfig(strict_mode=True)
    try:
        parse_source(source, "test_strict", config=config)
        assert False, "Expected ParserError in strict mode"
    except ParserError as e:
        assert "try/except" in str(e)
        print(f"  Strict mode correctly raised: {e}")

    print("  PASS: test_strict_mode")


def test_no_unsupported_warnings_for_valid_code():
    """Verify valid code produces no warnings."""
    from ast_extractor import parse_source

    source = """
import torch

@requires("x > 0")
@ensures("result > 0")
def absolute(x: int) -> int:
    if x < 0:
        return -x
    return x
"""
    ir = parse_source(source, "test_valid")
    assert len(ir.warnings) == 0
    print("  PASS: test_no_unsupported_warnings_for_valid_code")


def test_mutable_object_warnings():
    """Verify mutable object operations generate warnings."""
    from ast_extractor import parse_source

    # attribute assignment
    source = """
class Counter:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count = self.count + 1
"""
    ir = parse_source(source, "test_attr_assign")
    assert any("attribute assignment" in w.construct for w in ir.warnings)

    # subscript assignment
    source = """
def set_first(items: list, val: int):
    items[0] = val
"""
    ir = parse_source(source, "test_subscript_assign")
    assert any("subscript assignment" in w.construct for w in ir.warnings)

    # mutating method call
    source = """
def add_item(items: list, x: int):
    items.append(x)
"""
    ir = parse_source(source, "test_mutating_method")
    assert any("mutating method" in w.construct for w in ir.warnings)

    print("  PASS: test_mutable_object_warnings")


def test_termination_warnings():
    """Verify non-terminating patterns generate warnings."""
    from ast_extractor import parse_source

    # infinite while loop (no break)
    source = """
def poll_until():
    while True:
        pass
"""
    ir = parse_source(source, "test_infinite_loop")
    assert any("infinite loop" in w.construct for w in ir.warnings)

    # while True with break (should NOT warn)
    source = """
def find_first_with_break(items: list, target: int) -> int:
    i = 0
    while True:
        if items[i] == target:
            break
        i = i + 1
    return i
"""
    ir = parse_source(source, "test_break_loop")
    infinite_warnings = [w for w in ir.warnings if "infinite loop" in w.construct]
    assert len(infinite_warnings) == 0

    # recursive function
    source = """
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    ir = parse_source(source, "test_recursion")
    assert any("recursion" in w.construct for w in ir.warnings)

    print("  PASS: test_termination_warnings")


def test_warning_suppression():
    """Verify warnings can be suppressed via config."""
    from ast_extractor import parse_source, ParserConfig

    source = """
def unsafe_divide(x: int, y: int) -> int:
    try:
        return x // y
    except:
        return 0
"""
    config = ParserConfig(collect_warnings=False)
    ir = parse_source(source, "test_suppress", config=config)
    assert len(ir.warnings) == 0
    print("  PASS: test_warning_suppression")


def test_full_pipeline():
    """Run the full Phase 1 pipeline end-to-end."""
    from ast_extractor import parse_source, normalize
    from abstract_interpreter import analyze
    from spec_ingestion import extract_specs

    source = """
import torch
import torch.nn.functional as F
from typing import List

@requires("x > 0")
@ensures("result > 0")
def relu_forward(x: torch.Tensor) -> torch.Tensor:
    result = F.relu(x)
    return result

class SimpleMLP:
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        self.w1 = torch.randn(in_dim, hidden_dim)
        self.w2 = torch.randn(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.w1
        h = F.relu(h)
        out = h @ self.w2
        return out
"""
    print("\n  Step 1: Parsing source...")
    ir = parse_source(source, "full_pipeline")
    assert len(ir.functions) == 1
    assert len(ir.classes) == 1
    assert "torch" in ir.imports
    print(f"    OK - Parsed {len(ir.functions)} functions, {len(ir.classes)} classes, {len(ir.imports)} imports")

    print("\n  Step 2: Normalizing...")
    ir = normalize(ir)
    print(f"    OK - Normalized {len(ir.all_functions)} total functions")

    print("\n  Step 3: Abstract interpretation...")
    state = analyze(ir)
    print(f"    OK - Analyzed {len(state.function_envs)} function environments")
    print(f"    OK - Generated {len(state.shape_facts)} shape facts")
    print(f"    OK - Generated {len(state.type_constraints)} type constraints")

    print("\n  Step 4: Spec ingestion...")
    specs = extract_specs(ir, state)
    print(f"    OK - Extracted {specs.total_count} proof obligations")

    print("\n" + "=" * 50)
    print("FULL PIPELINE RESULTS")
    print("=" * 50)

    for func_name, env in state.function_envs.items():
        print(f"\n  Function: {func_name}")
        for var, val in env.items():
            print(f"    {var}: {val}")

    if state.tensor_ops_metadata:
        print(f"\n  Tensor Operations Detected:")
        for func_name, ops in state.tensor_ops_metadata.items():
            print(f"    {func_name}: {len(ops)} operations")

    print(f"\n  Shape Facts: {len(state.shape_facts)}")
    for fact in state.shape_facts[:5]:
        print(f"    - {fact}")

    print(f"\n  Proof Obligations: {specs.total_count}")
    for ob in specs.all:
        print(f"    [{ob.kind.name}] {ob.predicate[:60]}")

    print("\n  PASS: test_full_pipeline")


def run_all_tests():
    """Run all Phase 1 tests."""
    test_functions = [
        test_ast_extractor_basic,
        test_ast_extractor_tensor_ops,
        test_ast_extractor_conditionals,
        test_ast_extractor_loops,
        test_ast_extractor_classes,
        test_ast_extractor_type_annotations,
        test_abstract_interpreter,
        test_abstract_interpreter_tensors,
        test_spec_ingestion,
        test_unsupported_construct_warnings,
        test_strict_mode,
        test_no_unsupported_warnings_for_valid_code,
        test_warning_suppression,
        test_mutable_object_warnings,
        test_termination_warnings,
        test_full_pipeline,
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("AXIOM ZERO - Phase 1 Parser Pipeline Tests")
    print("=" * 60)

    for test_fn in test_functions:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
