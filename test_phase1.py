"""
Test suite for Phase 1: Parser Pipeline
Tests the AST extractor, abstract interpreter, and spec ingestion modules end-to-end.
"""

import sys
import os

# Ensure modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_ast_extractor_basic():
    """Test basic AST extraction: functions, assignments, returns."""
    from ast_extractor import parse_source, normalize

    source = """
def add(x: int, y: int) -> int:
    result = x + y
    return result
"""
    ir = parse_source(source, "test_basic")
    assert ir is not None, "IR should not be None"
    assert len(ir.functions) == 1, f"Expected 1 function, got {len(ir.functions)}"
    
    func = ir.functions[0]
    assert func.signature.name == "add", f"Expected 'add', got {func.signature.name}"
    assert len(func.signature.parameters) == 2, f"Expected 2 params, got {len(func.signature.parameters)}"
    assert func.signature.parameters[0].name == "x"
    assert func.signature.parameters[1].name == "y"
    assert func.signature.return_type is not None
    assert func.signature.return_type.name == "int"
    
    # Should have 2 statements: assignment and return
    assert len(func.body) == 2, f"Expected 2 statements, got {len(func.body)}"
    assert func.body[0].stmt_type == "assignment"
    assert func.body[1].stmt_type == "return"
    
    # Test normalization
    normalized = normalize(ir)
    assert normalized is not None
    
    print("  PASS: test_ast_extractor_basic")


def test_ast_extractor_tensor_ops():
    """Test detection of tensor operations."""
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
    
    assert len(ir.imports) >= 2, f"Expected >=2 imports, got {len(ir.imports)}"
    assert "torch" in ir.imports
    
    func = ir.functions[0]
    assert len(func.tensor_operations) > 0, "Should detect tensor operations"
    
    # Check specific tensor ops
    op_kinds = [op.tensor_op_kind.name for op in func.tensor_operations]
    print(f"  Tensor ops detected: {op_kinds}")
    
    print("  PASS: test_ast_extractor_tensor_ops")


def test_ast_extractor_conditionals():
    """Test parsing of if/else conditionals."""
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
    """Test parsing of for/while loops."""
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
    loop_stmt = func.body[1]  # The for loop
    assert loop_stmt.stmt_type == "for"
    assert loop_stmt.iter_var is not None
    assert loop_stmt.iter_var.name == "i"
    assert len(loop_stmt.body) == 1
    
    print("  PASS: test_ast_extractor_loops")


def test_ast_extractor_classes():
    """Test parsing of class definitions."""
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
    assert cls.methods[0].is_method == True
    
    print("  PASS: test_ast_extractor_classes")


def test_ast_extractor_type_annotations():
    """Test parsing of complex type annotations."""
    from ast_extractor import parse_source
    from ast_extractor.ir import TypeAnnotation

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
    """Test the abstract interpreter on a simple function."""
    from ast_extractor import parse_source, normalize, to_ssa
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
    assert state.analysis_complete == True
    
    # Check function was analyzed
    assert "add" in state.function_envs, "Function 'add' should be in environments"
    env = state.function_envs["add"]
    assert "x" in env, "'x' should be in local env"
    assert "y" in env, "'y' should be in local env"
    
    print(f"  Types inferred: x={env.get('x')}, y={env.get('y')}")
    print(f"  Shape facts: {state.shape_facts}")
    print(f"  Type constraints: {state.type_constraints}")
    
    print("  PASS: test_abstract_interpreter")


def test_abstract_interpreter_tensors():
    """Test the abstract interpreter with tensor operations."""
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
    
    # Check tensor operation metadata was captured
    has_tensor_ops = any(
        sig.get("has_tensor_ops", False)
        for sig in state.function_signatures.values()
    )
    assert has_tensor_ops, "Should have tensor operations detected"
    
    print(f"  Function signatures: {state.function_signatures}")
    print(f"  Tensor ops metadata: {state.tensor_ops_metadata}")
    
    print("  PASS: test_abstract_interpreter_tensors")


def test_spec_ingestion():
    """Test spec ingestion with decorator-based specs."""
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
    """Test that unsupported Python constructs generate warnings."""
    from ast_extractor import parse_source, ParserConfig, ParserWarning

    # Test try/except warning
    source = """
def unsafe_divide(x: int, y: int) -> int:
    try:
        return x // y
    except:
        return 0
"""
    ir = parse_source(source, "test_try")
    warnings = ir.warnings
    assert len(warnings) >= 1, f"Expected at least 1 warning, got {len(warnings)}"
    assert any("try/except" in w.construct for w in warnings), \
        f"Expected 'try/except' warning, got: {[str(w) for w in warnings]}"
    print(f"  Warnings found: {len(warnings)}")
    for w in warnings:
        print(f"    - {w}")

    # Test generator warning
    source = """
def get_squares(n: int) -> list:
    return (x * x for x in range(n))
"""
    ir = parse_source(source, "test_gen")
    warnings = ir.warnings
    assert len(warnings) >= 1, f"Expected at least 1 warning, got {len(warnings)}"
    assert any("generator" in w.construct for w in warnings), \
        f"Expected 'generator' warning, got: {[str(w) for w in warnings]}"

    # Test walrus operator warning
    source = """
def process(x: int) -> bool:
    if (y := x + 1) > 0:
        return True
    return False
"""
    ir = parse_source(source, "test_walrus")
    warnings = ir.warnings
    assert len(warnings) >= 1, f"Expected at least 1 warning, got {len(warnings)}"
    assert any("walrus" in w.construct for w in warnings), \
        f"Expected 'walrus' warning, got: {[str(w) for w in warnings]}"

    # Test f-string warning
    source = """
def greet(name: str) -> str:
    return f"Hello, {name}!"
"""
    ir = parse_source(source, "test_fstring")
    warnings = ir.warnings
    assert len(warnings) >= 1, f"Expected at least 1 warning, got {len(warnings)}"
    assert any("f-string" in w.construct for w in warnings), \
        f"Expected 'f-string' warning, got: {[str(w) for w in warnings]}"

    print("  PASS: test_unsupported_construct_warnings")


def test_strict_mode():
    """Test that strict mode raises ParserError on unsupported constructs."""
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
        ir = parse_source(source, "test_strict", config=config)
        assert False, "Expected ParserError in strict mode"
    except ParserError as e:
        assert "try/except" in str(e), f"Expected 'try/except' in error, got: {e}"
        print(f"  Strict mode correctly raised: {e}")

    print("  PASS: test_strict_mode")


def test_no_unsupported_warnings_for_valid_code():
    """Test that valid code produces no warnings."""
    from ast_extractor import parse_source, ParserConfig

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
    # Only warnings we expect are from imports inside functions, not at module level
    warnings = [w for w in ir.warnings]
    assert len(warnings) == 0, f"Expected 0 warnings, got {len(warnings)}: {[str(w) for w in warnings]}"

    print("  PASS: test_no_unsupported_warnings_for_valid_code")


def test_mutable_object_warnings():
    """Test that mutable object operations generate warnings."""
    from ast_extractor import parse_source

    # Test attribute assignment warning (obj.attr = val)
    source = """
class Counter:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count = self.count + 1
"""
    ir = parse_source(source, "test_attr_assign")
    warnings = ir.warnings
    assert len(warnings) >= 2, f"Expected at least 2 warnings, got {len(warnings)}: {[str(w) for w in warnings]}"
    assert any("attribute assignment" in w.construct for w in warnings), \
        f"Expected 'attribute assignment' warning, got: {[str(w) for w in warnings]}"
    print(f"  Attribute assignment warnings: {[str(w) for w in warnings if 'attribute' in w.construct]}")

    # Test subscript assignment warning (list[i] = val)
    source = """
def set_first(items: list, val: int):
    items[0] = val
"""
    ir = parse_source(source, "test_subscript_assign")
    warnings = ir.warnings
    assert len(warnings) >= 1, f"Expected at least 1 warning, got {len(warnings)}"
    assert any("subscript assignment" in w.construct for w in warnings), \
        f"Expected 'subscript assignment' warning, got: {[str(w) for w in warnings]}"
    print(f"  Subscript assignment warnings: {[str(w) for w in warnings if 'subscript' in w.construct]}")

    # Test mutating method call warning
    source = """
def add_item(items: list, x: int):
    items.append(x)
"""
    ir = parse_source(source, "test_mutating_method")
    warnings = ir.warnings
    assert len(warnings) >= 1, f"Expected at least 1 warning, got {len(warnings)}"
    assert any("mutating method" in w.construct for w in warnings), \
        f"Expected 'mutating method' warning, got: {[str(w) for w in warnings]}"
    print(f"  Mutating method warnings: {[str(w) for w in warnings if 'mutating' in w.construct]}")

    print("  PASS: test_mutable_object_warnings")


def test_termination_warnings():
    """Test that non-terminating patterns generate warnings."""
    from ast_extractor import parse_source

    # Test while True with no break
    source = """
def poll_until():
    while True:
        pass
"""
    ir = parse_source(source, "test_infinite_loop")
    warnings = ir.warnings
    assert len(warnings) >= 1, f"Expected at least 1 warning, got {len(warnings)}"
    assert any("infinite loop" in w.construct for w in warnings), \
        f"Expected 'infinite loop' warning, got: {[str(w) for w in warnings]}"
    print(f"  Infinite loop warnings: {[str(w) for w in warnings if 'infinite' in w.construct]}")

    # Test while True with break (should NOT warn)
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
    assert len(infinite_warnings) == 0, \
        f"Expected no 'infinite loop' warnings when break exists, got: {[str(w) for w in infinite_warnings]}"

    # Test recursive function
    source = """
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    ir = parse_source(source, "test_recursion")
    warnings = ir.warnings
    assert len(warnings) >= 1, f"Expected at least 1 warning, got {len(warnings)}"
    assert any("recursion" in w.construct for w in warnings), \
        f"Expected 'recursion' warning, got: {[str(w) for w in warnings]}"
    print(f"  Recursion warnings: {[str(w) for w in warnings if 'recursion' in w.construct]}")

    print("  PASS: test_termination_warnings")


def test_warning_suppression():
    """Test that warnings can be suppressed via config."""
    from ast_extractor import parse_source, ParserConfig, ParserWarning

    source = """
def unsafe_divide(x: int, y: int) -> int:
    try:
        return x // y
    except:
        return 0
"""
    config = ParserConfig(collect_warnings=False)
    ir = parse_source(source, "test_suppress", config=config)
    assert len(ir.warnings) == 0, f"Expected 0 warnings when suppressed, got {len(ir.warnings)}"

    print("  PASS: test_warning_suppression")


def test_full_pipeline():
    """Test the full Phase 1 pipeline end-to-end."""
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
    # Step 1: Parse
    print("\n  Step 1: Parsing source...")
    ir = parse_source(source, "full_pipeline")
    assert len(ir.functions) == 1, f"Expected 1 function, got {len(ir.functions)}"
    assert len(ir.classes) == 1, f"Expected 1 class, got {len(ir.classes)}"
    assert "torch" in ir.imports
    print(f"    OK - Parsed {len(ir.functions)} functions, {len(ir.classes)} classes, {len(ir.imports)} imports")

    # Step 2: Normalize
    print("\n  Step 2: Normalizing...")
    ir = normalize(ir)
    print(f"    OK - Normalized {len(ir.all_functions)} total functions")

    # Step 3: Abstract interpretation
    print("\n  Step 3: Abstract interpretation...")
    state = analyze(ir)
    print(f"    OK - Analyzed {len(state.function_envs)} function environments")
    print(f"    OK - Generated {len(state.shape_facts)} shape facts")
    print(f"    OK - Generated {len(state.type_constraints)} type constraints")

    # Step 4: Spec ingestion
    print("\n  Step 4: Spec ingestion...")
    specs = extract_specs(ir, state)
    print(f"    OK - Extracted {specs.total_count} proof obligations")

    # Step 5: Print summary
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
    test_names = [
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
    
    for test in test_names:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {test.__name__}: {e}")
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
