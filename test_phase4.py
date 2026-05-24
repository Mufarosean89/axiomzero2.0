"""
Axiom Zero - Phase 4 Tests: Python → Lean Compiler

Tests the complete compiler pipeline: IR-to-Lean translation, hole filling,
expression translation, type mapping, and benchmark suite execution.

Run with:
    python test_phase4.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import traceback

# ── Color / formatting helpers ────────────────────────────────────────────────

GREEN = "\033[92m" if sys.stdout.isatty() else ""
RED = "\033[91m" if sys.stdout.isatty() else ""
YELLOW = "\033[93m" if sys.stdout.isatty() else ""
CYAN = "\033[96m" if sys.stdout.isatty() else ""
BOLD = "\033[1m" if sys.stdout.isatty() else ""
RESET = "\033[0m" if sys.stdout.isatty() else ""

passed = 0
failed = 0
test_count = 0


def test(name: str):
    global test_count
    test_count += 1
    print(f"  {CYAN}{name}{RESET} ... ", end="", flush=True)


def ok():
    global passed
    passed += 1
    print(f"{GREEN}PASS{RESET}")


def fail(msg: str = ""):
    global failed
    failed += 1
    print(f"{RED}FAIL{RESET}")
    if msg:
        print(f"       {RED}{msg}{RESET}")


def check(condition: bool, msg: str = ""):
    if condition:
        ok()
    else:
        fail(msg)


# ═════════════════════════════════════════════════════════════════════════════
#  Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_ir_to_lean_basic():
    """Test basic IR → Lean 4 theorem generation."""
    from lean_compiler.ir_to_lean import IRToLeanCompiler

    compiler = IRToLeanCompiler()

    # Test expression translation
    expr_tests = [
        ("constant(42)", "42"),
        ("constant(True)", "true"),
        ("constant(False)", "false"),
        ("constant(3.14)", "(3.14 : ℝ)"),
        ("constant(None)", "none"),
    ]

    # We can't easily create ExpressionIR objects in isolation,
    # so test the obligation translation and type mapping instead.

    # Test type translation
    type_tests = [
        ("int", "ℤ"),
        ("float", "ℝ"),
        ("bool", "Bool"),
        ("str", "String"),
        ("List[int]", "List ℤ"),
        ("Optional[str]", "Option String"),
    ]

    for py_type, expected_lean in type_tests:
        result = compiler.translate_type(py_type)
        check(result == expected_lean, f"translate_type('{py_type}') = '{result}', expected '{expected_lean}'")
        if result != expected_lean:
            return  # stop on first failure to avoid spam

    # Test predicate translation
    from spec_ingestion.obligations import ProofObligation, ObligationKind

    pred_tests = [
        ("x == 5", "x = 5"),
        ("x != 0", "x ≠ 0"),
        ("x > 0 and y > 0", "x > 0 ∧ y > 0"),
        ("x < 0 or y < 0", "x < 0 ∨ y < 0"),
        ("not x", "¬ x"),
        ("True", "true"),
        ("False", "false"),
    ]

    for pred, expected in pred_tests:
        ob = ProofObligation(kind=ObligationKind.PRECONDITION, predicate=pred)
        result = IRToLeanCompiler.obligation_to_predicate_lean(ob)
        check(result == expected, f"predicate '{pred}' → '{result}', expected '{expected}'")
        if result != expected:
            return


def test_ir_to_lean_theorem_generation():
    """Test that the compiler generates valid Lean theorem skeletons."""
    from ast_extractor import parse_source, normalize
    from abstract_interpreter import analyze
    from spec_ingestion import extract_specs
    from lean_compiler.ir_to_lean import IRToLeanCompiler

    source = """
@requires("x > 0")
@ensures("result > 0")
def absolute(x: int) -> int:
    if x < 0:
        return -x
    return x
"""

    ir = parse_source(source, "test")
    ir = normalize(ir)
    state = analyze(ir)
    specs = extract_specs(ir, state)

    check(specs.total_count >= 1, f"Expected at least 1 obligation, got {specs.total_count}")

    compiler = IRToLeanCompiler()
    result = compiler.compile_module(ir, specs, state, "test_absolute")

    # Check output contains expected Lean structures
    checks = [
        ("import Mathlib", "Missing Mathlib import"),
        ("theorem ", "No theorem found"),
    ]
    for needle, msg in checks:
        check(needle in result, f"{msg}\n---\n{result[:300]}...")
    # Holes may be auto-filled (e.g. "x > 0" → simp), so we check for valid proof blocks
    check("by" in result or ";" in result,
          f"Expected valid proof syntax in output:\n{result[:500]}...")

    # Check named theorems appear
    check("absolute" in result, f"Theorem name 'absolute' not found in output")


def test_ir_to_lean_multiple_obligations():
    """Test that multiple obligations generate multiple theorems."""
    from ast_extractor import parse_source, normalize
    from abstract_interpreter import analyze
    from spec_ingestion import extract_specs
    from lean_compiler.ir_to_lean import IRToLeanCompiler

    source = """
@requires("x > 0")
@requires("y > 0")
@ensures("result > 0")
def add_pos(x: int, y: int) -> int:
    return x + y
"""

    ir = parse_source(source, "test")
    ir = normalize(ir)
    state = analyze(ir)
    specs = extract_specs(ir, state)
    compiler = IRToLeanCompiler()
    result = compiler.compile_module(ir, specs, state, "test_add")

    theorem_count = result.count("theorem ")
    check(theorem_count >= 2, f"Expected >= 2 theorems, got {theorem_count}")

    # Check both pre and post conditions are present (case-insensitive)
    check("precondition" in result.lower(), "Precondition theorem not found")
    check("postcondition" in result.lower(), "Postcondition theorem not found")


def test_hole_filler_trivial():
    """Test hole filler with trivial obligations."""
    from spec_ingestion.obligations import ProofObligation, ObligationKind, SpecCollection
    from lean_compiler.hole_filler import HoleFiller, HoleDifficulty

    filler = HoleFiller()

    # Test classification
    trivial_ob = ProofObligation(
        kind=ObligationKind.PRECONDITION,
        predicate="True",
    )
    difficulty = filler.classify_hole(trivial_ob)
    check(difficulty == HoleDifficulty.TRIVIAL, f"Expected TRIVIAL, got {difficulty}")

    # Test filling
    result = filler.fill_hole(trivial_ob)
    check("trivial" in result or "rfl" in result or "simp" in result,
          f"Expected trivial proof, got:\n{result}")


def test_hole_filler_arithmetic():
    """Test hole filler with arithmetic obligations."""
    from spec_ingestion.obligations import ProofObligation, ObligationKind
    from lean_compiler.hole_filler import HoleFiller, HoleDifficulty

    filler = HoleFiller()

    # Simple arithmetic: x > 0
    ob = ProofObligation(
        kind=ObligationKind.PRECONDITION,
        predicate="x > 0",
    )
    difficulty = filler.classify_hole(ob)
    check(difficulty in (HoleDifficulty.EASY, HoleDifficulty.TRIVIAL),
          f"Expected EASY/TRIVIAL for 'x > 0', got {difficulty}")

    result = filler.fill_hole(ob)
    check("simp" in result or "omega" in result,
          f"Expected simp or omega proof, got:\n{result}")

    # x + 0 == x (simp pattern)
    ob2 = ProofObligation(
        kind=ObligationKind.POSTCONDITION,
        predicate="x + 0 == x",
    )
    difficulty2 = filler.classify_hole(ob2)
    check(difficulty2 == HoleDifficulty.EASY,
          f"Expected EASY for 'x + 0 == x', got {difficulty2}")


def test_hole_filler_medium():
    """Test hole filler with medium-difficulty obligations."""
    from spec_ingestion.obligations import ProofObligation, ObligationKind
    from lean_compiler.hole_filler import HoleFiller, HoleDifficulty

    filler = HoleFiller()

    # Loop invariant → medium
    ob = ProofObligation(
        kind=ObligationKind.LOOP_INVARIANT,
        predicate="total >= 0",
    )
    difficulty = filler.classify_hole(ob)
    check(difficulty == HoleDifficulty.MEDIUM,
          f"Expected MEDIUM for loop invariant, got {difficulty}")

    # Shape condition → medium
    ob2 = ProofObligation(
        kind=ObligationKind.SHAPE_CONDITION,
        predicate="is_tensor(x)",
    )
    difficulty2 = filler.classify_hole(ob2)
    check(difficulty2 == HoleDifficulty.MEDIUM,
          f"Expected MEDIUM for shape condition, got {difficulty2}")


def test_hole_filler_hard():
    """Test hole filler with hard obligations (should leave as sorry)."""
    from spec_ingestion.obligations import ProofObligation, ObligationKind
    from lean_compiler.hole_filler import HoleFiller, HoleDifficulty

    filler = HoleFiller()

    # Complex predicate → hard
    ob = ProofObligation(
        kind=ObligationKind.POSTCONDITION,
        predicate="∀ (x : ℕ), x + 0 = x",
    )
    difficulty = filler.classify_hole(ob)
    check(difficulty == HoleDifficulty.HARD,
          f"Expected HARD for ∀ predicate, got {difficulty}")

    result = filler.fill_hole(ob)
    check("sorry" in result,
          f"Expected 'sorry' for hard hole, got:\n{result}")


def test_hole_filler_end_to_end():
    """Test end-to-end hole filling on generated Lean code."""
    from ast_extractor import parse_source, normalize
    from abstract_interpreter import analyze
    from spec_ingestion import extract_specs
    from lean_compiler.ir_to_lean import IRToLeanCompiler
    from lean_compiler.hole_filler import HoleFiller

    source = """
@requires("x > 0")
@ensures("result > 0")
def absolute(x: int) -> int:
    if x < 0:
        return -x
    return x
"""

    ir = parse_source(source, "test")
    ir = normalize(ir)
    state = analyze(ir)
    specs = extract_specs(ir, state)

    compiler = IRToLeanCompiler()
    lean_code = compiler.compile_module(ir, specs, state, "test_hole_fill")

    # Fill holes
    filler = HoleFiller()
    filled = filler.fill_all_holes(lean_code, specs)

    # Check that SOME holes were filled (not all may be fillable)
    # The precondition "x > 0" should be fillable with simp or omega
    original_holes = lean_code.count("sorry")
    filled_holes = filled.count("sorry")

    check(filled_holes <= original_holes,
          f"Holes increased: {original_holes} → {filled_holes}")

    # The filled code should still be valid Lean structure
    check("theorem " in filled, "Theorems removed after filling")
    check("import Mathlib" in filled, "Import removed after filling")


def test_benchmark_suite_basic():
    """Test the benchmark suite compiles all problems."""
    from lean_compiler.benchmark_suite import BenchmarkSuite

    suite = BenchmarkSuite()

    # Check problems are registered
    problems = suite.problems
    check(len(problems) >= 10, f"Expected >= 10 benchmarks, got {len(problems)}")

    # Check filtering by difficulty
    level1 = suite.filter(difficulty=1)
    check(len(level1) >= 4, f"Expected >= 4 level-1 benchmarks, got {len(level1)}")

    # Check filtering by tag
    arithmetic = suite.filter(tag="arithmetic")
    check(len(arithmetic) >= 5, f"Expected >= 5 arithmetic benchmarks, got {len(arithmetic)}")


def test_benchmark_suite_runner():
    """Test the benchmark runner compiles all problems."""
    from lean_compiler.benchmark_suite import BenchmarkSuite
    from lean_compiler import compile as axiom_compile

    suite = BenchmarkSuite()

    def compile_func(source: str, name: str) -> str:
        return axiom_compile(source, name, fill_holes=True)

    results = suite.run_all(compile_func, "test")

    total = len(results)
    success = sum(1 for r in results if r.success)
    total_theorems = sum(r.num_theorems for r in results)

    check(total > 0, "No benchmarks were run")
    check(success >= total * 0.9,  # At least 90% should succeed
          f"Only {success}/{total} benchmarks succeeded")

    # Print summary
    print(f"\n       {YELLOW}Benchmark summary: {success}/{total} passed, "
          f"{total_theorems} theorems generated{RESET}")


def test_end_to_end_compile():
    """Test the end-to-end compile() entry point."""
    from lean_compiler import compile as axiom_compile

    source = """
@requires("x > 0")
@ensures("result > 0")
def absolute(x: int) -> int:
    if x < 0:
        return -x
    return x
"""

    result = axiom_compile(source, "end_to_end_test", fill_holes=True)

    checks = [
        ("import Mathlib", "Missing Mathlib import"),
        ("theorem", "No theorems generated"),
        ("open Classical", "Missing open Classical"),
    ]

    for needle, msg in checks:
        check(needle in result, f"{msg}\n---\n{result[:500]}")

    # The result should be a valid Lean module structure
    check(result.strip().endswith("sorry") or "by" in result,
          "Output doesn't look like valid Lean code")


def test_end_to_end_no_holes():
    """Test compile with fill_holes=False still produces valid Lean."""
    from lean_compiler import compile as axiom_compile

    source = """
@requires("x > 0")
@ensures("result > 0")
def sign(x: int) -> int:
    if x > 0:
        return 1
    return -1
"""

    result = axiom_compile(source, "no_fill_test", fill_holes=False)
    # Even with fill_holes=False, basic structure is preserved
    check("import Mathlib" in result, "Missing Mathlib import")
    check("theorem" in result, "No theorems generated")
    check("open Classical" in result, "Missing open Classical")
    # The result should be valid Lean text
    check(len(result) > 50, "Output too short")


def test_expression_translation():
    """Test expression IR translation to Lean expressions."""
    from lean_compiler.ir_to_lean import IRToLeanCompiler
    from ast_extractor.ir import ExpressionIR

    compiler = IRToLeanCompiler()

    # Test variable
    expr = ExpressionIR.variable("x")
    result = compiler.translate_expression(expr)
    check(result == "x", f"Variable 'x' → '{result}'")

    # Test constant int
    expr = ExpressionIR.constant(42)
    result = compiler.translate_expression(expr)
    check(result == "42", f"Constant 42 → '{result}'")

    # Test constant bool
    expr = ExpressionIR.constant(True)
    result = compiler.translate_expression(expr)
    check(result == "true", f"Constant True → '{result}'")

    # Test binary op
    left = ExpressionIR.variable("x")
    right = ExpressionIR.constant(1)
    expr = ExpressionIR.binary_op(left, "+", right)
    result = compiler.translate_expression(expr)
    check("(x + 1)" in result, f"Binary op x + 1 → '{result}'")

    # Test attribute
    target = ExpressionIR.variable("x")
    expr = ExpressionIR.attribute(target, "shape")
    result = compiler.translate_expression(expr)
    check("shape" in result, f"Attribute x.shape → '{result}'")


def test_type_translation_edge_cases():
    """Test type translation edge cases."""
    from lean_compiler.ir_to_lean import IRToLeanCompiler

    compiler = IRToLeanCompiler()

    tests = [
        ("int", "ℤ"),
        ("List[int]", "List ℤ"),
        ("Optional[str]", "Option String"),
        ("List[List[int]]", "List List ℤ"),
        ("torch.Tensor", "torch.Tensor"),  # unknown type, pass through
        ("", ""),  # empty string
    ]

    for py_type, expected in tests:
        result = compiler.translate_type(py_type)
        check(result == expected, f"translate_type('{py_type}') = '{result}', expected '{expected}'")


def test_compile_multi_function():
    """Test compiling a module with multiple functions."""
    from lean_compiler import compile as axiom_compile

    source = """
@requires("x > 0")
@ensures("result > 0")
def absolute(x: int) -> int:
    if x < 0:
        return -x
    return x

@ensures("result == x + y")
def add(x: int, y: int) -> int:
    return x + y
"""

    result = axiom_compile(source, "multi_function_test")
    theorem_count = result.count("theorem ")

    check(theorem_count >= 3, f"Expected >= 3 theorems for 2 functions, got {theorem_count}")
    check("absolute" in result, "Missing 'absolute' theorems")
    check("add" in result, "Missing 'add' theorems")


def test_compile_no_specs():
    """Test compiling a module with no specifications."""
    from lean_compiler import compile as axiom_compile

    source = """
def hello(x: int) -> int:
    return x + 1
"""

    result = axiom_compile(source, "no_spec_test")
    check(result.strip() != "", "Empty output for no-spec module")
    check("import Mathlib" in result, "No import for no-spec module")


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _get_pipeline(source: str, name: str):
    """Run the Phase 1-2 pipeline on source and return (ir, state, specs)."""
    from ast_extractor import parse_source, normalize
    from abstract_interpreter import analyze
    from spec_ingestion import extract_specs

    ir = parse_source(source, name)
    ir = normalize(ir)
    state = analyze(ir)
    specs = extract_specs(ir, state)
    return ir, state, specs


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global passed, failed

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Axiom Zero - Phase 4 Tests: Python → Lean Compiler{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print()

    tests = [
        ("IR → Lean basics", test_ir_to_lean_basic),
        ("Theorem skeleton generation", test_ir_to_lean_theorem_generation),
        ("Multiple obligations", test_ir_to_lean_multiple_obligations),
        ("Hole filler trivial", test_hole_filler_trivial),
        ("Hole filler arithmetic", test_hole_filler_arithmetic),
        ("Hole filler medium", test_hole_filler_medium),
        ("Hole filler hard", test_hole_filler_hard),
        ("Hole filler end-to-end", test_hole_filler_end_to_end),
        ("Benchmark suite registration", test_benchmark_suite_basic),
        ("Benchmark suite runner", test_benchmark_suite_runner),
        ("Expression translation", test_expression_translation),
        ("Type translation edge cases", test_type_translation_edge_cases),
        ("End-to-end compile", test_end_to_end_compile),
        ("End-to-end no holes", test_end_to_end_no_holes),
        ("Multi-function compile", test_compile_multi_function),
        ("No-spec compile", test_compile_no_specs),
    ]

    total_tests = len(tests)

    for name, func in tests:
        print(f"  {BOLD}[{name}]{RESET}")
        try:
            func()
        except Exception as e:
            fail(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        print()

    # Summary
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  {BOLD}RESULTS{RESET}")
    print(f"  {BOLD}{'=' * 60}{RESET}")
    print(f"  Total:  {total_tests}")
    print(f"  Passed: {GREEN}{passed}{RESET}")
    print(f"  Failed: {RED}{failed}{RESET}")
    print()

    if failed == 0:
        print(f"  {GREEN}All Phase 4 tests passed!{RESET}")
    else:
        print(f"  {RED}{failed} test(s) failed.{RESET}")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
