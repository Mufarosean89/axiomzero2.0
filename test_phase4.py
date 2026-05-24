"""Phase 4 tests: Python → Lean Compiler pipeline.

Tests IR-to-Lean translation, hole filling, expression translation,
type mapping, and benchmark suite execution.

Run with:
    python test_phase4.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Fix stdout encoding for Windows (cp1252 can't handle Unicode arrows, math symbols)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import traceback

# ── ANSI formatting helpers ─────────────────────────────────────────────────

GREEN = "\033[92m" if sys.stdout.isatty() else ""
RED = "\033[91m" if sys.stdout.isatty() else ""
YELLOW = "\033[93m" if sys.stdout.isatty() else ""
CYAN = "\033[96m" if sys.stdout.isatty() else ""
BOLD = "\033[1m" if sys.stdout.isatty() else ""
RESET = "\033[0m" if sys.stdout.isatty() else ""

# ── Test state ──────────────────────────────────────────────────────────────

passed = 0
failed = 0
test_count = 0


def test(name: str):
    """Begin a named test section."""
    global test_count
    test_count += 1
    print(f"  {CYAN}{name}{RESET} ... ", end="", flush=True)


def ok():
    """Mark the most recent test as passed."""
    global passed
    passed += 1
    print(f"{GREEN}PASS{RESET}")


def fail(msg: str = ""):
    """Mark the most recent test as failed."""
    global failed
    failed += 1
    print(f"{RED}FAIL{RESET}")
    if msg:
        print(f"       {RED}{msg}{RESET}")


def check(condition: bool, msg: str = ""):
    """Assert a condition and record pass/fail."""
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
    from spec_ingestion.obligations import ProofObligation, ObligationKind

    # Type translation
    type_tests = [
        ("int", "ℤ"),
        ("float", "ℝ"),
        ("bool", "Bool"),
        ("str", "String"),
        ("List[int]", "List ℤ"),
        ("Optional[str]", "Option String"),
    ]
    for py_type, expected in type_tests:
        result = compiler.translate_type(py_type)
        check(result == expected, f"translate_type('{py_type}') = '{result}', expected '{expected}'")
        if result != expected:
            return

    # Predicate translation
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
    """Verify the compiler generates valid Lean theorem skeletons."""
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

    check(specs.total_count >= 1, f"Expected >= 1 obligations, got {specs.total_count}")

    compiler = IRToLeanCompiler()
    result = compiler.compile_module(ir, specs, state, "test_absolute")

    check("import Mathlib" in result, "Missing Mathlib import")
    check("theorem " in result, "No theorem found")
    check("by" in result or ";" in result, f"Expected valid proof syntax:\n{result[:500]}...")
    check("absolute" in result, "Theorem name 'absolute' not found")


def test_ir_to_lean_multiple_obligations():
    """Verify multiple obligations generate multiple theorems."""
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

    # 1 postcondition → 1 theorem, with preconditions as hypotheses
    theorem_count = result.count("theorem ")
    check(theorem_count >= 1, f"Expected >= 1 theorem, got {theorem_count}")
    # Preconditions appear as hypotheses (h0, h1) in the theorem, not as separate theorems
    check("h0" in result or "hpre" in result, "Hypothesis not found in theorem")
    check("add_pos" in result, "Function name not found in output")


def test_hole_filler_trivial():
    """Verify hole filler handles trivial obligations."""
    from spec_ingestion.obligations import ProofObligation, ObligationKind
    from lean_compiler.hole_filler import HoleFiller, HoleDifficulty

    filler = HoleFiller()

    ob = ProofObligation(kind=ObligationKind.PRECONDITION, predicate="True")
    difficulty = filler.classify_hole(ob)
    check(difficulty == HoleDifficulty.TRIVIAL, f"Expected TRIVIAL, got {difficulty}")

    result = filler.fill_hole(ob)
    check("trivial" in result or "rfl" in result or "simp" in result,
          f"Expected trivial proof, got:\n{result}")


def test_hole_filler_arithmetic():
    """Verify hole filler handles arithmetic obligations."""
    from spec_ingestion.obligations import ProofObligation, ObligationKind
    from lean_compiler.hole_filler import HoleFiller, HoleDifficulty

    filler = HoleFiller()

    ob = ProofObligation(kind=ObligationKind.PRECONDITION, predicate="x > 0")
    difficulty = filler.classify_hole(ob)
    check(difficulty in (HoleDifficulty.EASY, HoleDifficulty.TRIVIAL),
          f"Expected EASY/TRIVIAL, got {difficulty}")

    result = filler.fill_hole(ob)
    check("simp" in result or "omega" in result,
          f"Expected simp or omega proof, got:\n{result}")

    ob2 = ProofObligation(kind=ObligationKind.POSTCONDITION, predicate="x + 0 == x")
    difficulty2 = filler.classify_hole(ob2)
    check(difficulty2 == HoleDifficulty.EASY, f"Expected EASY, got {difficulty2}")


def test_hole_filler_medium():
    """Verify hole filler handles medium-difficulty obligations."""
    from spec_ingestion.obligations import ProofObligation, ObligationKind
    from lean_compiler.hole_filler import HoleFiller, HoleDifficulty

    filler = HoleFiller()

    ob = ProofObligation(kind=ObligationKind.LOOP_INVARIANT, predicate="total >= 0")
    difficulty = filler.classify_hole(ob)
    check(difficulty == HoleDifficulty.MEDIUM, f"Expected MEDIUM, got {difficulty}")

    ob2 = ProofObligation(kind=ObligationKind.SHAPE_CONDITION, predicate="is_tensor(x)")
    difficulty2 = filler.classify_hole(ob2)
    check(difficulty2 == HoleDifficulty.MEDIUM, f"Expected MEDIUM, got {difficulty2}")


def test_hole_filler_hard():
    """Verify hole filler leaves hard obligations as sorry."""
    from spec_ingestion.obligations import ProofObligation, ObligationKind
    from lean_compiler.hole_filler import HoleFiller, HoleDifficulty

    filler = HoleFiller()

    ob = ProofObligation(kind=ObligationKind.POSTCONDITION, predicate="∀ (x : ℕ), x + 0 = x")
    difficulty = filler.classify_hole(ob)
    check(difficulty == HoleDifficulty.HARD, f"Expected HARD, got {difficulty}")

    result = filler.fill_hole(ob)
    check("sorry" in result, f"Expected sorry for hard hole, got:\n{result}")


def test_hole_filler_end_to_end():
    """Run hole filler on generated Lean code."""
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

    filler = HoleFiller()
    filled = filler.fill_all_holes(lean_code, specs)

    original_holes = lean_code.count("sorry")
    filled_holes = filled.count("sorry")
    check(filled_holes <= original_holes, f"Holes increased: {original_holes} → {filled_holes}")
    check("theorem " in filled, "Theorems removed after filling")
    check("import Mathlib" in filled, "Import removed after filling")


def test_benchmark_suite_basic():
    """Verify benchmark suite problem registration."""
    from lean_compiler.benchmark_suite import BenchmarkSuite

    suite = BenchmarkSuite()
    problems = suite.problems
    check(len(problems) >= 10, f"Expected >= 10 benchmarks, got {len(problems)}")

    level1 = suite.filter(difficulty=1)
    check(len(level1) >= 4, f"Expected >= 4 level-1 benchmarks, got {len(level1)}")

    arithmetic = suite.filter(tag="arithmetic")
    check(len(arithmetic) >= 5, f"Expected >= 5 arithmetic benchmarks, got {len(arithmetic)}")


def test_benchmark_suite_runner():
    """Verify benchmark runner compiles all problems."""
    from lean_compiler.benchmark_suite import BenchmarkSuite
    from lean_compiler import compile as axiom_compile

    suite = BenchmarkSuite()

    def compile_func(source: str, name: str) -> str:
        return axiom_compile(source, name, fill_holes=True)

    results = suite.run_all(compile_func, "test")

    total = len(results)
    success = sum(1 for r in results if r.success)

    check(total > 0, "No benchmarks were run")
    check(success >= total * 0.9, f"Only {success}/{total} benchmarks succeeded")

    print(f"\n       {YELLOW}Benchmark summary: {success}/{total} passed{RESET}")


def test_end_to_end_compile():
    """Verify the end-to-end compile() entry point."""
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
    check("import Mathlib" in result, "Missing Mathlib import")
    check("theorem" in result, "No theorems generated")
    check("absolute" in result, "Function name not found")
    check("by" in result or result.strip().endswith("sorry"),
          "Output doesn't look like valid Lean code")


def test_end_to_end_no_holes():
    """Verify compile with fill_holes=False still produces valid Lean."""
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
    check("import Mathlib" in result, "Missing Mathlib import")
    check("theorem" in result, "No theorems generated")
    check("sign" in result, "Function name not found")
    check(len(result) > 50, "Output too short")


def test_expression_translation():
    """Verify expression IR translation to Lean expressions."""
    from lean_compiler.ir_to_lean import IRToLeanCompiler
    from ast_extractor.ir import ExpressionIR

    compiler = IRToLeanCompiler()

    check(compiler.translate_expression(ExpressionIR.variable("x")) == "x", "Variable 'x'")
    check(compiler.translate_expression(ExpressionIR.constant(42)) == "42", "Constant 42")
    check(compiler.translate_expression(ExpressionIR.constant(True)) == "true", "Constant True")

    left = ExpressionIR.variable("x")
    right = ExpressionIR.constant(1)
    binop = ExpressionIR.binary_op(left, "+", right)
    check("(x + 1)" in compiler.translate_expression(binop), "Binary op x + 1")

    attr = ExpressionIR.attribute(ExpressionIR.variable("x"), "shape")
    check("shape" in compiler.translate_expression(attr), "Attribute x.shape")


def test_type_translation_edge_cases():
    """Verify type translation edge cases."""
    from lean_compiler.ir_to_lean import IRToLeanCompiler

    compiler = IRToLeanCompiler()
    tests = [
        ("int", "ℤ"),
        ("List[int]", "List ℤ"),
        ("Optional[str]", "Option String"),
        ("List[List[int]]", "List List ℤ"),
        ("torch.Tensor", "torch.Tensor"),
        ("", "ℤ"),
    ]
    for py_type, expected in tests:
        result = compiler.translate_type(py_type)
        check(result == expected, f"translate_type('{py_type}') = '{result}', expected '{expected}'")


def test_compile_multi_function():
    """Verify compiling a module with multiple functions."""
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
    # 2 functions, 2 postconditions total → 2 theorems
    theorem_count = result.count("theorem ")
    check(theorem_count >= 2, f"Expected >= 2 theorems for 2 functions, got {theorem_count}")
    check("absolute" in result, "Missing 'absolute' theorems")
    check("add" in result, "Missing 'add' theorems")


def test_compile_no_specs():
    """Verify compiling a module with no specifications."""
    from lean_compiler import compile as axiom_compile

    source = """
def hello(x: int) -> int:
    return x + 1
"""
    result = axiom_compile(source, "no_spec_test")
    check(result.strip() != "", "Empty output for no-spec module")
    check("import Mathlib" in result, "No import for no-spec module")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global passed, failed

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  Axiom Zero - Phase 4 Tests: Python -> Lean Compiler{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")

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

    for name, func in tests:
        print(f"  {BOLD}[{name}]{RESET}")
        try:
            func()
        except Exception as e:
            fail(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        print()

    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"  {BOLD}RESULTS{RESET}")
    print(f"  Total:  {len(tests)}")
    print(f"  Passed: {GREEN}{passed}{RESET}")
    print(f"  Failed: {RED}{failed}{RESET}\n")
    print(f"  {GREEN if failed == 0 else RED}{'All Phase 4 tests passed!' if failed == 0 else f'{failed} test(s) failed.'}{RESET}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
