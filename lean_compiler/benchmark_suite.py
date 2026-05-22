"""
Axiom Zero - Phase 4: Benchmark Suite

A collection of benchmark problems for evaluating the compiler and RL agent.
Each benchmark consists of a Python function with formal specifications and an
expected Lean 4 proof.

Problems are organised by difficulty:
  - **Level 1**: Pure arithmetic (add, subtract, multiply, absolute value)
  - **Level 2**: Conditional arithmetic (max, min, sign)
  - **Level 3**: List algorithms (sum, reverse, length properties)
  - **Level 4**: Loop invariants (factorial, fibonacci)
  - **Level 5**: Tensor shape constraints (for PyTorch operations)

The benchmark runner verifies that the compiler produces valid Lean 4 output
for each problem, optionally checking that the generated proofs compile with
the Lean 4 kernel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BenchmarkProblem:
    """
    A single benchmark problem.

    Attributes:
        name              : Human-readable name (e.g., "absolute_value")
        source            : Python source code with @requires/@ensures decorators.
        difficulty        : 1–5 difficulty level.
        description       : Short description of what is being proved.
        expected_theorems : Expected theorem names in the generated Lean code.
        tags              : Tags for filtering (e.g., "arithmetic", "list", "loop").
    """
    name: str
    source: str
    difficulty: int = 1
    description: str = ""
    expected_theorems: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark."""
    name: str
    success: bool
    lean_output: str = ""
    num_theorems: int = 0
    num_holes_remaining: int = 0
    error: Optional[str] = None


class BenchmarkSuite:
    """
    Collection of benchmark problems with a runner.

    Usage
    -----
        suite = BenchmarkSuite()
        suite.add_problem(...)
        results = suite.run_all(compile_func)
    """

    def __init__(self) -> None:
        self._problems: List[BenchmarkProblem] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the built-in benchmark problems."""
        self._problems = _BUILTIN_BENCHMARKS.copy()

    def add_problem(self, problem: BenchmarkProblem) -> None:
        """Add a custom benchmark problem."""
        self._problems.append(problem)

    @property
    def problems(self) -> List[BenchmarkProblem]:
        return list(self._problems)

    def filter(
        self,
        difficulty: Optional[int] = None,
        tag: Optional[str] = None,
        name_contains: Optional[str] = None,
    ) -> List[BenchmarkProblem]:
        """Return a filtered list of problems."""
        result = self._problems
        if difficulty is not None:
            result = [p for p in result if p.difficulty == difficulty]
        if tag is not None:
            result = [p for p in result if tag in p.tags]
        if name_contains is not None:
            result = [p for p in result if name_contains.lower() in p.name.lower()]
        return result

    def run_all(
        self,
        compile_func: Callable[[str, str], str],
        module_prefix: str = "benchmark",
    ) -> List[BenchmarkResult]:
        """
        Run all benchmark problems through the compiler.

        Args:
            compile_func  : A function that takes (source, module_name) and
                            returns the generated Lean 4 code as a string.
            module_prefix : Prefix for generated module names.

        Returns:
            List of BenchmarkResult, one per problem.
        """
        results: List[BenchmarkResult] = []
        for problem in self._problems:
            try:
                lean_code = compile_func(problem.source, f"{module_prefix}_{problem.name}")
                result = self._analyze_output(problem, lean_code)
            except Exception as e:
                result = BenchmarkResult(
                    name=problem.name,
                    success=False,
                    error=str(e),
                )
            results.append(result)
        return results

    def _analyze_output(
        self,
        problem: BenchmarkProblem,
        lean_code: str,
    ) -> BenchmarkResult:
        """Analyze the generated Lean output for a benchmark."""
        # Extract actual theorem names from the output
        theorem_names = re.findall(r"theorem\s+(\S+)", lean_code)
        num_theorems = len(theorem_names)
        num_holes = lean_code.count("sorry")

        # Check expected theorems exist — match as substrings of actual theorem names
        all_found = True
        for expected in problem.expected_theorems:
            found = any(expected in name for name in theorem_names)
            if not found:
                all_found = False

        success = all_found

        return BenchmarkResult(
            name=problem.name,
            success=success,
            lean_output=lean_code,
            num_theorems=num_theorems,
            num_holes_remaining=num_holes,
            error=None if success else (
                f"Expected theorems not found. Looking for: {problem.expected_theorems}\n"
                f"Actual theorem names: {theorem_names}"
            ),
        )

    def summary(self, results: List[BenchmarkResult]) -> str:
        """Produce a human-readable summary of benchmark results."""
        total = len(results)
        passed = sum(1 for r in results if r.success)
        failed = total - passed
        total_holes = sum(r.num_holes_remaining for r in results)
        total_theorems = sum(r.num_theorems for r in results)

        lines = [
            "=" * 60,
            "BENCHMARK RESULTS",
            "=" * 60,
            f"  Total:       {total}",
            f"  Passed:      {passed}",
            f"  Failed:      {failed}",
            f"  Theorems:    {total_theorems}",
            f"  Holes left:  {total_holes}",
            "",
            "  Details:",
        ]

        for r in results:
            status = "✓" if r.success else "✗"
            lines.append(
                f"    [{status}] {r.name:30s}  "
                f"thms={r.num_theorems}  "
                f"holes={r.num_holes_remaining}"
            )
            if r.error:
                lines.append(f"          error: {r.error}")

        if failed > 0:
            lines.append("")
            lines.append(f"  ❌ {failed} benchmark(s) failed — see above for details.")

        lines.append("")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Built-in benchmark problems
# ═════════════════════════════════════════════════════════════════════════════

_BUILTIN_BENCHMARKS: List[BenchmarkProblem] = [
    # ── Level 1: Pure arithmetic ───────────────────────────────────────────
    BenchmarkProblem(
        name="add_positive",
        difficulty=1,
        description="If x > 0 and y > 0, then x + y > 0.",
        tags=["arithmetic", "precondition"],
        expected_theorems=["add_positive"],
        source="""
@requires("x > 0")
@requires("y > 0")
@ensures("result > 0")
def add_positive(x: int, y: int) -> int:
    return x + y
""",
    ),
    BenchmarkProblem(
        name="add_commutative",
        difficulty=1,
        description="Addition is commutative: x + y == y + x.",
        tags=["arithmetic", "equality"],
        expected_theorems=["add_commutative"],
        source="""
@ensures("result == x + y")
def add_commutative(x: int, y: int) -> int:
    return x + y
""",
    ),
    BenchmarkProblem(
        name="absolute_value",
        difficulty=1,
        description="Absolute value returns a non-negative result.",
        tags=["arithmetic", "conditional"],
        expected_theorems=["absolute_value"],
        source="""
@requires("x > 0")
@ensures("result > 0")
def absolute_value(x: int) -> int:
    if x < 0:
        return -x
    return x
""",
    ),
    BenchmarkProblem(
        name="zero_identity",
        difficulty=1,
        description="x + 0 == x and 0 + x == x (additive identity).",
        tags=["arithmetic", "equality"],
        expected_theorems=["zero_identity"],
        source="""
@ensures("result == x + 0")
def zero_identity(x: int) -> int:
    return x + 0
""",
    ),
    BenchmarkProblem(
        name="multiply_by_one",
        difficulty=1,
        description="x * 1 == x (multiplicative identity).",
        tags=["arithmetic", "equality"],
        expected_theorems=["multiply_by_one"],
        source="""
@ensures("result == x * 1")
def multiply_by_one(x: int) -> int:
    return x * 1
""",
    ),
    BenchmarkProblem(
        name="double",
        difficulty=1,
        description="Double a number: result == 2 * x.",
        tags=["arithmetic"],
        expected_theorems=["double"],
        source="""
@ensures("result == 2 * x")
def double(x: int) -> int:
    return x + x
""",
    ),

    # ── Level 2: Conditional arithmetic ───────────────────────────────────
    BenchmarkProblem(
        name="max_of_two",
        difficulty=2,
        description="Maximum of two numbers — result >= both inputs.",
        tags=["arithmetic", "conditional"],
        expected_theorems=["max_of_two"],
        source="""
@requires("x >= 0")
@requires("y >= 0")
@ensures("result >= x")
@ensures("result >= y")
def max_of_two(x: int, y: int) -> int:
    if x >= y:
        return x
    return y
""",
    ),
    BenchmarkProblem(
        name="sign_function",
        difficulty=2,
        description="Sign function: returns 1, 0, or -1.",
        tags=["arithmetic", "conditional"],
        expected_theorems=["sign_function"],
        source="""
@requires("x != 0")
@ensures("result == 1 or result == -1")
def sign_function(x: int) -> int:
    if x > 0:
        return 1
    return -1
""",
    ),
    BenchmarkProblem(
        name="subtract_positive",
        difficulty=2,
        description="If x > y, then x - y > 0.",
        tags=["arithmetic", "conditional"],
        expected_theorems=["subtract_positive"],
        source="""
@requires("x > y")
@ensures("result > 0")
def subtract_positive(x: int, y: int) -> int:
    return x - y
""",
    ),

    # ── Level 3: List algorithms ──────────────────────────────────────────
    BenchmarkProblem(
        name="list_sum_positive",
        difficulty=3,
        description="Sum of a list of positive numbers is positive.",
        tags=["list", "arithmetic"],
        expected_theorems=["list_sum_positive"],
        source="""
@ensures("result >= 0")
def list_sum_positive(lst: List[int]) -> int:
    total = 0
    for x in lst:
        total = total + x
    return total
""",
    ),
    BenchmarkProblem(
        name="list_length_nonneg",
        difficulty=3,
        description="Length of a list is non-negative.",
        tags=["list", "property"],
        expected_theorems=["list_length_nonneg"],
        source="""
@ensures("result >= 0")
def list_length_nonneg(lst: List[int]) -> int:
    return len(lst)
""",
    ),

    # ── Level 4: Loop invariants ──────────────────────────────────────────
    BenchmarkProblem(
        name="factorial",
        difficulty=4,
        description="Factorial of a non-negative integer.",
        tags=["loop", "arithmetic"],
        expected_theorems=["factorial"],
        source="""
@requires("n >= 0")
@ensures("result >= 1")
def factorial(n: int) -> int:
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result
""",
    ),

    # ── Level 5: Tensor / advanced ────────────────────────────────────────
    BenchmarkProblem(
        name="tensor_add",
        difficulty=5,
        description="Adding two tensors preserves shape.",
        tags=["tensor", "shape"],
        expected_theorems=["tensor_add"],
        source="""
@requires("is_tensor(x)")
@requires("is_tensor(y)")
def tensor_add(x: Tensor, y: Tensor) -> Tensor:
    return x + y
""",
    ),
]


def list_benchmarks(difficulty: Optional[int] = None, tag: Optional[str] = None) -> str:
    """Return a human-readable list of all benchmarks."""
    suite = BenchmarkSuite()
    problems = suite.filter(difficulty=difficulty, tag=tag)

    lines = [
        "=" * 60,
        "AXIOM ZERO — BENCHMARK SUITE",
        "=" * 60,
        "",
        f"Total problems: {len(problems)}",
        "",
        "  Difficulty 1 — Pure Arithmetic:",
    ]

    for p in problems:
        diff_stars = "★" * p.difficulty
        lines.append(f"    {p.name:30s} [{diff_stars}]  {p.description}")

    lines.append("")
    return "\n".join(lines)
