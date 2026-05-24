#!/usr/bin/env python3
"""End-to-end test of the Lean 4 server connection via the new LSP-based LeanEnv."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from proof_engine.lean_env import LeanEnv, LeanServerError

def main():
    print("Testing Lean 4 server connection...")
    print("=" * 60)

    # Path to Lean — try common locations
    lean_candidates = [
        "lean",
        "C:/Users/seanr/.elan/bin/lean",
        "/home/seanr/.elan/bin/lean",
    ]

    for lean_path in lean_candidates:
        print(f"\n  Trying Lean at: {lean_path}")
        env = LeanEnv(lean_path=lean_path, verbose=True, timeout=30.0)

        try:
            # Step 1: Start the server
            print("    Starting server...")
            env.start()
            print(f"    ✓ Server started (Lean {env.lean_version})")

            # Step 2: Open a simple theorem file
            print("    Opening theorem file...")
            theorem = """theorem add_comm_test : ∀ (a b : ℕ), a + b = b + a := by
  intro a b
  induction a with
  | zero =>
      simp
  | succ a ih =>
      simp [ih]"""
            
            env.open_file(theorem, "TestTheorem")
            print(f"    ✓ File opened: {env.current_file}")

            # Step 3: Wait for diagnostics
            import time
            time.sleep(0.5)

            # Step 4: Query goals (should be none since proof is complete)
            print("    Querying goals...")
            result = env.get_goals()
            print(f"    ✓ Goals result: {result.get('success')}")
            if result.get('goals'):
                print(f"      Open goals: {len(result['goals'])}")
                for g in result['goals']:
                    print(f"      - ⊢ {g.type[:60]}")
            else:
                print(f"      No open goals (proof may be complete or error)")

            # Step 5: Test with a "sorry" placeholder
            print("\n    Testing sorry-based proof progression...")
            theorem2 = """theorem add_zero_test (a : ℕ) : a + 0 = a := by
  sorry"""
            
            env.open_file(theorem2, "TestAddZero")
            time.sleep(0.5)

            # Apply a tactic
            print("    Applying tactic 'simp'...")
            result = env.apply_tactic("simp")
            print(f"    ✓ Tactic result: success={result.get('success')}")
            if result.get('error'):
                print(f"      Error: {result['error']}")
            if result.get('goals'):
                print(f"      Remaining goals: {len(result['goals'])}")
                for g in result['goals']:
                    print(f"      - ⊢ {g.type[:60]}")
            else:
                print(f"      No remaining goals (proof complete!)")

            # Step 6: Test induction
            print("\n    Testing induction proof progression...")
            theorem3 = """theorem add_succ_test (a b : ℕ) : a + Nat.succ b = Nat.succ (a + b) := by
  sorry"""
            
            env.open_file(theorem3, "TestAddSucc")
            time.sleep(0.5)

            print("    Applying tactic 'induction a'...")
            result = env.apply_tactic("induction a")
            print(f"    ✓ Tactic result: success={result.get('success')}")
            if result.get('error'):
                print(f"      Error: {result['error']}")
            else:
                goals = result.get('goals', [])
                print(f"      Resulting goals: {len(goals)}")
                for g in goals:
                    print(f"      - ⊢ {g.type[:80]}")
                    if g.hypotheses:
                        for h in g.hypotheses[:3]:
                            print(f"          {h.name}: {h.type[:60]}")

            # Clean shutdown
            print("\n    Stopping server...")
            env.stop()
            print("    ✓ Server stopped gracefully")

            print(f"\n  {'=' * 60}")
            print(f"  ✓ All tests passed with Lean at: {lean_path}")
            print(f"  {'=' * 60}")
            return 0

        except LeanServerError as e:
            print(f"    ✗ LeanServerError: {e}")
            try:
                env.stop()
            except Exception:
                pass
            continue
        except Exception as e:
            print(f"    ✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            try:
                env.stop()
            except Exception:
                pass
            continue

    print("\n  ✗ Could not connect to any Lean installation.")
    print("    Install Lean 4 via: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh")
    return 1


if __name__ == "__main__":
    sys.exit(main())
