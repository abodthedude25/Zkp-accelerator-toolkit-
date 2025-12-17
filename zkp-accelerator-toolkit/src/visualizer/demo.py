"""
SumCheck Visualizer Demo

This script demonstrates the SumCheck visualizer with several examples,
from simple to more realistic scenarios.

Run with:
    python -m src.visualizer.demo
"""

import sys
import random
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.field import PrimeField
from src.visualizer.mle import MLETable
from src.visualizer.core import SumCheckVisualizer


def demo_tiny_example():
    """
    Minimal example: 2 variables (4 entries), 2 MLEs.
    
    This is small enough to verify by hand.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: TINY EXAMPLE (2 variables, 2 MLEs)")
    print("=" * 70)
    print("\nThis example is small enough to verify each step by hand.")
    print("f(X1, X2) = a(X1, X2) × b(X1, X2)")
    
    field = PrimeField(97)
    
    # 2 variables → 4 entries
    # a(0,0)=3, a(0,1)=7, a(1,0)=2, a(1,1)=5
    a = MLETable("a", [3, 7, 2, 5], field)
    b = MLETable("b", [1, 4, 6, 2], field)
    
    print(f"\na = {a.values}")
    print(f"b = {b.values}")
    
    # Manual calculation of sum
    print("\nManual sum calculation:")
    print("  f(0,0) = 3 × 1 = 3")
    print("  f(0,1) = 7 × 4 = 28")
    print("  f(1,0) = 2 × 6 = 12")
    print("  f(1,1) = 5 × 2 = 10")
    print("  Sum = 3 + 28 + 12 + 10 = 53")
    
    # Run visualizer
    viz = SumCheckVisualizer([a, b], field, seed=42)
    result = viz.run_full_protocol(verbose=True)
    
    return result


def demo_three_mles():
    """
    Example with 3 variables and 3 MLEs.
    
    f(X) = a(X) × b(X) × c(X)
    """
    print("\n" + "=" * 70)
    print("DEMO 2: THREE MLES (3 variables, 3 MLEs)")
    print("=" * 70)
    print("\nf(X1, X2, X3) = a(X) × b(X) × c(X)")
    print("This is the example used in the README.")
    
    field = PrimeField(97)
    
    # 3 variables → 8 entries
    a = MLETable("a", [3, 7, 2, 5, 1, 8, 4, 6], field)
    b = MLETable("b", [1, 4, 6, 2, 8, 3, 5, 7], field)
    c = MLETable("c", [5, 2, 8, 1, 4, 7, 3, 6], field)
    
    viz = SumCheckVisualizer([a, b, c], field, seed=42)
    result = viz.run_full_protocol(verbose=True)
    
    return result


def demo_larger_example():
    """
    Larger example with 5 variables (32 entries).
    
    Shows how the protocol scales.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: LARGER EXAMPLE (5 variables)")
    print("=" * 70)
    print("\nTable size: 2^5 = 32 entries each")
    print("5 rounds of SumCheck")
    
    field = PrimeField(97)
    
    # Generate random MLEs
    random.seed(123)
    
    a = MLETable("a", [random.randint(1, 96) for _ in range(32)], field)
    b = MLETable("b", [random.randint(1, 96) for _ in range(32)], field)
    
    viz = SumCheckVisualizer([a, b], field, seed=42)
    
    # Run without full verbose (too long)
    print("\nRunning SumCheck (showing summary only)...")
    result = viz.run_full_protocol(verbose=False)
    
    print(f"\nClaimed sum: {result.claimed_sum}")
    print(f"Challenges: {result.challenges}")
    print(f"Final value: {result.final_value}")
    print(f"Verified: {result.verified}")
    
    print("\nRound-by-round summary:")
    for rd in result.round_data:
        tables_before_size = len(rd.tables_before[0][1])
        tables_after_size = len(rd.tables_after[0][1])
        print(f"  Round {rd.round_num}: challenge={rd.challenge:2d}, "
              f"polynomial=[{', '.join(str(v) for v in rd.round_polynomial[:3])}...], "
              f"table_size: {tables_before_size} → {tables_after_size}")
    
    return result


def demo_step_by_step():
    """
    Interactive step-by-step demonstration.
    
    Shows how to use the visualizer for learning.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: STEP-BY-STEP WALKTHROUGH")
    print("=" * 70)
    
    field = PrimeField(97)
    
    # Simple 2-variable example
    a = MLETable("a", [3, 7, 2, 5], field)
    b = MLETable("b", [1, 4, 6, 2], field)
    
    print("\nStarting with simple 2-variable MLEs:")
    print(f"  a = {a.values}")
    print(f"  b = {b.values}")
    
    viz = SumCheckVisualizer([a, b], field)
    
    # Step 1: Compute initial sum
    print("\n" + "-" * 50)
    print("STEP 1: Compute initial sum")
    print("-" * 50)
    
    initial_sum = viz.compute_sum_over_hypercube(viz.current_tables)
    print(f"\nSum over hypercube: {initial_sum}")
    print("\nHow it's computed:")
    for i in range(4):
        bits = f"({(i>>1)&1}, {i&1})"
        product = viz.compute_product_at_index(viz.current_tables, i)
        print(f"  f{bits} = a[{i}] × b[{i}] = {a.values[i]} × {b.values[i]} = {product}")
    
    # Step 2: First round
    print("\n" + "-" * 50)
    print("STEP 2: Execute Round 1")
    print("-" * 50)
    
    # Manually show extension computation
    print("\nComputing extensions for X1 = 0, 1, 2:")
    print("\nFor MLE 'a':")
    print(f"  Pair (0,1): a[0]={a.values[0]}, a[1]={a.values[1]}")
    ext_a0 = a.compute_extension(0, 1, 3)
    print(f"    Extensions: {ext_a0}")
    print(f"    (Formula: {a.values[0]} + k×({a.values[1]}-{a.values[0]}) = {a.values[0]} + k×{field.sub(a.values[1], a.values[0])})")
    
    print(f"\n  Pair (2,3): a[2]={a.values[2]}, a[3]={a.values[3]}")
    ext_a1 = a.compute_extension(2, 3, 3)
    print(f"    Extensions: {ext_a1}")
    
    # Execute round 1
    challenge1 = 23
    rd1 = viz.execute_round(challenge=challenge1, expected_sum=initial_sum)
    
    print(f"\nRound 1 polynomial: {rd1.round_polynomial}")
    print(f"Check: s(0) + s(1) = {rd1.round_polynomial[0]} + {rd1.round_polynomial[1]} = "
          f"{field.add(rd1.round_polynomial[0], rd1.round_polynomial[1])} (should be {initial_sum})")
    
    print(f"\nChallenge: r1 = {challenge1}")
    print(f"\nUpdated tables:")
    for name, values in rd1.tables_after:
        print(f"  {name}: {values}")
    
    # Step 3: Second round
    print("\n" + "-" * 50)
    print("STEP 3: Execute Round 2 (final round)")
    print("-" * 50)
    
    # Expected sum for round 2
    expected_sum2 = viz._evaluate_polynomial_at_challenge(rd1.round_polynomial, challenge1)
    print(f"\nExpected sum (s1({challenge1})): {expected_sum2}")
    
    challenge2 = 45
    rd2 = viz.execute_round(challenge=challenge2, expected_sum=expected_sum2)
    
    print(f"\nRound 2 polynomial: {rd2.round_polynomial}")
    print(f"Challenge: r2 = {challenge2}")
    
    # Step 4: Final verification
    print("\n" + "-" * 50)
    print("STEP 4: Final Verification")
    print("-" * 50)
    
    final_value = viz.compute_product_at_index(viz.current_tables, 0)
    expected_final = viz._evaluate_polynomial_at_challenge(rd2.round_polynomial, challenge2)
    
    print(f"\nFinal evaluation: f({challenge1}, {challenge2}) = {final_value}")
    print(f"Expected (from s2({challenge2})): {expected_final}")
    print(f"\nVerification: {'PASSED ✓' if final_value == expected_final else 'FAILED ✗'}")


def demo_understanding_extensions():
    """
    Deep dive into extension computation.
    
    This is the key operation in each SumCheck round.
    """
    print("\n" + "=" * 70)
    print("DEMO 5: UNDERSTANDING EXTENSIONS")
    print("=" * 70)
    
    field = PrimeField(97)
    
    print("\nExtensions are how we evaluate the polynomial at non-boolean points.")
    print("\nFor a multilinear polynomial that's LINEAR in each variable:")
    print("  f(X) = v0 + X × (v1 - v0)")
    print("\nThis means:")
    print("  f(0) = v0 (the first value)")
    print("  f(1) = v1 (the second value)")
    print("  f(2) = v0 + 2(v1 - v0) = 2v1 - v0")
    print("  f(k) = v0 + k(v1 - v0)")
    
    # Example
    v0, v1 = 10, 30
    diff = field.sub(v1, v0)
    
    print(f"\nExample: v0 = {v0}, v1 = {v1}")
    print(f"Difference: v1 - v0 = {diff}")
    
    print(f"\nExtensions:")
    for k in range(5):
        ext = field.add(v0, field.mul(k, diff))
        print(f"  f({k}) = {v0} + {k} × {diff} = {ext}")
    
    print("\nWhy we need extensions:")
    print("  - The round polynomial has degree = (number of MLEs)")
    print("  - A degree-d polynomial needs d+1 points to be determined")
    print("  - So with 3 MLEs, we compute f(0), f(1), f(2), f(3)")
    print("  - The verifier can then check consistency and evaluate at any point")


def main():
    """Run all demos."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "SUMCHECK VISUALIZER DEMONSTRATION" + " " * 18 + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("\nThis demo shows the SumCheck protocol step by step.")
    print("SumCheck is the core algorithm in HyperPlonk for proving")
    print("that a polynomial sums to a claimed value over the boolean hypercube.")
    print("\n" + "─" * 70)
    
    demos = [
        ("Tiny Example", demo_tiny_example),
        ("Three MLEs", demo_three_mles),
        ("Larger Example", demo_larger_example),
        ("Step by Step", demo_step_by_step),
        ("Understanding Extensions", demo_understanding_extensions),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "─" * 70)
        input("Press Enter to continue to next demo...")
    
    print("\n" + "═" * 70)
    print("DEMOS COMPLETE")
    print("═" * 70)
    print("\nKey takeaways:")
    print("  1. SumCheck reduces O(2^μ) checks to O(μ) rounds")
    print("  2. Each round, prover sends a univariate polynomial")
    print("  3. Verifier checks s(0) + s(1) = claimed sum")
    print("  4. Random challenge fixes one variable, halving table size")
    print("  5. After μ rounds, verifier does one final evaluation check")
    print("\nFor zkSpeed/zkPHIRE context:")
    print("  - Extensions require memory bandwidth (read all MLE entries)")
    print("  - Products require compute (multiplications)")
    print("  - This is why SumCheck is memory-bound for simple polynomials")
    print("  - High-degree polynomials (Jellyfish) shift toward compute-bound")


if __name__ == "__main__":
    main()
